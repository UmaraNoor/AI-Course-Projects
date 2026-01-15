pip install pandas numpy scikit-learn matplotlib gradio lime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif
from lime.lime_tabular import LimeTabularExplainer

# Load dataset
df = pd.read_csv("synthetic_heart_disease_dataset.csv")

# Encode categorical features (excluding target and probability)
df_encoded = pd.get_dummies(df.drop(columns=['HeartDisease', 'DiseaseProbability(%)']), drop_first=True)
X = df_encoded
y = df['HeartDisease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Evaluate models
def evaluate_models(X_train, X_test, y_train, y_test, models):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0)
        }
    return results

# Initial evaluation
results = evaluate_models(X_train, X_test, y_train, y_test, models)
df_results = pd.DataFrame(results).T

# Feature selection using Information Gain
info_gain = mutual_info_classif(X, y)
ig_scores = pd.Series(info_gain, index=X.columns).sort_values(ascending=False)
selected_features = ig_scores[ig_scores > 0.01].index.tolist()

# Dataset with selected features
X_selected = X[selected_features]
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Re-train best model (Random Forest assumed best for UI)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_sel, y_train_sel)

# LIME Explainer
explainer = LimeTabularExplainer(X_train_sel.values, feature_names=X_selected.columns.tolist(),
                                 class_names=["No Disease", "Disease"], mode="classification")

# Health suggestion logic
def get_suggestions(inputs):
    suggestions = []
    if inputs["SmokingHabit"] == 1:
        suggestions.append("🚭 Stop smoking immediately.")
    if inputs["AlcoholConsumption"] == 1:
        suggestions.append("🍷 Limit alcohol consumption.")
    if inputs["ExerciseFrequency"] < 3:
        suggestions.append("🏃‍♂️ Exercise at least 3 times/week.")
    if inputs["DietType"] == 0:
        suggestions.append("🥗 Eat more fruits, veggies, avoid fried food.")
    if inputs["StressLevel"] == "High":
        suggestions.append("🧘 Practice stress management.")
    if inputs["SleepDuration"] < 6:
        suggestions.append("🛌 Get at least 7 hours of sleep.")
    if inputs["BMI"] > 29:
        suggestions.append("⚖️ Consider gradual weight loss.")
    if inputs["FamilyHistory"] == 1:
        suggestions.append("🧬 Family history noted. Regular checkups recommended.")
    if not suggestions:
        suggestions.append("✅ You're following a healthy lifestyle. Keep it up!")
    return suggestions

def doctor_advice(prediction, prob):
    if prediction == 1 and prob > 70:
        return "⚠️ High risk! See a cardiologist immediately."
    elif prediction == 0 and prob < 30:
        return "✅ Low risk. Keep it up!"
    else:
        return "📋 Moderate risk. Consider a medical check-up."

# Gradio prediction function
def predict_heart_disease(Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG,
                          MaxHR, ExerciseAngina, SmokingHabit, AlcoholConsumption, ExerciseFrequency,
                          DietType, StressLevel, SleepDuration, FamilyHistory, BMI):
    try:
        input_dict = {
            "Age": Age, "Sex": Sex, "ChestPainType": ChestPainType, "RestingBP": RestingBP,
            "Cholesterol": Cholesterol, "FastingBS": FastingBS, "RestingECG": RestingECG,
            "MaxHR": MaxHR, "ExerciseAngina": ExerciseAngina, "SmokingHabit": SmokingHabit,
            "AlcoholConsumption": AlcoholConsumption, "ExerciseFrequency": ExerciseFrequency,
            "DietType": DietType, "StressLevel": StressLevel, "SleepDuration": SleepDuration,
            "FamilyHistory": FamilyHistory, "BMI": BMI
        }

        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=X_selected.columns, fill_value=0)

        prediction = model.predict(input_encoded)[0]
        prob = model.predict_proba(input_encoded)[0][1] * 100

        suggestions = get_suggestions(input_dict)
        advice = doctor_advice(prediction, prob)

        exp = explainer.explain_instance(input_encoded.values[0], model.predict_proba, num_features=10)
        fig = exp.as_pyplot_figure()
        lime_path = "lime_explanation.png"
        plt.savefig(lime_path)
        plt.close()

        lime_explanation = "🔍 Explanation (LIME):\n"
        for feature, weight in exp.as_list():
            direction = "🟥 Increases risk" if weight > 0 else "🟩 Decreases risk"
            lime_explanation += f"- {feature}: {direction}\n"

        return (
            f"Prediction: {'❤️ Has Heart Disease' if prediction == 1 else '💚 No Heart Disease'}\nProbability: {prob:.2f}%",
            "\n".join(suggestions),
            advice + "\n\n" + lime_explanation,
            lime_path
        )
    except Exception as e:
        return f"Error: {str(e)}", "Suggestion Error", "Advice Error", None

# Gradio UI
interface = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Age", value=40),
        gr.Radio([0, 1], label="Sex (0=Female, 1=Male)"),
        gr.Dropdown(["TA", "ATA", "NAP", "ASY"], label="Chest Pain Type"),
        gr.Number(label="Resting BP", value=120),
        gr.Number(label="Cholesterol", value=200),
        gr.Radio([0, 1], label="Fasting Blood Sugar > 120 mg/dl?"),
        gr.Dropdown(["Normal", "ST", "LVH"], label="Resting ECG"),
        gr.Number(label="Max Heart Rate", value=150),
        gr.Radio(["Y", "N"], label="Exercise-induced Angina"),
        gr.Radio([0, 1], label="Smoking Habit"),
        gr.Radio([0, 1], label="Alcohol Consumption"),
        gr.Slider(0, 7, step=1, label="Exercise Frequency (days/week)"),
        gr.Radio([0, 1], label="Diet Type (0=Unhealthy, 1=Healthy)"),
        gr.Dropdown(["Low", "Medium", "High"], label="Stress Level"),
        gr.Slider(4.0, 9.0, step=0.1, label="Sleep Duration (hours)"),
        gr.Radio([0, 1], label="Family History"),
        gr.Number(label="BMI", value=25)
    ],
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Textbox(label="Health Suggestions"),
        gr.Textbox(label="Doctor's Advice + LIME Explanation"),
        gr.Image(label="LIME Explanation Image")
    ],
    title="🫀 Smart Heart Disease Risk Predictor",
    description="AI-based early detection of heart risk with lifestyle suggestions and LIME explanations"
)

# Run the app
interface.launch()