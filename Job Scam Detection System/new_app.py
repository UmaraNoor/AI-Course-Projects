import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model, vectorizer, and expected features
model = joblib.load("best_model_random_forest.pkl")
vectorizer = joblib.load("vectorizer.pkl")
expected_cols = joblib.load("expected_features.pkl")

# Page settings
st.set_page_config(page_title="Job Scam Detector", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
            padding: 2rem;
            border-radius: 10px;
        }
        .stTextInput>div>div>input,
        .stTextArea>div>textarea,
        .stSelectbox>div>div>div>div {
            border: 1px solid #d0d3d6;
            border-radius: 8px;
            padding: 0.5rem;
        }
        .stButton>button {
            background-color: #3366cc;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #254a9b;
        }
        .stMarkdown h2, .stMarkdown h3 {
            color: #1f4e79;
        }
        .result-box {
            background-color: #ffffff;
            border-left: 6px solid #4a90e2;
            padding: 1rem;
            margin-top: 1rem;
            border-radius: 8px;
        }
        footer {
            margin-top: 50px;
            text-align: center;
            font-size: 0.85rem;
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

# App Title and Tagline
st.title("🔍 Job Scam Detector")
st.markdown("##### Because not every job that sounds too good to be true... *is real.*")

st.markdown("Enter the job posting details below. The system will analyze and predict whether it's **Legitimate** or a **Scam**, and provide reasoning.")

# Input Form
with st.container():
    st.markdown("### 📝 Job Details")
    title = st.text_input("Job Title")
    company_profile = st.text_area("Company Profile", height=100)
    description = st.text_area("Job Description", height=100)
    requirements = st.text_area("Requirements", height=100)
    benefits = st.text_area("Benefits", height=100)

    st.markdown("### ⚙️ Job Settings")
    telecommuting = st.radio("Is Telecommuting?", ["Yes", "No"], horizontal=True)
    has_company_logo = st.radio("Company Logo Present?", ["Yes", "No"], horizontal=True)
    has_questions = st.radio("Has Screening Questions?", ["Yes", "No"], horizontal=True)

# Predict Button
if st.button("🔎 Predict"):
    combined_text = f"{title} {company_profile} {description} {requirements} {benefits}"
    text_vector = vectorizer.transform([combined_text])
    text_vector_df = pd.DataFrame(text_vector.toarray(), columns=vectorizer.get_feature_names_out())

    suspicious_phrases = any(word in combined_text.lower() for word in [
        "quick money", "no experience", "immediate start", "investment", "limited time", "work from home"
    ])
    short_description = len(description.strip()) < 50
    missing_logo = has_company_logo == "No"
    no_questions = has_questions == "No"

    extra_data = pd.DataFrame([{
        "telecommuting": 1 if telecommuting == "Yes" else 0,
        "has_company_logo": 1 if has_company_logo == "Yes" else 0,
        "has_questions": 1 if has_questions == "Yes" else 0,
        "suspicious_phrases": int(suspicious_phrases),
        "short_description": int(short_description),
        "missing_logo": int(missing_logo),
        "no_questions": int(no_questions)
    }])

    final_input = pd.concat([extra_data.reset_index(drop=True), text_vector_df.reset_index(drop=True)], axis=1)

    for col in expected_cols:
        if col not in final_input:
            final_input[col] = 0
    final_input = final_input[expected_cols]

    prediction = model.predict(final_input)[0]
    label = "🚨 Scam" if prediction == 1 else "✅ Legitimate"

    explanation_points = []
    if missing_logo:
        explanation_points.append("• Missing company logo is suspicious.")
    if no_questions:
        explanation_points.append("• Lack of screening questions may indicate fraud.")
    if suspicious_phrases:
        explanation_points.append("• Contains common scam phrases like 'quick money', 'no experience'.")
    if short_description:
        explanation_points.append("• Job description is too short for a legit job.")

    explanation = "\n".join(explanation_points) if explanation_points else "• Job details appear standard and professional."

    # Output result
    st.markdown(f"""
        <div class="result-box">
            <h3>Prediction: {label}</h3>
            <p><strong>Reason:</strong><br>{explanation}</p>
        </div>
    """, unsafe_allow_html=True)

# Footer note
st.markdown("""<footer>🔐 Built to help you stay safe in the job hunt. Stay smart, stay secure.</footer>""", unsafe_allow_html=True)
