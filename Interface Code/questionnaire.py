from fastapi import FastAPI, Form, Request 
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlencode
import uvicorn
import pandas as pd
import joblib
import shap

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and SHAP explainer
model = joblib.load("xgb_model.pkl")
explainer = shap.TreeExplainer(model)

selected_features = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age', 'result', 'autism', 'jaundice', 'relation'
]

# Style block
def css_block():
    return """
    <style>
        body { font-family: Arial; background-color: #f2f2f2; margin: 20px; }
        .container { background: white; padding: 20px; border-radius: 10px; max-width: 700px; margin: auto; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h2 { text-align: center; }
        label { margin-top: 10px; }
        input, select { padding: 5px; margin-top: 5px; width: 100%; }
        button { margin-top: 20px; background-color: #10253b; color: white; padding: 10px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background-color: #011021; }
        
        /* Improved spacing and non-bold radio labels */
        .question { margin-bottom: 15px; }
        .radio-group {
            display: flex;
            gap: 20px;
            margin-top: 5px;
        }
        .radio-group label {
            font-weight: normal;
            display: flex;
            align-items: center;
            gap: 5px;
        }
    </style>
    """


# AQ-10 Questions HTML
def generate_questions_html():
    questions = [
        "I often notice small sounds when others do not.",
        "When I’m reading a story, I find it difficult to work out the characters’ intentions.",
        'I find it easy to "read between the lines" when someone is talking to me.',
        "I usually concentrate more on the whole picture, rather than the small details.",
        "I know how to tell if someone listening to me is getting bored.",
        "I find it easy to do more than one thing at once.",
        "I find it easy to work out what someone is thinking or feeling just by looking at their face.",
        "If there is an interruption, I can switch back to what I was doing very quickly.",
        "I like to collect information about categories of things.",
        "I find it difficult to work out people’s intentions."
    ]
    
    html = ""
    for i, q in enumerate(questions, 1):
            score_name = f"A{i}_Score"
        # Set values directly since it's the same for all questions
            agree_value = "1"
            disagree_value = "0"
        
            html += f"""
        <div class="question">
            <label>{i}. {q}</label>
            <div class="radio-group">
                <label><input type="radio" name="{score_name}" value="{agree_value}" required> Agree</label>
                <label><input type="radio" name="{score_name}" value="{disagree_value}"> Disagree</label>
            </div>
        </div>
        """

    return html

# Page 1: Questionnaire
@app.get("/", response_class=HTMLResponse)
async def questionnaire_form():
    return f"""
    <html><head><title>AQ-10 Autism Questionnaire</title>{css_block()}</head>
    <body><div class="container">
        <h2>AQ-10 Autism Questionnaire</h2>
        <form action="/survey" method="post">
            {generate_questions_html()}
            <button type="submit">Next</button>
        </form>
    </div></body></html>
    """

# Handle AQ-10 form submission and redirect
@app.post("/survey")
async def handle_survey(
    A1_Score: int = Form(...), A2_Score: int = Form(...), A3_Score: int = Form(...),
    A4_Score: int = Form(...), A5_Score: int = Form(...), A6_Score: int = Form(...),
    A7_Score: int = Form(...), A8_Score: int = Form(...), A9_Score: int = Form(...),
    A10_Score: int = Form(...),
):
    params = urlencode({
        "A1_Score": A1_Score, "A2_Score": A2_Score, "A3_Score": A3_Score,
        "A4_Score": A4_Score, "A5_Score": A5_Score, "A6_Score": A6_Score,
        "A7_Score": A7_Score, "A8_Score": A8_Score, "A9_Score": A9_Score,
        "A10_Score": A10_Score
    })
    return RedirectResponse(url=f"/survey-details?{params}", status_code=302)

# Page 2: Additional Info
@app.get("/survey-details", response_class=HTMLResponse)
async def survey_details(request: Request):
    params = request.query_params
    aq_scores = sum([int(params.get(f"A{i}_Score", 0)) for i in range(1, 11)])
    hidden_inputs = ''.join([f'<input type="hidden" name="{key}" value="{value}">' for key, value in params.items()])
    hidden_inputs += f'<input type="hidden" name="result" value="{aq_scores}">'

    relation_options = ["Self", "Parent", "Relative", "Health care professional", "Others"]
    relation_select = '<select name="relation" required>' + ''.join([f'<option value="{opt}">{opt}</option>' for opt in relation_options]) + '</select>'

    return f"""
    <html><head><title>Additional Info</title>{css_block()}</head>
    <body><div class="container">
        <h2>Additional User Information</h2>
        <form action="/predict" method="post">
            {hidden_inputs}
            <label>Age:</label>
            <input type="number" name="age" required><br>

            <label>Gender:</label>
            <select name="gender" required>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="others">Others</option>
            </select><br>

            <label>Autism in immediate family?</label>
            <input type="radio" name="autism" value="1" required> Yes
            <input type="radio" name="autism" value="0"> No<br>

            <label>Jaundice (in infancy)?</label>
            <input type="radio" name="jaundice" value="1" required> Yes
            <input type="radio" name="jaundice" value="0"> No<br>

            <label>Used this app before?</label>
            <input type="radio" name="used_app_before" value="1" required> Yes
            <input type="radio" name="used_app_before" value="0"> No<br>

            <label>Country of Residence:</label>
            <input type="text" name="country_of_res" required><br>

            <label>Relation with Patient:</label>
            {relation_select}<br>

            <button type="submit">Predict</button>
        </form>
    </div></body></html>
    """

# Final Prediction Report
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    try:
        form = await request.form()
        relation_mapping = {
            "Self": 0,
            "Parent": 1,
            "Relative": 2,
            "Health care professional": 3,
            "Others": 4
        }

        numeric_fields = [
            "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
            "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
            "age", "result", "autism", "jaundice", "used_app_before"
        ]

        data = {}
        original_data = {}

        for key in form.keys():
            original_data[key] = form[key]
            if key in numeric_fields:
                data[key] = int(form[key])
            elif key == 'relation':
                data[key] = relation_mapping.get(form[key], 4)  # default to 4 = "Others"
            else:
                pass  # ignore non-numeric fields like gender, country_of_res

        input_df = pd.DataFrame([{key: data[key] for key in selected_features}])

        prediction = model.predict(input_df)[0]
        # prob = model.predict_proba(input_df)[0][1]  # removed model confidence display

        result_text = "Autistic" if prediction == 1 else "Not Autistic"

        # SHAP explanation (still generating shap_result.html)
        shap_values = explainer.shap_values(input_df)
        shap_html_path = "shap_result.html"
        shap.save_html(shap_html_path, shap.force_plot(
            explainer.expected_value, shap_values[0], input_df.iloc[0], matplotlib=False
        ))

        # SHAP table generation (not displayed)
        shap_scores = list(zip(input_df.columns, shap_values[0]))
        sorted_scores = sorted(shap_scores, key=lambda x: abs(x[1]), reverse=True)
        # Not adding to HTML, but you could log or save if needed

        # AQ-10 question interpretations
        questions = [
            "You do notice small sounds when others do not.",
            "You find it difficult to work out the characters’ intentions when reading a story.",
            "You find it easy to \"read between the lines\" when someone is talking to you.",
            "You usually concentrate more on the whole picture, rather than the small details.",
            "You know how to tell if someone listening to you is getting bored.",
            "You find it easy to do more than one thing at once.",
            "You find it easy to work out what someone is thinking or feeling just by looking at their face.",
            "If there is an interruption, you can switch back to what you were doing very quickly.",
            "You like to collect information about categories of things.",
            "You find it difficult to work out people’s intentions."
        ]
        agree_for_1_2_9_10 = [1, 2, 9, 10]

        answers = []
        for i in range(1, 11):
            score = data.get(f"A{i}_Score", 0)
            q = questions[i-1]
            if i in agree_for_1_2_9_10:
                ans = q if score == 1 else q.replace("do ", "do not ")
            else:
                ans = q if score == 0 else q.replace("do ", "do not ")
            answers.append(f"{i}. {ans}")

        gender_text = original_data.get("gender", "Not specified").capitalize()
        autism_text = "Yes" if int(original_data.get("autism", 0)) == 1 else "No"
        jaundice_text = "Yes" if int(original_data.get("jaundice", 0)) == 1 else "No"
        used_app_text = "Yes" if int(original_data.get("used_app_before", 0)) == 1 else "No"
        country_text = original_data.get("country_of_res", "Not specified").capitalize()
        relation_text = original_data.get("relation", "Others")

        answers_html = "<ul>"
        for ans in answers:
            answers_html += f"<li>{ans}</li>"
        answers_html += "</ul>"

        return f"""
        <html>
        <head>
            <title>Prediction Report</title>
            <meta charset="UTF-8">
            {css_block()}
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 700px; margin: auto; padding: 20px; }}
                h2 {{ color: #10253b; }}
                ul {{ padding-left: 20px; }}
                .disclaimer {{ margin-top: 30px; font-style: italic; color: #555; }}
            </style>
        </head>
        <body>
            <h2>Autism Prediction Report</h2>
            <p><strong>Prediction:</strong> <span style="color:{'red' if prediction == 1 else 'green'}">{result_text}</span></p>

            <h3>AQ-10 Responses Summary</h3>
            <p>Based on your responses to the AQ-10 questions:</p>
            {answers_html}

            <h3>Personal Information</h3>
            <ul>
                <li><strong>Age:</strong> {original_data.get("age", "Not specified")}</li>
                <li><strong>Gender:</strong> {gender_text}</li>
                <li><strong>Autism in Family History:</strong> {autism_text}</li>
                <li><strong>Jaundice at Birth:</strong> {jaundice_text}</li>
                <li><strong>Used App Before:</strong> {used_app_text}</li>
                <li><strong>Country of Residence:</strong> {country_text}</li>
                <li><strong>Relation to Individual:</strong> {relation_text}</li>
            </ul>

            <p><a href="/shap" target="_blank">View Visual SHAP Explanation</a></p>

            <p class="disclaimer">
                <strong>Important Note:</strong> This assessment is based on predictive modeling and is not a definitive diagnosis.
                For a comprehensive evaluation, it is recommended to consult a licensed professional.
            </p>
        </body>
        </html>
        """

    except Exception as e:
        return HTMLResponse(content=f"<h2>Internal Server Error</h2><p>{str(e)}</p>", status_code=500)

from fastapi.responses import FileResponse
import os

@app.get("/shap", response_class=HTMLResponse)
async def serve_shap_html():
    shap_html_path = "shap_result.html"
    if os.path.exists(shap_html_path):
        return FileResponse(shap_html_path, media_type='text/html')
    else:
        return HTMLResponse("<h3>SHAP explanation not available.</h3>", status_code=404)

# # Final Prediction Report
# @app.post("/predict", response_class=HTMLResponse)
# async def predict(request: Request):
#     try:
#         form = await request.form()
#         relation_mapping = {
#             "Self": 0,
#             "Parent": 1,
#             "Relative": 2,
#             "Health care professional": 3,
#             "Others": 4
#         }

#         numeric_fields = [
#             "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
#             "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
#             "age", "result", "autism", "jaundice", "used_app_before"
#         ]

#         data = {}
#         original_data = {}

#         for key in form.keys():
#             original_data[key] = form[key]
#             if key in numeric_fields:
#                 data[key] = int(form[key])
#             elif key == 'relation':
#                 data[key] = relation_mapping.get(form[key], 4)  # default to 4 = "Others"
#             else:
#                 pass  # ignore non-numeric fields like gender, country_of_res

#         # Prepare input dataframe for prediction with selected features
#         input_df = pd.DataFrame([{key: data[key] for key in selected_features}])

#         # Predict
#         prediction = model.predict(input_df)[0]
#         prob = model.predict_proba(input_df)[0][1]
#         result_text = "Autistic" if prediction == 1 else "Not Autistic"

#         # SHAP explanation
#         shap_values = explainer.shap_values(input_df)
#         shap_html_path = "shap_result.html"
#         shap.save_html(shap_html_path, shap.force_plot(
#             explainer.expected_value, shap_values[0], input_df.iloc[0], matplotlib=False
#         ))

#         # SHAP Feature Importance Table
#         shap_scores = list(zip(input_df.columns, shap_values[0]))
#         sorted_scores = sorted(shap_scores, key=lambda x: abs(x[1]), reverse=True)
#         importance_html = "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse;'>"
#         importance_html += "<tr><th>Feature</th><th>SHAP Value</th></tr>"
#         for feature, value in sorted_scores:
#             importance_html += f"<tr><td>{feature}</td><td>{round(value, 4)}</td></tr>"
#         importance_html += "</table>"

#         # Natural language AQ-10 questions
#         questions = [
#             "You do notice small sounds when others do not.",
#             "You find it difficult to work out the characters’ intentions when reading a story.",
#             "You find it easy to \"read between the lines\" when someone is talking to you.",
#             "You usually concentrate more on the whole picture, rather than the small details.",
#             "You know how to tell if someone listening to you is getting bored.",
#             "You find it easy to do more than one thing at once.",
#             "You find it easy to work out what someone is thinking or feeling just by looking at their face.",
#             "If there is an interruption, you can switch back to what you were doing very quickly.",
#             "You like to collect information about categories of things.",
#             "You find it difficult to work out people’s intentions."
#         ]
#         agree_for_1_2_9_10 = [1, 2, 9, 10]

#         answers = []
#         for i in range(1, 11):
#             score = data.get(f"A{i}_Score", 0)
#             q = questions[i-1]
#             if i in agree_for_1_2_9_10:
#                 ans = q if score == 1 else q.replace("do ", "do not ")
#             else:
#                 ans = q if score == 0 else q.replace("do ", "do not ")
#             answers.append(f"{i}. {ans}")

#         # Compose user info summary (some keys from original_data)
#         gender_text = original_data.get("gender", "Not specified").capitalize()
#         autism_text = "Yes" if int(original_data.get("autism", 0)) == 1 else "No"
#         jaundice_text = "Yes" if int(original_data.get("jaundice", 0)) == 1 else "No"
#         used_app_text = "Yes" if int(original_data.get("used_app_before", 0)) == 1 else "No"
#         country_text = original_data.get("country_of_res", "Not specified").capitalize()
#         relation_text = original_data.get("relation", "Others")

#         # Compose answers HTML list
#         answers_html = "<ul>"
#         for ans in answers:
#             answers_html += f"<li>{ans}</li>"
#         answers_html += "</ul>"

#         # Final HTML
#         return f"""
#         <html>
#         <head>
#             <title>Prediction Report</title>
#             <meta charset="UTF-8">
#             {css_block()}
#             <style>
#                 body {{ font-family: Arial, sans-serif; max-width: 700px; margin: auto; padding: 20px; }}
#                 h2 {{ color: #10253b; }}
#                 ul {{ padding-left: 20px; }}
#                 table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
#                 th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
#                 th {{ background-color: #f2f2f2; }}
#                 .disclaimer {{ margin-top: 30px; font-style: italic; color: #555; }}
#             </style>
#         </head>
#         <body>
#             <h2>Autism Prediction Report</h2>
#             <p><strong>Prediction:</strong> <span style="color:{'red' if prediction == 1 else 'green'}">{result_text}</span></p>
#             <p><strong>Model Confidence:</strong> {prob * 100:.2f}%</p>

#             <h3>AQ-10 Responses Summary</h3>
#             <p>Based on your responses to the AQ-10 questions:</p>
#             {answers_html}

#             <h3>Personal Information</h3>
#             <ul>
#                 <li><strong>Age:</strong> {original_data.get("age", "Not specified")}</li>
#                 <li><strong>Gender:</strong> {gender_text}</li>
#                 <li><strong>Autism in Family History:</strong> {autism_text}</li>
#                 <li><strong>Jaundice at Birth:</strong> {jaundice_text}</li>
#                 <li><strong>Used App Before:</strong> {used_app_text}</li>
#                 <li><strong>Country of Residence:</strong> {country_text}</li>
#                 <li><strong>Relation to Individual:</strong> {relation_text}</li>
#             </ul>

#             <h3>Top Feature Contributions (SHAP)</h3>
#             {importance_html}

#             <p><a href="/shap" target="_blank">View Visual SHAP Explanation</a></p>

#             <p class="disclaimer">
#                 <strong>Important Note:</strong> This assessment is based on predictive modeling and is not a definitive diagnosis.
#                 For a comprehensive evaluation, it is recommended to consult a licensed professional./
