# 🌸 BloomWell – AI-Powered Mental Health Screening & Support System

**BloomWell** is a mental wellness screening web application that uses artificial intelligence to help individuals identify signs of depression and anxiety, and then provides therapeutic support through a conversation-like interface. It blends clinical screening tools with machine learning and LLMs to offer a seamless, private, and supportive experience.

---

## 💡 What is BloomWell?

Mental health issues like depression and anxiety often go undetected or unspoken. BloomWell aims to **bridge that gap using AI**, providing users with an accessible and non-judgmental way to:

1. **Self-assess their mental health** using trusted questionnaires (PHQ-9 and GAD-7).
2. **Get instant predictions** of their depression and anxiety levels using a trained machine learning model.
3. **Engage in a therapeutic AI chat** that responds empathetically based on their scores, offering conversation and self-help strategies.

This system can be used by:
- Psychology students or researchers
- AI/ML learners
- People interested in digital mental health
- Educational institutions for demos

> **Disclaimer**: BloomWell is a prototype for academic purposes and is **not a substitute for clinical diagnosis or treatment**.

---

## 🤖 How the AI Works

BloomWell uses two layers of intelligence:

### 1. **Machine Learning Prediction**
- Uses **Random Forest** classifiers trained on a labeled dataset of questionnaire responses.
- Input: PHQ-9 (9 depression questions), GAD-7 (7 anxiety questions), and lifestyle factors like sleep and activity.
- Output: Mental health severity levels such as:
  - No Depression / Mild Anxiety
  - Moderate Depression / Severe Anxiety
- Predictions are based on pattern recognition learned from synthetic but structured data.

### 2. **Therapeutic LLM Chat**
- After prediction, the user enters a **chat interface** powered by an AI language model (LLaMA).
- The chatbot acts like a therapist: validating feelings, asking clarifying questions, and gently offering self-care advice based on the user’s profile.

---

## 🖥️ How to Run BloomWell on Your Desktop

Here’s how **anyone** can run BloomWell on their own computer (Windows/macOS/Linux):

### ✅ Step 1: Make Sure You Have Python Installed

BloomWell requires **Python 3.8+**.  
To check, open a terminal/command prompt and run:

```bash
python --version
```

If you don’t have Python, download it from: https://www.python.org/downloads/

---

### ✅ Step 2: Download the BloomWell Project

- Either clone the GitHub repository (after it’s uploaded there), or
- If it’s a folder on your computer, **unzip it** or open the project folder.

---

## 🔗 Download Model Files

Due to GitHub's 100MB file size limit, the trained model files are hosted externally:

🔗 [Click here to download all model files (.pkl)](https://drive.google.com/drive/folders/12FwrMCcBHvwvhW9qj4sVitkTFpLcegLY?usp=sharing)

This folder contains:
- `rf_anxiety_model.pkl`
- `rf_depression_model.pkl`
- `le_anxiety.pkl`
- `le_depression.pkl`

📂 After downloading, place **all 4 files** inside the local `models/` folder before running the app.


---

### ✅ Step 3: Open a Terminal in the Project Folder

For example:

```bash
cd Downloads/BloomWell
```

Or right-click inside the folder and choose **“Open Terminal”** or **“Open PowerShell/Command Prompt here”**.

---

### ✅ Step 4: Set Up a Virtual Environment (Recommended)

This keeps dependencies clean.

```bash
python -m venv venv
```

Activate it:

- On **Windows**:
  ```bash
  venv\Scripts\activate
  ```

- On **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

---

### ✅ Step 5: Install the Required Libraries

```bash
pip install -r requirements.txt
```

This will install Flask, scikit-learn, pandas, etc.

---

### ✅ Step 6: Run the Web Application

```bash
python app.py
```

If everything works, you will see:

```
Running on http://127.0.0.1:5000/
```

---

### ✅ Step 7: Open in Your Browser

Visit this address in your browser:

```
http://127.0.0.1:5000/
```

Now you can:
- Fill out the form
- See your AI-based prediction
- Chat with the therapy assistant

---

## 🔐 Ethics & Responsibility

While BloomWell is inspired by mental health needs, it is not a clinical tool. Any serious mental health concern should always be discussed with a licensed professional.

---

Created with care for academic learning 💙
