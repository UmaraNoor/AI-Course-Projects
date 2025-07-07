import streamlit as st
import pandas as pd
import re
import tempfile
import os
from pypdf import PdfReader
import docx
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt  # Required for background_gradient

# ======================
# CUSTOM CSS FOR BACKGROUND
# ======================

def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .main {{
            background-color: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 15px;
            margin: 2rem 0;
        }}
        .sidebar .sidebar-content {{
            background-color: rgba(255, 255, 255, 0.9);
        }}
        .stRadio > div {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 10px;
        }}
        .stDataFrame {{
            background-color: rgba(255, 255, 255, 0.8) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ======================
# 1. DATASET LOADING & PROCESSING
# ======================

def load_requirements_dataset(file_path):
    """Load the requirements dataset from Excel"""
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # Check if required columns exist
        required_cols = ['Functional Requirement', 'Non-Functional Requirement', 'Constraint']
        if not all(col in df.columns for col in required_cols):
            st.error("Requirements dataset must contain columns: " + ", ".join(required_cols))
            return None
            
        # Create lists for each requirement type
        functional_reqs = df['Functional Requirement'].dropna().tolist()
        non_functional_reqs = df['Non-Functional Requirement'].dropna().tolist()
        constraints = df['Constraint'].dropna().tolist()
        
        # Create the combined dataset
        requirements_data = {
            "text": functional_reqs + non_functional_reqs + constraints,
            "label": (["functional"] * len(functional_reqs) + 
                     ["non-functional"] * len(non_functional_reqs) + 
                     ["constraint"] * len(constraints))
        }
        
        return pd.DataFrame(requirements_data)
        
    except Exception as e:
        st.error(f"Error loading requirements dataset: {str(e)}")
        return None

def load_test_case_dataset(file_path):
    """Load the test case dataset from Excel"""
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # Check if required columns exist
        if 'Requirement' not in df.columns or 'Test Case' not in df.columns:
            st.error("Test Case dataset must contain columns: 'Requirement' and 'Test Case'")
            return None
        
        return df
    
    except Exception as e:
        st.error(f"Error loading test case dataset: {str(e)}")
        return None

# ======================
# 2. CORE FUNCTIONS
# ======================

def initialize_model(requirements_df):
    """Create and train the requirements classifier"""
    X_train, X_test, y_train, y_test = train_test_split(
        requirements_df['text'], requirements_df['label'], test_size=0.2, random_state=42
    )
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    st.write("Model Evaluation:")
    st.text(classification_report(y_test, y_pred))
    
    return model

def extract_requirements(text):
    """Extract requirement-like sentences from text"""
    patterns = [
        r'(?:The system shall|The software will|The application must|Users can|The \w+ shall|The \w+ must|The \w+ will) .+?[\.;]',
        r'(?:It is required that|It is necessary to|The \w+ should) .+?[\.;]',
        r'(?:shall|must|will|should|can|may not) .+?[\.;]'
    ]
    
    requirements = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        requirements.extend(matches)
    
    return requirements if requirements else None

def generate_test_cases(requirement, test_case_df):
    """Generate test cases for a functional requirement using the test case dataset"""
    # Find similar requirements in the test case dataset
    similar_cases = test_case_df[
        test_case_df['Requirement'].str.contains(requirement[:30], case=False, na=False)
    ].head(3)  # Get top 3 most similar
    
    if not similar_cases.empty:
        return similar_cases['Test Case'].tolist()
    else:
        # Fallback template if no similar cases found
        operation = requirement.split('shall')[-1].split('must')[-1].split('will')[-1].strip()
        operation = re.sub(r'^be\s+', '', operation)
        
        return [
            f"Verify {operation} functionality works as expected",
            f"Test error handling for {operation}",
            f"Validate performance of {operation}",
            f"Check security requirements for {operation}",
            f"Verify edge cases for {operation}"
        ]

def process_uploaded_file(uploaded_file):
    """Extract text from uploaded file"""
    file_ext = uploaded_file.name.split(".")[-1].lower()
    text = ""
    
    if file_ext == "pdf":
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            reader = PdfReader(tmp.name)
            text = "\n".join([page.extract_text() for page in reader.pages])
            os.unlink(tmp.name)
    elif file_ext == "docx":
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:  # txt
        text = uploaded_file.getvalue().decode("utf-8")
    
    return text

# ======================
# 3. STREAMLIT UI
# ======================

def main():
    # Set page config
    st.set_page_config(
        page_title="AI Requirements Processor",
        page_icon="🤖",
        layout="wide"
    )
    
    # Set background image
    set_background("https://img.freepik.com/free-photo/technology-human-touch-background-modern-remake-creation-adam_53876-129794.jpg")
    
    # Main container with styling
    with st.container():
        st.markdown("""
        <div style='background-color: rgba(255, 255, 255, 0.8); 
                    padding: 2rem; 
                    border-radius: 15px;
                    margin-bottom: 2rem;'>
            <h1 style='color: #333; text-align: center;'>AI-Powered Requirements Extraction</h1>
            <p style='color: #555; text-align: center;'>Upload a document or enter requirements to extract functional requirements and generate test cases</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load datasets
    requirements_df = None
    test_case_df = None
    
    # Sidebar with file uploaders
    with st.sidebar:
        st.markdown("""
        <div style='background-color: rgba(255, 255, 255, 0.8); 
                    padding: 1rem; 
                    border-radius: 10px;
                    margin-bottom: 1rem;'>
            <h3 style='color: #333;'>Upload Datasets</h3>
        </div>
        """, unsafe_allow_html=True)
        
        requirements_file = st.file_uploader(
            "📁 Requirements Dataset (XLSX)", 
            type=["xlsx"],
            help="Must contain columns: Functional Requirement, Non-Functional Requirement, Constraint"
        )
        test_case_file = st.file_uploader(
            "📁 Test Case Dataset (XLSX)", 
            type=["xlsx"],
            help="Must contain columns: Requirement, Test Case"
        )
        
        if requirements_file and test_case_file:
            with st.spinner("Loading datasets..."):
                try:
                    requirements_df = load_requirements_dataset(requirements_file)
                    test_case_df = load_test_case_dataset(test_case_file)
                    
                    if requirements_df is None or test_case_df is None:
                        st.error("❌ Failed to load datasets. Please ensure:")
                        st.error("- The files are in .xlsx format")
                        st.error("- The columns match expected names")
                        st.stop()
                    
                    # Initialize or load model
                    if 'model' not in st.session_state:
                        st.session_state.model = initialize_model(requirements_df)
                    
                    st.success("✅ Datasets loaded successfully!")
                    
                except Exception as e:
                    st.error(f"⚠ Error loading datasets: {str(e)}")
                    st.error("Please check the file formats and try again.")
                    st.stop()
    
    # Input method selection
    input_method = st.radio(
        "Select Input Method:",
        ("📄 Document Upload", "✍ Manual Input"),
        horizontal=True,
        help="Choose how to provide your requirements"
    )
    
    if requirements_df is None or test_case_df is None:
        st.warning("ℹ Please upload both datasets in the sidebar to proceed")
        st.stop()
    
    # Main content area
    if input_method == "📄 Document Upload":
        with st.container():
            st.markdown("""
            <div style='background-color: rgba(255, 255, 255, 0.8); 
                        padding: 1.5rem; 
                        border-radius: 10px;
                        margin-bottom: 1rem;'>
                <h3 style='color: #333;'>Upload Requirements Document</h3>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose a file (PDF, DOCX, TXT)", 
                type=["pdf", "docx", "txt"],
                label_visibility="collapsed"
            )
            
            if uploaded_file:
                with st.spinner("🔍 Processing document..."):
                    text = process_uploaded_file(uploaded_file)
                    
                    if text:
                        requirements = extract_requirements(text)
                        
                        if requirements:
                            # Classify requirements
                            df = pd.DataFrame({
                                "Original Text": requirements,
                                "Classification": st.session_state.model.predict(requirements),
                                "Confidence": st.session_state.model.predict_proba(requirements).max(axis=1)
                            })
                            
                            st.success(f"✅ Extracted {len(requirements)} requirements")
                            
                            # Display in an expandable section
                            with st.expander("📋 View Extracted Requirements", expanded=True):
                                st.dataframe(df.style.background_gradient(cmap='Blues'))
                            
                            # Generate test cases for functional requirements
                            func_reqs = df[df['Classification'] == 'functional']
                            if not func_reqs.empty:
                                st.markdown("""
                                <div style='background-color: rgba(255, 255, 255, 0.8); 
                                            padding: 1.5rem; 
                                            border-radius: 10px;
                                            margin: 1rem 0;'>
                                    <h3 style='color: #333;'>Generated Test Cases</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                test_case_data = []
                                
                                for _, row in func_reqs.iterrows():
                                    cases = generate_test_cases(row['Original Text'], test_case_df)
                                    for i, case in enumerate(cases, 1):
                                        test_case_data.append({
                                            "Requirement": row['Original Text'],
                                            "Test Case ID": f"TC-{len(test_case_data)+1}",
                                            "Test Case": case,
                                            "Confidence": row['Confidence']
                                        })
                                
                                test_cases_df = pd.DataFrame(test_case_data)
                                st.dataframe(test_cases_df.style.background_gradient(cmap='Greens'))
                                
                                # Export options
                                csv = test_cases_df.to_csv(index=False)
                                st.download_button(
                                    "💾 Download Test Cases",
                                    data=csv,
                                    file_name="test_cases.csv",
                                    mime="text/csv",
                                    help="Download the generated test cases as CSV"
                                )
                        else:
                            st.warning("⚠ No requirements patterns found in the document")
    
    else:  # Manual Input
        with st.container():
            st.markdown("""
            <div style='background-color: rgba(255, 255, 255, 0.8); 
                        padding: 1.5rem; 
                        border-radius: 10px;
                        margin-bottom: 1rem;'>
                <h3 style='color: #333;'>Enter Requirements</h3>
            </div>
            """, unsafe_allow_html=True)
            
            manual_input = st.text_area(
                "Enter requirements (one per line or paragraph):", 
                height=200,
                label_visibility="collapsed"
            )
            
            if manual_input and st.button("🚀 Process Requirements"):
                requirements = [r.strip() for r in re.split(r'\n|\r', manual_input) if r.strip()]
                
                if requirements:
                    # Classify requirements
                    df = pd.DataFrame({
                        "Original Text": requirements,
                        "Classification": st.session_state.model.predict(requirements),
                        "Confidence": st.session_state.model.predict_proba(requirements).max(axis=1)
                    })
                    
                    st.success(f"✅ Processed {len(requirements)} requirements")
                    
                    with st.expander("📋 View Classified Requirements", expanded=True):
                        st.dataframe(df.style.background_gradient(cmap='Blues'))
                    
                    # Generate test cases for functional requirements
                    func_reqs = df[df['Classification'] == 'functional']
                    if not func_reqs.empty:
                        st.markdown("""
                        <div style='background-color: rgba(255, 255, 255, 0.8); 
                                    padding: 1.5rem; 
                                    border-radius: 10px;
                                    margin: 1rem 0;'>
                            <h3 style='color: #333;'>Generated Test Cases</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        test_case_data = []
                        
                        for _, row in func_reqs.iterrows():
                            cases = generate_test_cases(row['Original Text'], test_case_df)
                            for i, case in enumerate(cases, 1):
                                test_case_data.append({
                                    "Requirement": row['Original Text'],
                                    "Test Case ID": f"TC-{len(test_case_data)+1}",
                                    "Test Case": case,
                                    "Confidence": row['Confidence']
                                })
                        
                        test_cases_df = pd.DataFrame(test_case_data)
                        st.dataframe(test_cases_df.style.background_gradient(cmap='Greens'))
                        
                        # Export options
                        csv = test_cases_df.to_csv(index=False)
                        st.download_button(
                            "💾 Download Test Cases",
                            data=csv,
                            file_name="test_cases.csv",
                            mime="text/csv",
                            help="Download the generated test cases as CSV"
                        )

if __name__ == "__main__":

    main()