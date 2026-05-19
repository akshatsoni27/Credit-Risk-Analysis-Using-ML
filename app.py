import streamlit as st
import pandas as pd
import joblib
import os

# Set page config for a premium look
st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #e6edf3;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #58a6ff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Metrics / Callouts */
    div[data-testid="stMetricValue"] {
        color: #58a6ff;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #238636;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.4);
    }
    
    /* Widget backgrounds */
    .stSelectbox, .stSlider {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = 'model.joblib'
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_model()

st.title("💳 Premium Credit Risk Analysis System")
st.markdown("Evaluate credit applications in real-time using an advanced XGBoost model. This system is heavily optimized to detect high-risk applicants, incorporating a 5:1 asymmetrical cost matrix to prevent costly defaults.")

if not model:
    st.error("⚠️ Model artifact `model.joblib` not found. Please run `train.py` first to bake the model.")
    st.stop()

# Layout: Split into two columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("👤 Applicant Profile")
    
    age = st.slider("Age (years)", min_value=18, max_value=100, value=30, step=1)
    sex = st.selectbox("Sex", options=["male", "female"])
    
    job_desc = st.selectbox(
        "Employment Status (Job)", 
        options=[
            "0 - Unskilled and Non-Resident",
            "1 - Unskilled and Resident",
            "2 - Skilled",
            "3 - Highly Skilled"
        ],
        index=2
    )
    job = job_desc.split(" ")[0] # extract just the number
    
    housing = st.selectbox("Housing Situation", options=["own", "free", "rent"])

with col2:
    st.header("🏦 Financial Details")
    
    credit_amount = st.slider("Requested Credit Amount", min_value=250, max_value=20000, value=2500, step=50, format="$%d")
    duration = st.slider("Duration of Credit (months)", min_value=4, max_value=72, value=12, step=1)
    
    purpose = st.selectbox("Purpose of Credit", options=[
        "radio/TV", "education", "furniture/equipment", "car", 
        "business", "domestic appliances", "repairs", "vacation/others"
    ])
    
    checking_account = st.selectbox("Checking Account Status", options=["Unknown", "little", "moderate", "rich"])
    saving_accounts = st.selectbox("Savings Account Status", options=["Unknown", "little", "moderate", "quite rich", "rich"])

st.markdown("---")

# Prediction Button
if st.button("Predict Risk Verdict", use_container_width=True):
    # Construct input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Job': [job],  # Will be converted to str in pipeline
        'Housing': [housing],
        'Saving accounts': [saving_accounts],
        'Checking account': [checking_account],
        'Credit amount': [credit_amount],
        'Duration': [duration],
        'Purpose': [purpose]
    })
    
    try:
        # Get probability of class 1 (Bad Risk)
        prob_bad = float(model.predict_proba(input_data)[0][1])
        prob_good = 1.0 - prob_bad
        
        # Prediction class
        prediction = model.predict(input_data)[0]
        
        st.header("📊 Verdict")
        
        if prediction == 1:
            st.error("### 🛑 Bad Credit - High Risk")
            st.markdown(f"**Confidence:** {prob_bad:.1%} likelihood of default.")
            st.info("💡 **System Note**: This applicant was flagged due to our strict 5:1 recall optimization strategy.")
            st.progress(prob_bad)
        else:
            st.success("### ✅ Good Credit - Low Risk")
            st.markdown(f"**Confidence:** {prob_good:.1%} likelihood of successful repayment.")
            st.progress(prob_good)
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")
