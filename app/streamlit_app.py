import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hypothyroidism CDSS",
    page_icon="🏥",
    layout="wide"
)

# ─── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.join(os.path.dirname(__file__), '..', 'models')
    models = {
        'Gradient Boosting (Best)':  joblib.load(f'{base}/gradient_boosting.pkl'),
        'Random Forest':             joblib.load(f'{base}/random_forest.pkl'),
        'Logistic Regression':       joblib.load(f'{base}/logistic_regression.pkl'),
    }
    scaler = joblib.load(f'{base}/scaler.pkl')
    return models, scaler

models, scaler = load_artifacts()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/hospital.png", width=60)
st.sidebar.title("CDSS Settings")

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance")
perf = {
    'Gradient Boosting (Best)':  {'Accuracy': '96.05%', 'Recall': '66.67%', 'AUC': '0.979'},
    'Random Forest':             {'Accuracy': '94.94%', 'Recall': '80.00%', 'AUC': '0.978'},
    'Logistic Regression':       {'Accuracy': '94.94%', 'Recall': '80.00%', 'AUC': '0.942'},
}
for k, v in perf[selected_model].items():
    st.sidebar.metric(k, v)

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚕️ This tool assists healthcare providers. "
    "It does NOT replace clinical diagnosis."
)

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🏥 Hypothyroidism Clinical Decision Support System")
st.markdown(
    "Enter patient data below and click **Run Prediction** "
    "to assess the risk of hypothyroidism."
)
st.markdown("---")

# ─── Input Form ───────────────────────────────────────────────────────────────
st.subheader("📋 Patient Data Entry")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    age = st.number_input("Age (years)", 1, 120, 35)
    sex = st.selectbox("Sex", ["Female", "Male"])

    st.markdown("**Medications**")
    on_thyroxine       = st.selectbox("On Thyroxine?",              ["No", "Yes"])
    query_on_thyroxine = st.selectbox("Query on Thyroxine?",        ["No", "Yes"])
    on_antithyroid     = st.selectbox("On Antithyroid Medication?", ["No", "Yes"])

with col2:
    st.markdown("**Medical History**")
    sick            = st.selectbox("Currently Sick?",           ["No", "Yes"])
    pregnant        = st.selectbox("Pregnant?",                 ["No", "Yes"])
    thyroid_surgery = st.selectbox("Thyroid Surgery History?",  ["No", "Yes"])
    I131_treatment  = st.selectbox("I131 Treatment?",           ["No", "Yes"])
    lithium         = st.selectbox("On Lithium?",               ["No", "Yes"])
    goitre          = st.selectbox("Goitre Present?",           ["No", "Yes"])
    tumor           = st.selectbox("Tumor Present?",            ["No", "Yes"])
    hypopituitary   = st.selectbox("Hypopituitary?",            ["No", "Yes"])
    psych           = st.selectbox("Psychological Symptoms?",   ["No", "Yes"])

with col3:
    st.markdown("**Query Flags**")
    query_hypothyroid  = st.selectbox("Query Hypothyroid?",  ["No", "Yes"])
    query_hyperthyroid = st.selectbox("Query Hyperthyroid?", ["No", "Yes"])

    st.markdown("**Lab Values**")
    TSH = st.number_input("TSH (mIU/L)",   0.0, 200.0, 2.5,  step=0.1)
    T3  = st.number_input("T3  (nmol/L)",  0.0,  20.0, 1.8,  step=0.1)
    TT4 = st.number_input("TT4 (nmol/L)",  0.0, 400.0, 100.0, step=1.0)
    T4U = st.number_input("T4U (ratio)",   0.0,   5.0, 1.0,  step=0.1)

st.markdown("---")

# ─── Prediction ───────────────────────────────────────────────────────────────
def yn(val):
    return 1 if val == "Yes" else 0

def build_input():
    features = np.array([[
        age,
        1 if sex == "Male" else 0,
        yn(on_thyroxine),
        yn(query_on_thyroxine),
        yn(on_antithyroid),
        yn(sick),
        yn(pregnant),
        yn(thyroid_surgery),
        yn(I131_treatment),
        yn(query_hypothyroid),
        yn(query_hyperthyroid),
        yn(lithium),
        yn(goitre),
        yn(tumor),
        yn(hypopituitary),
        yn(psych),
        TSH, T3, TT4, T4U
    ]])
    return scaler.transform(features)

if st.button("🔍 Run Prediction", use_container_width=True, type="primary"):

    input_data  = build_input()
    model       = models[selected_model]
    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        if prediction == 1:
            if probability >= 0.7:
                st.error("### ⚠️ HIGH RISK — Hypothyroidism Likely")
                recommendation = (
                    "**Immediate action recommended:** Order full thyroid "
                    "function panel (TSH, Free T3, Free T4). "
                    "Refer to endocrinologist."
                )
            else:
                st.warning("### 🔶 MODERATE RISK — Further Testing Advised")
                recommendation = (
                    "**Suggested action:** Repeat TSH test in 4–6 weeks. "
                    "Monitor symptoms. Consider specialist referral."
                )
        else:
            st.success("### ✅ LOW RISK — Normal Thyroid Function Suggested")
            recommendation = (
                "**Suggested action:** Routine follow-up. "
                "Re-evaluate if symptoms develop or worsen."
            )

        st.markdown(recommendation)

    with res_col2:
        st.metric(
            label="Hypothyroidism Probability",
            value=f"{probability * 100:.1f}%"
        )
        st.progress(float(probability))
        st.caption(f"Model used: {selected_model}")

    # Detailed probability breakdown
    with st.expander("🔬 Detailed Probability Breakdown"):
        prob_df = pd.DataFrame({
            'Class':       ['Normal', 'Hypothyroid'],
            'Probability': [
                f"{model.predict_proba(input_data)[0][0]*100:.1f}%",
                f"{model.predict_proba(input_data)[0][1]*100:.1f}%"
            ]
        })
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

    st.info(
        "⚕️ **Disclaimer:** This system is a decision support tool only. "
        "Final diagnosis must be confirmed by a qualified healthcare professional "
        "using laboratory tests and clinical examination."
    )

# ─── About section ────────────────────────────────────────────────────────────
with st.expander("ℹ️ About This System"):
    st.markdown("""
    **Machine Learning-Based CDSS for Hypothyroidism Detection**

    This system was developed to assist Ethiopian healthcare providers 
    in the early identification of hypothyroidism, especially in areas 
    with limited access to endocrinology specialists.

    **Dataset:** UCI Hypothyroid Dataset (3,163 patients)  
    **Models:** Logistic Regression, Random Forest, Gradient Boosting  
    **Best Model:** Gradient Boosting (AUC-ROC: 0.979)  

    **Key Features Used:**
    TSH, T3, TT4, T4U levels, age, sex, medications, and clinical symptoms.
    """)
    ##----------------------Gracias-----------------------------------