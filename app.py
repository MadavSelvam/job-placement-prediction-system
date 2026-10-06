
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Job Acceptance Prediction System - Streamlit Application
# ============================================================

st.set_page_config(
    page_title="Job Acceptance Prediction System",
    page_icon="💼",
    layout="wide",
)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "final_model_metadata.joblib"


# ------------------------------------------------------------
# Page styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #666;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Load model and preprocessor
# ------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model not found:\n{MODEL_PATH}\n\n"
            "Make sure models/final_model.joblib exists."
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found:\n{PREPROCESSOR_PATH}\n\n"
            "Make sure models/preprocessor.joblib exists."
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    threshold = 0.50
    metadata = {}
    if METADATA_PATH.exists():
        metadata = joblib.load(METADATA_PATH)
        threshold = float(metadata.get("threshold", 0.50) or 0.50)
    return model, preprocessor, threshold, metadata


try:
    model, preprocessor, threshold, model_metadata = load_artifacts()
except Exception as exc:
    st.error("The application could not load the saved ML artifacts.")
    st.code(str(exc))
    st.stop()


# ------------------------------------------------------------
# Feature engineering
# Must match Notebook 03 exactly.
# ------------------------------------------------------------
def create_model_features(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()

    # Experience category
    df["Experience_Category"] = pd.cut(
        df["Years_of_Experience"],
        bins=[-np.inf, 1, 5, 10, np.inf],
        labels=["Fresher/Entry", "Junior", "Mid-Level", "Senior"],
    )

    # Academic performance category
    df["Academic_Performance_Band"] = pd.cut(
        df["CGPA"],
        bins=[-np.inf, 6.49, 7.99, np.inf],
        labels=["Low", "Medium", "High"],
    )

    # Skills match category
    df["Skills_Match_Level"] = pd.cut(
        df["Skills_Match_Percentage"],
        bins=[-np.inf, 49.99, 74.99, np.inf],
        labels=["Low", "Medium", "High"],
    )

    # Interview performance category
    df["Interview_Performance_Category"] = pd.cut(
        df["Interview_Score"],
        bins=[-np.inf, 49.99, 74.99, np.inf],
        labels=["Low", "Medium", "High"],
    )

    # Salary-derived features
    df["Salary_Gap"] = (
        df["Offered_Salary"] - df["Expected_Salary"]
    )

    df["Salary_Match_Percentage"] = np.where(
        df["Expected_Salary"] > 0,
        df["Offered_Salary"] / df["Expected_Salary"] * 100,
        np.nan,
    )

    return df


# ------------------------------------------------------------
# Input options based on the cleaned dataset used by the project
# ------------------------------------------------------------
OPTIONS = {
    "Gender": ["Male", "Female", "Other"],
    "Education_Level": ["Diploma", "Bachelor's", "Master's", "PhD"],
    "Degree_Field": [
        "Computer Science",
        "Engineering",
        "Business Administration",
        "Data Science",
        "Finance",
        "Other",
        "Arts",
        "Information Technology",
        "Commerce",
    ],
    "University_Tier": ["Tier 1", "Tier 2", "Tier 3"],
    "Internship_Experience": ["Yes", "No"],
    "Job_Role": [
        "Data Scientist",
        "Operations Analyst",
        "Software Engineer",
        "Marketing Analyst",
        "Data Analyst",
        "Financial Analyst",
        "Business Analyst",
        "HR Analyst",
    ],
    "Industry": [
        "Telecom",
        "Healthcare",
        "Consulting",
        "IT",
        "E-commerce",
        "Finance",
        "Manufacturing",
        "Retail",
    ],
    "Company_Tier": ["Tier 1", "Tier 2", "Tier 3"],
    "Company_Size": ["Small", "Medium", "Large"],
    "Job_Type": ["Full-Time", "Part-Time", "Contract"],
    "Work_Mode": ["Onsite", "Hybrid", "Remote"],
    "Job_Location": [
        "Chennai",
        "Bangalore",
        "Delhi",
        "Hyderabad",
        "Kolkata",
        "Mumbai",
        "Pune",
    ],
    "Preferred_Location": [
        "Chennai",
        "Bangalore",
        "Delhi",
        "Hyderabad",
        "Kolkata",
        "Mumbai",
        "Pune",
    ],
    "Relocation_Willingness": ["Yes", "No"],
    "Competition_Level": ["Low", "Medium", "High"],
}


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.markdown(
    '<div class="main-title">💼 Job Acceptance Prediction System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    "Predict whether a candidate is likely to accept a job offer "
    "using the trained machine-learning model."
    "</div>",
    unsafe_allow_html=True,
)

st.info(
    "The application uses the same feature engineering and saved "
    "preprocessing pipeline used during model training."
)

with st.form("prediction_form"):

    st.subheader("👤 Candidate Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=70,
            value=25,
            step=1,
        )

        gender = st.selectbox(
            "Gender",
            OPTIONS["Gender"],
        )

        education = st.selectbox(
            "Education Level",
            OPTIONS["Education_Level"],
        )

        degree_field = st.selectbox(
            "Degree Field",
            OPTIONS["Degree_Field"],
        )

        university_tier = st.selectbox(
            "University Tier",
            OPTIONS["University_Tier"],
        )

        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=8.0,
            step=0.1,
        )

        years_experience = st.number_input(
            "Years of Experience",
            min_value=0,
            max_value=50,
            value=2,
            step=1,
        )

        previous_companies = st.number_input(
            "Previous Companies",
            min_value=0,
            max_value=30,
            value=1,
            step=1,
        )

        internship = st.selectbox(
            "Internship Experience",
            OPTIONS["Internship_Experience"],
        )

        certifications = st.number_input(
            "Certifications Count",
            min_value=0,
            max_value=30,
            value=2,
            step=1,
        )

    with col2:
        technical_skills = st.number_input(
            "Technical Skills Score",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0,
        )

        soft_skills = st.number_input(
            "Soft Skills Score",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0,
        )

        skills_match = st.number_input(
            "Skills Match Percentage",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0,
        )

        aptitude = st.number_input(
            "Aptitude Score",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0,
        )

        communication = st.number_input(
            "Communication Score",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0,
        )

        interview_score = st.number_input(
            "Interview Score",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0,
        )

        interview_rounds = st.number_input(
            "Interview Rounds",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
        )

        job_role = st.selectbox(
            "Job Role",
            OPTIONS["Job_Role"],
        )

        industry = st.selectbox(
            "Industry",
            OPTIONS["Industry"],
        )

        company_tier = st.selectbox(
            "Company Tier",
            OPTIONS["Company_Tier"],
        )

    with col3:
        company_size = st.selectbox(
            "Company Size",
            OPTIONS["Company_Size"],
        )

        job_type = st.selectbox(
            "Job Type",
            OPTIONS["Job_Type"],
        )

        work_mode = st.selectbox(
            "Work Mode",
            OPTIONS["Work_Mode"],
        )

        job_location = st.selectbox(
            "Job Location",
            OPTIONS["Job_Location"],
        )

        preferred_location = st.selectbox(
            "Preferred Location",
            OPTIONS["Preferred_Location"],
        )

        relocation = st.selectbox(
            "Relocation Willingness",
            OPTIONS["Relocation_Willingness"],
        )

        competition = st.selectbox(
            "Competition Level",
            OPTIONS["Competition_Level"],
        )

        expected_salary = st.number_input(
            "Expected Salary",
            min_value=0.0,
            value=60000.0,
            step=1000.0,
            help="Use the same salary unit used by the training dataset.",
        )

        offered_salary = st.number_input(
            "Offered Salary",
            min_value=0.0,
            value=65000.0,
            step=1000.0,
            help="Use the same salary unit used by the training dataset.",
        )

        career_growth = st.number_input(
            "Career Growth Score",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0,
        )

        benefits = st.number_input(
            "Benefits Score",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0,
        )

        job_security = st.number_input(
            "Job Security Score",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0,
        )

        notice_period = st.number_input(
            "Notice Period (Days)",
            min_value=0,
            max_value=365,
            value=30,
            step=1,
        )

        offer_to_joining = st.number_input(
            "Offer to Joining (Days)",
            min_value=0,
            max_value=365,
            value=30,
            step=1,
        )

    submitted = st.form_submit_button(
        "🔮 Predict Job Acceptance",
        use_container_width=True,
    )


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
if submitted:

    raw_input = pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": gender,
                "Education_Level": education,
                "Degree_Field": degree_field,
                "University_Tier": university_tier,
                "CGPA": cgpa,
                "Years_of_Experience": years_experience,
                "Previous_Companies": previous_companies,
                "Internship_Experience": internship,
                "Certifications_Count": certifications,
                "Technical_Skills_Score": technical_skills,
                "Soft_Skills_Score": soft_skills,
                "Skills_Match_Percentage": skills_match,
                "Aptitude_Score": aptitude,
                "Communication_Score": communication,
                "Interview_Score": interview_score,
                "Interview_Rounds": interview_rounds,
                "Job_Role": job_role,
                "Industry": industry,
                "Company_Tier": company_tier,
                "Company_Size": company_size,
                "Job_Type": job_type,
                "Work_Mode": work_mode,
                "Job_Location": job_location,
                "Preferred_Location": preferred_location,
                "Relocation_Willingness": relocation,
                "Competition_Level": competition,
                "Expected_Salary": expected_salary,
                "Offered_Salary": offered_salary,
                "Career_Growth_Score": career_growth,
                "Benefits_Score": benefits,
                "Job_Security_Score": job_security,
                "Notice_Period_Days": notice_period,
                "Offer_to_Joining_Days": offer_to_joining,
            }
        ]
    )

    try:
        # Create the exact six engineered features used in Notebook 03.
        model_input = create_model_features(raw_input)

        # The saved preprocessor was fitted on X after Candidate_ID was
        # removed, so Candidate_ID is intentionally not included here.
        transformed_input = preprocessor.transform(model_input)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(transformed_input)[0]
            rejected_probability = float(probabilities[0])
            accepted_probability = float(probabilities[1])
            prediction = int(accepted_probability >= threshold)
        else:
            # Fallback for models without predict_proba.
            decision = float(model.decision_function(transformed_input)[0])
            accepted_probability = 1 / (1 + np.exp(-decision))
            rejected_probability = 1 - accepted_probability
            prediction = int(accepted_probability >= threshold)

        st.divider()
        st.subheader("📊 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == 1:
                st.success("### ✅ ACCEPTED")
                st.write(
                    "The model predicts that the candidate is likely to "
                    "accept the job offer."
                )
            else:
                st.warning("### ❌ REJECTED")
                st.write(
                    "The model predicts that the candidate is likely to "
                    "reject the job offer."
                )

        with result_col2:
            st.metric(
                "Acceptance Probability",
                f"{accepted_probability * 100:.2f}%",
            )
            st.metric(
                "Rejection Probability",
                f"{rejected_probability * 100:.2f}%",
            )

        st.progress(
            min(max(accepted_probability, 0.0), 1.0),
            text=f"Acceptance probability: {accepted_probability * 100:.2f}%",
        )

        st.caption(
            "Probability shown by the saved final model. It should be "
            "interpreted as a model output, not a guarantee of the candidate's decision."
        )

    except Exception as exc:
        st.error("Prediction failed.")
        st.exception(exc)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.header("About the Model")
    st.write("**Final model:** Gradient Boosting")
    st.write(f"**Classification threshold:** {threshold:.2f}")
    st.write("**Training rows:** 39,888")
    st.write("**Testing rows:** 9,972")
    st.write("**Model features:** 102 encoded features")

    st.divider()

    st.caption(
        "This application uses the saved model and preprocessor "
        "generated by the project's training notebooks."
    )
