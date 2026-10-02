Job Acceptance Prediction System

1. Project Overview

The Job Acceptance Prediction System is a machine-learning classification project that predicts whether a candidate is likely to accept a job offer based on candidate characteristics and job-offer attributes.

The target variable is:

Job_Accepted = 1 → Accepted

Job_Accepted = 0 → Rejected

The project includes data cleaning, exploratory data analysis, feature engineering, preprocessing, model training and comparison, model validation, and a Streamlit prediction application.

2. Problem Statement

Recruitment and placement teams work with candidate information such as education, experience, technical skills, interview performance, salary expectations, job characteristics, and company attributes.

The objective of this project is to build a machine-learning classification model that uses these attributes to predict the candidate's job acceptance outcome.

The trained model is integrated into a Streamlit application so that a new candidate's information can be entered and a prediction can be generated.

3. Dataset

The raw dataset contains:

50,000 records

36 columns

The target column is:

Job_Accepted

The cleaned dataset contains:

49,860 records

36 columns

0 missing values

0 exact duplicate rows

0 duplicate-like records remaining

The cleaned target distribution is approximately:

Accepted: 63.85%

Rejected: 36.15%

4. Project Workflow

Raw Dataset
     |
     v
Notebook 01
Data Cleaning & Preprocessing
     |
     v
Cleaned Dataset
     |
     v
Notebook 02
Exploratory Data Analysis
     |
     v
Data Understanding
     |
     v
Notebook 03
Feature Engineering
+ Train/Test Split
+ Encoding
+ Scaling
     |
     v
Model-Ready Data
     |
     v
Notebook 04
Model Training & Comparison
     |
     v
Final SVM Model
     |
     v
Streamlit Application
     |
     v
New Candidate Input
     |
     v
Accepted / Rejected Prediction

5. Notebook 01 — Data Cleaning & Preprocessing

Notebook 01 prepares the raw dataset for analysis and machine learning.

Main activities

Load the raw CSV

Inspect dataset structure and data types

Analyze missing values

Identify exact duplicates

Identify duplicate-like records

Standardize categorical values

Validate numerical ranges

Validate salary fields

Handle missing values

Validate the target variable

Perform final quality checks

Save the cleaned dataset

Output

data/processed/cleaned_job_acceptance_data.csv

6. Notebook 02 — Exploratory Data Analysis

Notebook 02 explores the cleaned dataset without training a machine-learning model.

Main analysis

Dataset structure and quality

Target distribution

Numerical feature distributions

Categorical feature distributions

Accepted vs rejected comparisons

Salary analysis

Experience analysis

Skills analysis

Interview performance analysis

Score-band analysis

Correlation analysis

Important numerical relationships with Job_Accepted

The completed EDA identified the following correlations among the analyzed numerical features:

Feature

Correlation

Interview_Score

0.177

Skills_Match_Percentage

0.161

Technical_Skills_Score

0.115

Soft_Skills_Score

0.112

Communication_Score

0.112

These values describe relationships in this dataset; they do not mean that any individual feature alone determines job acceptance.

EDA outputs

outputs/eda_data_quality_summary.csv
outputs/eda_target_distribution.csv
outputs/eda_accepted_vs_rejected_means.csv
outputs/eda_target_correlations.csv

7. Notebook 03 — Feature Engineering & Train/Test Split

Notebook 03 prepares the data for machine learning.

Derived features

The project creates:

Experience_Category

Academic_Performance_Band

Skills_Match_Level

Interview_Performance_Category

Salary_Gap

Salary_Match_Percentage

Candidate_ID is excluded from the model inputs.

The project does not create a target-derived placement probability feature because using information derived from the target could introduce target leakage.

Train/test split

The data is divided using a stratified 80/20 split:

Training rows: 39,888
Testing rows:   9,972

The acceptance rate is preserved between the training and testing sets.

Preprocessing

Numerical features are processed using:

StandardScaler

Categorical features are processed using:

OneHotEncoder

The preprocessing pipeline is fitted using the training data and then applied to the test data.

Validation

Input model features before encoding: 40
Final encoded features: 102
Train/test overlap: 0
Training matrix contains NaN: False
Test matrix contains NaN: False

Saved artifacts

data/processed/feature_engineered_job_acceptance_data.csv

models/preprocessor.joblib
models/feature_names.joblib
models/train_test_data.joblib

8. Notebook 04 — Model Training & Comparison

Notebook 04 trains and evaluates five classification models:

Logistic Regression

Decision Tree

Random Forest

Gradient Boosting

Support Vector Machine (SVM)

The models are evaluated using:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Confusion Matrix

ROC Curve

Classification Report

Final candidate model

The completed Notebook 04 selected:

SVM

The selection was based on the highest F1 Score among the evaluated models.

Final SVM test-set metrics

Metric

Score

Accuracy

67.26%

Precision

68.38%

Recall

90.62%

F1 Score

77.95%

ROC-AUC

66.34%

Interpretation

The final model produced a high recall relative to its precision. This means it identified a large proportion of the candidates who actually accepted the offer, while also producing some false-positive acceptance predictions.

The ROC-AUC indicates the model has measurable ability to distinguish accepted and rejected candidates, but the model is not a perfect classifier.

These metrics should be presented as the performance of this trained model on this project's held-out test set.

Saved model

models/final_model.joblib

Notebook 04 also reloads the saved model and verifies that its predictions match the original model.

9. Streamlit Application

The project includes a Streamlit application:

app.py

The application loads:

models/final_model.joblib
models/preprocessor.joblib

Application workflow

User enters candidate information
          |
          v
Feature engineering
          |
          v
Saved preprocessing pipeline
          |
          v
102 encoded model features
          |
          v
Saved SVM model
          |
          v
Prediction
          |
          +----> ACCEPTED
          |
          +----> REJECTED

The application also displays the model's acceptance and rejection probability outputs.

The displayed probability is a model output and should not be interpreted as a guarantee of a candidate's real-world decision.

10. Application Testing

The Streamlit application was tested using two substantially different candidate profiles.

Test 1

Prediction: ACCEPTED
Acceptance probability: 71.19%
Rejection probability: 28.81%

Test 2

Prediction: REJECTED
Acceptance probability: 48.11%
Rejection probability: 51.89%

Both probability pairs sum to 100%, and the application successfully generated different predictions for the two input profiles.

11. Project Structure

job-acceptance-prediction-system/
│
├── data/
│   ├── raw/
│   │   └── job_acceptance_raw.csv
│   │
│   └── processed/
│       ├── cleaned_job_acceptance_data.csv
│       └── feature_engineered_job_acceptance_data.csv
│
├── notebooks/
│   ├── 01_data_cleaning_preprocessing.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering_train_test_split.ipynb
│   └── 04_model_training_and_comparison.ipynb
│
├── models/
│   ├── preprocessor.joblib
│   ├── feature_names.joblib
│   ├── train_test_data.joblib
│   └── final_model.joblib
│
├── outputs/
│   ├── eda_data_quality_summary.csv
│   ├── eda_target_distribution.csv
│   ├── eda_accepted_vs_rejected_means.csv
│   ├── eda_target_correlations.csv
│   └── model_comparison_results.csv
│
├── app.py
├── requirements.txt
└── README.md

12. Technologies Used

Python 3.11.9

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Joblib

Streamlit

Jupyter Notebook

13. How to Run the Project

Step 1 — Activate the virtual environment

From Git Bash:

cd "/d/job acceptance prediction system"
source .venv/Scripts/activate

Step 2 — Move to the project directory

In the current project setup, the project files are already located at:

/d/job acceptance prediction system

Verify:

ls

You should see:

app.py
data/
models/
notebooks/
outputs/
Readme.md
requirements.txt

Step 3 — Start Streamlit

streamlit run app.py

The application should open at:

http://localhost:8501

14. Key Machine-Learning Concepts Demonstrated

This project demonstrates:

Data cleaning

Missing-value handling

Duplicate detection

Categorical standardization

Exploratory Data Analysis

Feature engineering

Target leakage prevention

Train/test splitting

Stratification

One-Hot Encoding

Standardization

Classification

Logistic Regression

Decision Trees

Random Forest

Gradient Boosting

Support Vector Machines

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Confusion Matrix

ROC Curve

Model persistence with Joblib

Streamlit deployment/application development

15. Limitations

The model's predictions are based on patterns learned from the provided dataset.

The final model achieved:

Accuracy = 67.26%
ROC-AUC  = 66.34%

Therefore, predictions should be treated as model-generated estimates rather than guaranteed outcomes.

The dataset may not represent every real-world candidate, employer, industry, or labor-market condition.
