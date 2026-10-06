🎯 Job Acceptance Prediction System
A machine learning application that predicts whether a candidate is likely to accept or reject a job offer based on academic background, experience, skills, interview performance, salary expectations, company/job information, and other candidate attributes.
The project includes a complete machine learning workflow covering data cleaning, exploratory data analysis, feature engineering, model comparison, SMOTE experimentation, hyperparameter tuning, threshold optimization, final model selection, and a Streamlit prediction application.
📌 Project Overview
Recruitment and placement teams work with large volumes of candidate information when evaluating job offers. This project uses historical candidate data to build a binary classification model that predicts:
- 1 → Accepted
- 0 → Rejected
The final model is deployed through a Streamlit web application, where users can enter candidate details and receive an acceptance prediction with the model's probability output.
🎯 Objectives
- Clean and preprocess real-world candidate data.
- Analyze candidate characteristics and their relationship with job acceptance.
- Create meaningful analytical features.
- Compare multiple machine learning classification algorithms.
- Evaluate class-wise Precision, Recall, and F1-score.
- Experiment with SMOTE to address class imbalance.
- Perform fast hyperparameter tuning.
- Optimize the classification threshold.
- Select the model based on balanced class performance using Macro F1.
- Deploy the final model through a Streamlit application.
📊 Dataset
The original dataset contains:
- 50,000 candidate records
- 36 original columns
- Binary target: Job_Accepted
Original Target Distribution
Target	Candidates	Percentage
Accepted	31,935	63.87%
Rejected	18,065	36.13%


After duplicate removal and data cleaning:
- 49,860 records
- 36 original columns
- 0 missing values
- 0 exact duplicate records
- 0 duplicate-like records remaining
- 0 invalid numeric values remaining
The cleaned target distribution is approximately:
- Accepted: 63.85%
- Rejected: 36.15%
🔄 Machine Learning Workflow
Raw Candidate Dataset
        ↓
Data Cleaning & Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Train / Test Split
        ↓
Data Preprocessing & Encoding
        ↓
Baseline Model Comparison
        ↓
SMOTE Experiment
        ↓
Hyperparameter Tuning
        ↓
Class-wise Evaluation
        ↓
Threshold Optimization
        ↓
Macro F1 Comparison
        ↓
Final Model Selection
        ↓
Saved Model + Preprocessor
        ↓
Streamlit Application
📓 Project Notebooks
01 — Data Cleaning & Preprocessing
File:
notebooks/01_data_cleaning_preprocessing.ipynb
Main activities:
- Load the raw dataset.
- Inspect shape, columns, data types, and statistics.
- Analyze missing values.
- Identify exact duplicates.
- Identify duplicate-like records excluding Candidate_ID.
- Standardize categorical values.
- Validate numeric ranges.
- Handle invalid salary values.
- Impute missing numerical values using the median.
- Impute missing categorical values using the mode.
- Validate the target variable.
- Save the cleaned dataset.
Output:
data/processed/cleaned_job_acceptance_data.csv
Final cleaned dataset:
49,860 rows × 36 columns
02 — Exploratory Data Analysis
File:
notebooks/02_exploratory_data_analysis.ipynb
The EDA notebook analyzes:
- Overall job acceptance distribution.
- Candidate characteristics.
- Accepted vs. rejected candidate averages.
- Numerical relationships with job acceptance.
- Correlations between numerical variables.
- Data quality.
Some of the stronger numerical relationships with the target include:
Feature	Correlation
Interview Score	0.177
Skills Match Percentage	0.161
Technical Skills Score	0.115
Soft Skills Score	0.112
Communication Score	0.112


EDA outputs are stored under:
outputs/
03 — Feature Engineering & Train/Test Split
File:
notebooks/03_feature_engineering_train_test_split.ipynb
The notebook creates the following derived features:
- Experience_Category
- Academic_Performance_Band
- Skills_Match_Level
- Interview_Performance_Category
- Salary_Gap
- Salary_Match_Percentage
The target variable is kept separate to avoid target leakage.
The dataset is split using:
- 80% Training
- 20% Testing
- random_state = 42
- Stratified split
Result:
Dataset	Rows
Training	39,888
Testing	9,972


After preprocessing and one-hot encoding:
102 model features
The fitted preprocessing pipeline is saved as:
models/preprocessor.joblib
🤖 04 — Model Optimization & Balanced Model Selection
File:
notebooks/04_model_optimization_balanced_complete.ipynb
This notebook performs the final model evaluation and selection.
Baseline Models
The following models were evaluated:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. Support Vector Machine (SVM)
Additional Experiments
The notebook also includes:
- SMOTE experiment
- Fast hyperparameter tuning
- Class-wise Precision
- Class-wise Recall
- Class-wise F1-score
- Macro F1
- Threshold testing
- Balanced model selection
- Final model saving
- Saved-model reload verification
🏆 Final Model
The final model was selected based on Macro F1, because the project goal was not simply to maximize performance for the more common Accepted class.
The selected model is:
Gradient Boosting with a classification threshold of 0.60

Final Performance
Metric	Result
Accuracy	66.19%
Rejected Precision	53.23%
Rejected Recall	53.31%
Rejected F1	53.27%
Accepted Precision	73.54%
Accepted Recall	73.47%
Accepted F1	73.51%
Macro F1	63.39%


Why Gradient Boosting?
The original SVM baseline achieved a higher overall F1 for the Accepted class, but its performance on the Rejected class was considerably weaker.
The original SVM results included:
Metric	SVM Baseline
Rejected Recall	25.96%
Rejected F1	36.44%
Accepted F1	77.95%
Macro F1	57.20%
Accuracy	67.26%


After threshold optimization, Gradient Boosting provided a substantially more balanced result:
Metric	Final Gradient Boosting
Rejected Recall	53.31%
Rejected F1	53.27%
Accepted F1	73.51%
Macro F1	63.39%
Accuracy	66.19%


Therefore, Gradient Boosting with a 0.60 threshold was selected as the final model because it achieved the highest Macro F1 among the evaluated configurations.
💾 Saved Model Artifacts
The application uses the following saved files:
models/
├── final_model.joblib
├── final_model_metadata.joblib
├── preprocessor.joblib
└── train_test_data.joblib
final_model.joblib
Contains the selected Gradient Boosting model.
final_model_metadata.joblib
Contains the final model metadata, including the classification threshold used by the Streamlit application.
preprocessor.joblib
Contains the fitted preprocessing pipeline used to transform application inputs into the model's encoded feature representation.
train_test_data.joblib
Contains the prepared training and testing data used during model development and evaluation.
🖥️ Streamlit Application
The project includes an interactive Streamlit application:
app.py
The application allows users to enter candidate information such as:
- Age
- Gender
- Education Level
- Degree Field
- University Tier
- CGPA
- Years of Experience
- Previous Companies
- Internship Experience
- Certifications
- Technical Skills
- Soft Skills
- Skills Match
- Aptitude
- Communication
- Interview Score
- Interview Rounds
- Job Role
- Industry
- Company Tier
- Company Size
- Job Type
- Work Mode
- Job Location
- Preferred Location
- Relocation Willingness
- Competition Level
- Expected Salary
- Offered Salary
- Career Growth
- Benefits
- Job Security
- Notice Period
- Offer-to-Joining Days
The application performs the same feature engineering and preprocessing used during model training.
The final prediction uses the saved 0.60 classification threshold.
🚀 Running the Application
1. Clone the Repository
git clone https://github.com/MadavSelvam/job-placement-prediction-system.git
cd job-placement-prediction-system
2. Create a Virtual Environment
python -m venv .venv
Windows
.venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Run Streamlit
streamlit run app.py
The application will open in your browser.