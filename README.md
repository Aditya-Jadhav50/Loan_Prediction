# Loan Prediction using Machine Learning

Aim: The objective is to build a reliable machine learning model in Python to predict loan eligibility based on applicant characteristics using supervised classification techniques.

---

##  Repository Contents

| File/Folder        | Description |
|--------------------|-------------|
| `Loan Model.ipynb` | Jupyter Notebook with full machine learning pipeline for loan prediction |
| `dataset`          | Contains Data |
| `README.md`        | Project summary and usage documentation |

---

## Project Objective

To assist financial institutions in evaluating loan applications more reliably and efficiently using a machine learning model trained on historical applicant data. The model helps identify high-risk applicants based on features like income, credit history, employment status, and education.

---

## Dataset

- **Source**: [Kaggle Loan Prediction Dataset]
- **Records**: 614
- **Target Column**: `Loan_Status`
- **Features**: Gender, Marital Status, Education, Income, Loan Amount, Credit History, etc.

---

##  Technologies Used

- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Methodology Overview

1. **Data Preprocessing**
   - Missing value imputation
   - Label encoding of categorical variables
   - Feature scaling with `StandardScaler` or `MinMaxScaler`

2. **Exploratory Data Analysis (EDA)**
   - Visual insights using bar plots, histograms, and heatmaps
   - Key predictors: credit history, income, loan amount

3. **Model Training**
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Naive Bayes
   - K-Nearest Neighbors (KNN)

4. **Model Evaluation**
   - Accuracy score
   - Train/Test split
   - Hyperparameter tuning with `GridSearchCV` and `RandomizedSearchCV`

---

##  Results

| Model                  | Training Accuracy | Testing Accuracy |
|-----------------------|-------------------|------------------|
| Random Forest          | 100%              | 96.43%           |
| Logistic Regression    | 80.56%            | 96.43%           |
| Decision Tree          | 100%              | 96.43%           |
| K-Nearest Neighbors    | 100%              | 92.86%           |
| Bernoulli Naive Bayes  | 80.56%            | 92.86%           |
| Gaussian Naive Bayes   | 79.63%            | 89.29%           |

 **Best Model**: Random Forest  
**Most Important Features**: Credit History, Applicant Income, Loan Amount

---
