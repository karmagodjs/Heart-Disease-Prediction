# Heart Disease Prediction Using Machine Learning

A machine learning project that predicts the likelihood of heart disease using patient health-related attributes. This project compares Logistic Regression, Random Forest, and XGBoost classifiers and provides an interactive web application using Streamlit.

## Project Overview

The main objective is to develop a machine learning classification system that predicts whether heart disease may be present based on patient health-related data.

The project includes:
- Data cleaning and preprocessing
- Handling missing values
- Categorical data encoding
- Numerical feature scaling
- Training and comparison of ML models
- Model evaluation using classification metrics
- Interactive prediction using Streamlit

**Note:** This project is developed for educational purposes and is not a clinically validated diagnostic system.

## Dataset Information

**Dataset:** Heart Disease Dataset

**Source:** UCI Machine Learning Repository

**Official Dataset Link:** https://archive.ics.uci.edu/dataset/45/heart+disease

The dataset originates from four databases:
- Cleveland
- Hungary
- Switzerland
- VA Long Beach

This project uses a combined dataset containing 920 records and 16 columns.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Machine Learning Models

Three machine learning algorithms are trained and compared:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

Model selection is based primarily on cross validation recall.

## Model Performance

The following results were obtained on the held-out test dataset:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 82.61% | 83.02% | 86.27% | 84.62% | 92.17% |
| Random Forest | 85.33% | 85.71% | 88.24% | 86.96% | 92.55% |
| XGBoost | 85.33% | 83.78% | 91.18% | 87.32% | 90.79% |

**Selected Model: XGBoost**

XGBoost was selected because it achieved the highest cross-validation recall among the three models.

### Final Model Evaluation

- Accuracy: 85.33%
- Precision: 83.78%
- Recall: 91.18%
- F1 Score: 87.32%
- ROC-AUC: 90.79%

### Confusion Matrix

| Actual / Predicted | No Disease | Disease |
|---|---:|---:|
| No Disease | 64 | 18 |
| Disease | 9 | 93 |

The model correctly identified 93 out of 102 disease-positive records in the test dataset.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Heart-Disease-Prediction.git
```

### 2. Navigate to Project Directory

```bash
cd Heart-Disease-Prediction
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Run the Project

### Step 1: Train the Models

```bash
python -m src.train
```

This command:
- Loads the dataset
- Preprocesses the data
- Trains three machine learning models
- Compares model performance
- Saves the best model and metadata

### Step 2: Evaluate the Model

```bash
python -m src.evaluate
```

This generates:
- Accuracy, precision, recall and F1 score
- Classification report
- Confusion matrix
- ROC curve

Evaluation reports are saved in the `reports/` directory.

### Step 3: Run the Streamlit Application

```bash
streamlit run app.py
```

Open the application in your browser:

http://localhost:8501

Enter the required patient details and click the prediction button to see the model's prediction.

**Important:** The trained model files must exist in the `models/` directory before launching the application. If they are not included in the repository, run the training command first.

## Testing and Validation

The model was successfully trained and evaluated on a held-out test dataset.

**Evaluation Status: PASS**

| Test Case | Expected Result | Status |
|---|---|---|
| Dataset Loading | Dataset loads successfully | PASS |
| Data Preprocessing | Missing values handled | PASS |
| Model Training | All three models train successfully | PASS |
| Model Comparison | Performance metrics generated | PASS |
| Model Evaluation | Classification report generated | PASS |
| Confusion Matrix | Matrix generated and saved | PASS |
| ROC Curve | ROC curve generated and saved | PASS |

The Streamlit interface should be tested separately after launching the application.

## Future Improvements

- Improve model performance and generalization.
- Perform additional hyperparameter optimization.
- Add advanced input validation.
- Improve the user interface.
- Deploy the application online.
- Test using independent external datasets.

## Disclaimer

This project is developed for educational and research purposes only. Its predictions should not be considered medical advice or a substitute for professional medical diagnosis.

## Author

**Your Name**

GitHub: https://github.com/karmagodjs

---

If you find this project useful, consider giving the repository a ⭐ star!
