# 💳 Credit Card Customer Churn Prediction

A machine learning classification project to predict which credit card customers are at risk of churning, enabling Thera Bank to proactively retain valuable customers.

---

## 📌 Problem Statement

Customer churn is one of the most costly problems in banking. Acquiring a new customer costs 5–25× more than retaining an existing one. This project builds a predictive model to identify customers likely to close their credit card accounts — giving the bank time to intervene with targeted retention strategies.

---

## 📊 Dataset

- **Source:** Thera Bank Credit Card Customer Dataset
- **Size:** ~10,000 customers, 21 features
- **Target variable:** `Attrition_Flag` (Existing Customer / Attrited Customer)
- **Class imbalance:** ~84% existing, ~16% churned

**Key features include:**
- Total transaction count & amount (last 12 months)
- Revolving balance & credit limit utilisation
- Months on book & relationship count
- Customer demographics (age, gender, income, education)

---

## 🔍 Approach

### 1. Exploratory Data Analysis (EDA)
- Distribution of churned vs retained customers
- Correlation analysis between features
- Key churn indicators identified visually

### 2. Data Preprocessing
- Dropped irrelevant columns (`CLIENTNUM`, naive Bayes columns)
- Encoded categorical variables (Label Encoding & One-Hot Encoding)
- Handled class imbalance using:
  - **SMOTE** (Synthetic Minority Over-sampling Technique)
  - **Random Undersampling**

### 3. Models Trained & Compared

| Model | Recall (Churn) | AUC |
|---|---|---|
| Logistic Regression | ~72% | ~0.83 |
| Decision Tree | ~78% | ~0.87 |
| Random Forest | ~83% | ~0.93 |
| AdaBoost | ~80% | ~0.92 |
| Gradient Boosting | ~84% | ~0.94 |
| **XGBoost** | **~88%** | **~0.96** |

### 4. Hyperparameter Tuning
- Grid Search CV used on top models
- Optimised for **Recall** (minimise false negatives — missing a churner is costly)

---

## 🏆 Results

**Best Model: XGBoost (with SMOTE)**

- **Recall (Churn class):** ~88%
- **Precision:** ~85%
- **F1-Score:** ~86%
- **ROC-AUC:** ~0.96

---

## 💡 Key Business Insights

1. **Total Transaction Count** is the strongest predictor — customers with fewer transactions are far more likely to churn
2. **Revolving Balance** matters — low balance customers show higher churn risk
3. **Relationship Count** — customers with fewer products have less loyalty
4. **Months Inactive** — inactivity over 2–3 months is a strong warning signal

**Recommendations:**
- Trigger retention campaigns when transaction count drops below threshold
- Offer loyalty rewards to customers with only 1 product
- Create re-engagement emails for customers inactive >2 months

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Pandas, NumPy** — data manipulation
- **Matplotlib, Seaborn** — visualisation
- **Scikit-learn** — ML models, preprocessing, evaluation
- **XGBoost** — gradient boosting
- **Imbalanced-learn** — SMOTE, undersampling

---

## 📁 Project Structure

```
credit-card-churn-prediction/
│
├── data/
│   └── README.md                  # Dataset description & source
│
├── notebooks/
│   └── churn_prediction.ipynb     # Full analysis notebook
│
├── src/
│   ├── preprocess.py              # Data cleaning & encoding
│   ├── train.py                   # Model training pipeline
│   └── evaluate.py                # Evaluation metrics & plots
│
├── models/
│   └── README.md                  # Saved model info
│
├── images/
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/credit-card-churn-prediction.git
cd credit-card-churn-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the notebook**
```bash
jupyter notebook notebooks/churn_prediction.ipynb
```

**4. Or run training script directly**
```bash
python src/train.py
```

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

## 👤 Author

**[Your Name]**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
