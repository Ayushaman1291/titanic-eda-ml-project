# 🚢 Titanic Survival Predictor

🔗 **Live Demo:** [Click here to try the app](https://titanic-eda-ml-project-8j6mbycyjuqc6fozq6ij3i.streamlit.app/)

---
---

## 📌 Project Overview

This project performs end-to-end machine learning on the Titanic dataset — from exploratory data analysis to model deployment. It compares multiple ML models and serves the best one through an interactive Streamlit web app.

---

## 📊 Key Findings from EDA

- **Female passengers** had a ~74% survival rate vs ~19% for males
- **1st class passengers** survived at nearly double the rate of 3rd class
- **Children (age < 10)** had a higher survival rate than adults
- **Fare and Pclass** were the strongest predictors of survival after Sex

---

## 🤖 Model Comparison

| Model | Accuracy (5-Fold CV) |
|---|---|
| Logistic Regression | 79.12% |
| Random Forest | 80.81% |
| Gradient Boosting ✅ | **82.27%** |
| XGBoost | 81.48% |

**Winner: Gradient Boosting Classifier**

---

## 🛠️ Tech Stack

- **Python** — core language
- **Pandas, NumPy** — data manipulation
- **Matplotlib, Seaborn** — data visualization
- **Scikit-learn, XGBoost** — machine learning
- **Streamlit** — web app deployment
- **Joblib** — model serialization

---

## 📁 Project Structure
titanic-eda-ml-project/
│
├── data/
│   └── train.csv
├── model/
│   └── titanic_model.pkl
├── notebooks/
│   └── 01_eda.ipynb
├── app.py
├── requirements.txt
├── .gitignore
└── README.md


---

## 🚀 Run Locally

```bash
git clone https://github.com/Ayushaman1291/titanic-eda-ml-project.git
cd titanic-eda-ml-project
pip install -r requirements.txt
streamlit run app.py
```

---

## 📚 Dataset

- Source: [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

---

## 👤 Author

**Ayush Aman**
B.Tech — Artificial Intelligence & Data Science (2nd Year)
[GitHub](https://github.com/Ayushaman1291)
