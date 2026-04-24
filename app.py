import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model/titanic_model.pkl')

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Predictor")
st.markdown("Would **you** have survived the Titanic? Fill in your details below.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = First, 2 = Second, 3 = Third")
    sex = st.radio("Sex", ["Male", "Female"])
    age = st.slider("Age", 1, 80, 25)
    sibsp = st.number_input("Siblings/Spouses aboard", 0, 8, 0)

with col2:
    parch = st.number_input("Parents/Children aboard", 0, 6, 0)
    fare = st.number_input("Fare Paid (£)", 0.0, 520.0, 32.0)
    embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])

embarked_q = 1 if embarked == "Queenstown" else 0
embarked_s = 1 if embarked == "Southampton" else 0

input_data = pd.DataFrame([[
    pclass,
    1 if sex == "Female" else 0,
    age, sibsp, parch, fare,
    embarked_q, embarked_s
]], columns=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_Q','Embarked_S'])

st.divider()
if st.button("🔍 Predict My Survival", use_container_width=True):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ You would have SURVIVED! (Confidence: {probability:.1%})")
        st.balloons()
    else:
        st.error(f"❌ You would NOT have survived. (Confidence: {1-probability:.1%})")

st.divider()
st.caption("Model: Gradient Boosting | Accuracy: 82.27% | Dataset: Kaggle Titanic")