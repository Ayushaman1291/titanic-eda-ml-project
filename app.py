import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load('model/titanic_model.pkl')
df = pd.read_csv('data/train.csv')

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")
st.title("🚢 Titanic Survival Predictor")
st.markdown("Exploring survival patterns through data analysis and predictive modeling.")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Predict", "EDA Charts", "Model Info", "Data Explorer"])

# ── TAB 1: PREDICT ──
with tab1:
    st.subheader("Passenger Survival Prediction")
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = First, 2 = Second, 3 = Third")
        sex = st.radio("Sex", ["Male", "Female"])
        age = st.slider("Age", 1, 80, 25)
        sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)

    with col2:
        parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
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
    if st.button("Generate Prediction", use_container_width=True):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        col_a, col_b = st.columns(2)
        with col_a:
            if prediction == 1:
                st.success("✅ Survived — This passenger likely would have survived.")
                st.balloons()
            else:
                st.error("❌ Did Not Survive — This passenger likely would not have survived.")

        with col_b:
            st.metric("Survival Probability", f"{probability:.1%}")
            st.progress(float(probability))

    st.divider()
    st.caption("Model: Gradient Boosting Classifier | Accuracy: 82.27% | Dataset: Kaggle Titanic")

# ── TAB 2: EDA CHARTS ──
with tab2:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5,4))
        sns.countplot(x='Survived', data=df, palette='Set2', ax=ax)
        ax.set_title('Survival Count')
        ax.set_xticklabels(['Did Not Survive', 'Survived'])
        ax.set_xlabel('')
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5,4))
        sns.countplot(x='Sex', hue='Survived', data=df, palette='Set2', ax=ax)
        ax.set_title('Survival by Gender')
        ax.legend(['Did Not Survive', 'Survived'])
        ax.set_xlabel('')
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(5,4))
        sns.countplot(x='Pclass', hue='Survived', data=df, palette='Set2', ax=ax)
        ax.set_title('Survival by Passenger Class')
        ax.legend(['Did Not Survive', 'Survived'])
        ax.set_xlabel('Passenger Class')
        st.pyplot(fig)
        plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(5,4))
        sns.histplot(df['Age'].dropna(), bins=30, kde=True, color='steelblue', ax=ax)
        ax.set_title('Age Distribution of Passengers')
        ax.set_xlabel('Age')
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.subheader("Key Statistics")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Overall Survival Rate", f"{df['Survived'].mean():.1%}")
    col_s2.metric("Female Survival Rate", f"{df[df['Sex']=='female']['Survived'].mean():.1%}")
    col_s3.metric("Male Survival Rate", f"{df[df['Sex']=='male']['Survived'].mean():.1%}")
    col_s4.metric("1st Class Survival Rate", f"{df[df['Pclass']==1]['Survived'].mean():.1%}")

# ── TAB 3: MODEL INFO ──
with tab3:
    st.subheader("Model Performance Comparison")

    results = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost'],
        'Accuracy': ['79.12%', '80.81%', '82.27%', '81.48%'],
        'Std Dev': ['±0.0185', '±0.0270', '±0.0178', '±0.0266'],
        'Selected': ['', '', '✅', '']
    })
    st.dataframe(results, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Selected Model: Gradient Boosting Classifier")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "82.27%")
    col2.metric("Evaluation Method", "5-Fold Cross Validation")
    col3.metric("Std Deviation", "±0.0178")

    st.divider()
    st.subheader("Feature Importance Analysis")
    features = ['Sex', 'Fare', 'Pclass', 'Age', 'SibSp', 'Embarked_S', 'Parch', 'Embarked_Q']
    importance = [0.42, 0.18, 0.16, 0.13, 0.05, 0.03, 0.02, 0.01]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=importance, y=features, palette='viridis', ax=ax)
    ax.set_title('Feature Importance — Gradient Boosting Classifier')
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    st.pyplot(fig)
    plt.close()

# ── TAB 4: DATA EXPLORER ──
with tab4:
    st.subheader("Dataset Overview")
    st.markdown(f"**Total Passengers:** {len(df)} | **Survived:** {df['Survived'].sum()} | **Did Not Survive:** {len(df)-df['Survived'].sum()}")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        pclass_filter = st.multiselect("Filter by Class", [1, 2, 3], default=[1, 2, 3])
    with col2:
        sex_filter = st.multiselect("Filter by Sex", ["male", "female"], default=["male", "female"])

    filtered = df[(df['Pclass'].isin(pclass_filter)) & (df['Sex'].isin(sex_filter))]
    st.dataframe(filtered[['Name','Pclass','Sex','Age','Fare','Survived']].reset_index(drop=True),
                 use_container_width=True)
    st.caption(f"Displaying {len(filtered)} of 891 records")