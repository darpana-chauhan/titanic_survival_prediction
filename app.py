import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model
model = pickle.load(open("titanic_model.pkl", "rb"))

# App Title
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Prediction")
st.markdown("### Enter passenger details to see if they would survive.")

st.markdown("---")

# User Input Section
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex == "Male" else 0
    age = st.number_input("Age", 0, 100, 25)

with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
    parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
    fare = st.number_input("Fare", 0.0, 600.0, 32.0)
    embarked = st.selectbox("Embarked Port", ["S", "C", "Q"])
    embarked_map = {"S": 0, "C": 1, "Q": 2}
    embarked = embarked_map[embarked]

st.markdown("---")

# Predict Button
if st.button("🔍 Predict Survival"):
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    
    # Survival prediction
    result = model.predict(input_data)[0]

    # Probability
    prob = model.predict_proba(input_data)[0]

    survive_prob = round(prob[1] * 100, 2)
    not_survive_prob = round(prob[0] * 100, 2)

    # Result Display
    if result == 1:
        st.success("🎉 **Passenger is likely to SURVIVE!**")
    else:
        st.error("⚠️ **Passenger is NOT likely to survive.**")

    st.markdown("### 📊 Survival Probability:")
    
    # Chart
    st.progress(survive_prob / 100)
    st.write(f"🟩 **Survive: {survive_prob}%**")
    st.write(f"🟥 **Not Survive: {not_survive_prob}%**")

    # Bar Graph
    st.bar_chart(
        pd.DataFrame({
            'Probability': [survive_prob, not_survive_prob]
        }, index=["Survive", "Not Survive"])
    )
