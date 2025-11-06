import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("🩺 Cancer Prediction")

st.write("### 🧠 Training Model Automatically Using 'Cancer_prediction.csv'")

try:
    data_df = pd.read_csv("Cancer_prediction.csv")

    # Fill missing values
    data_df = data_df.fillna(0)

    # Encode non-numeric columns
    for col in data_df.columns:
        if data_df[col].dtype == 'object' and col != 'target':
            data_df[col] = pd.Categorical(data_df[col]).codes

    # Encode target if text
    if data_df['target'].dtype == 'object':
        data_df['target'] = pd.Categorical(data_df['target']).codes

    x = data_df.drop('target', axis=1)
    y = data_df['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

    LR = LogisticRegression(solver='liblinear', random_state=42)
    LR.fit(x_train, y_train)

    y_pred = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model trained successfully! Accuracy: **{accuracy:.2f}**")

except Exception as e:
    st.error(f"❌ Error loading or training model: {e}")
    st.stop()

st.header("🔍 Enter Patient Details for Prediction")

user_input = {}
for col in x.columns:
    value = st.number_input(f"Enter value for **{col}**:", value=0.0, format="%.4f")
    user_input[col] = value

if st.button("🔮 Predict"):
    try:
        input_df = pd.DataFrame([user_input])
        prediction = LR.predict(input_df)[0]

        if prediction == 1:
            st.error("⚠️ The model predicts **Cancer PRESENT (1)**.")
        else:
            st.success("✅ The model predicts **No Cancer (0)**.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
