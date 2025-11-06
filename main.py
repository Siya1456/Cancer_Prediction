import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Set page configuration for better aesthetics
st.set_page_config(
    page_title="Cancer Prediction Dashboard",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🩺 Cancer Prediction Model Deployment")
st.markdown("---")

st.write("### 🧠 Training Model Automatically Using 'Cancer_prediction.csv'")

# Constants and Initialization
LR = None
x = None
accuracy = None

try:
    # --- Data Loading and Preprocessing ---
    # Attempt to load the dataset
    data_df = pd.read_csv("Cancer_prediction.csv")

    # FIX: Strip whitespace from column names to prevent KeyError if headers have leading/trailing spaces.
    data_df.columns = data_df.columns.str.strip()

    st.info(f"Loaded dataset with {data_df.shape[0]} rows and {data_df.shape[1]} columns.")

    # Fill missing values (using 0 as per original script, though median might be better for real-world)
    data_df = data_df.fillna(0)

    # Convert non-numeric columns (excluding target) to numerical codes
    for col in data_df.columns:
        # We check for the *cleaned* column name 'target'
        if data_df[col].dtype == 'object' and col != 'target':
            data_df[col] = pd.Categorical(data_df[col]).codes

    # Encode target if text
    if data_df['target'].dtype == 'object':
        data_df['target'] = pd.Categorical(data_df['target']).codes

    # Ensure all features are numerical before training
    for col in data_df.columns:
        if col != 'target':
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce').fillna(0)

    # --- Model Training ---
    # The 'target' column is now guaranteed to be clean thanks to the fix above.
    x = data_df.drop('target', axis=1)
    y = data_df['target'].astype(int)  # Ensure target is integer type

    # Check if there's enough data and features
    if x.shape[0] < 2 or x.shape[1] == 0:
        st.error("❌ Dataset is too small or has no features after preprocessing.")
        st.stop()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

    # Initialize and fit the Logistic Regression model
    LR = LogisticRegression(solver='liblinear', random_state=42)
    LR.fit(x_train, y_train)

    y_pred = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model trained successfully! Accuracy on test set: **{accuracy:.2f}**")

except FileNotFoundError:
    st.error("❌ Error: The required file 'Cancer_prediction.csv' was not found.")
    st.markdown("Please ensure the data file is in the same directory as this script.")
    st.stop()

except Exception as e:
    st.error(f"❌ Error during model training or data processing: {e}")
    st.stop()

st.markdown("---")
st.header("🔍 Enter Patient Details for Prediction")
st.markdown("Please input the features corresponding to the model's training data.")

# --- Prediction Interface ---
if x is not None:
    user_input = {}
    cols = x.columns.tolist()

    # Use Streamlit columns for a cleaner, side-by-side input layout
    col1, col2 = st.columns(2)

    # Generate input fields dynamically
    for i, col in enumerate(cols):
        current_col = col1 if i % 2 == 0 else col2

        # Use a sensible default based on the mock data (e.g., 5, but enforce min 1)
        default_val = data_df[col].mean() if col in data_df.columns else 5.0

        value = current_col.number_input(
            f"Enter value for **{col.replace('_', ' ').title()}**:",
            min_value=1.0,
            max_value=10.0,
            value=float(default_val),
            step=1.0,
            format="%.1f",
            key=col
        )
        user_input[col] = value

    st.markdown("---")

    if st.button("🔮 Predict Cancer Risk", type="primary"):
        try:
            # Create DataFrame from user input
            input_df = pd.DataFrame([user_input])

            # Ensure the input DataFrame has the columns in the correct order
            input_df = input_df[x.columns]

            prediction = LR.predict(input_df)[0]
            prediction_proba = LR.predict_proba(input_df)[0]

            st.subheader("Prediction Result")
            if prediction == 1:
                risk_level = f"{(prediction_proba[1] * 100):.2f}%"
                st.error(f"⚠️ Prediction: **CANCER PRESENT** (1)")
                st.markdown(f"**High Risk.** Probability of Cancer: **{risk_level}**")
            else:
                risk_level = f"{(prediction_proba[0] * 100):.2f}%"
                st.success(f"✅ Prediction: **NO CANCER** (0)")
                st.markdown(f"**Low Risk.** Probability of No Cancer: **{risk_level}**")

            st.balloons()

        except Exception as e:
            st.error(f"❌ Prediction failed: An internal error occurred. Details: {e}")

else:
    st.warning("Prediction interface not available. Please check the model training log above for errors.")