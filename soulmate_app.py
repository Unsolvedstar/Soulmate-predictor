import streamlit as st
import pandas as pd
import joblib

# Title
st.title("💘 Soulmate Probability Estimator")

st.markdown("Estimate your chances of meeting your soulmate based on your lifestyle and preferences.")

# Sidebar inputs
st.sidebar.header("Your Information")

age = st.sidebar.slider("Age", 18, 60, 25)
people_met_per_year = st.sidebar.slider("People Met Per Year", 0, 1000, 100)
social_events_per_month = st.sidebar.slider("Social Events Per Month", 0, 30, 4)

personality_type = st.sidebar.selectbox("Personality Type", [
    "INTJ", "ENTP", "INFJ", "ENFP", "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "INFP", "ENFJ", "ISTP", "ISFP", "ESTP", "ESFP", "INTP", "ENTJ"
])

location = st.sidebar.selectbox("Location", [
    "Urban", "Suburban", "Rural"
])

# Prepare input dataframe with raw categorical columns (not encoded yet)
input_data = pd.DataFrame([{
    "age": age,
    "people_met_per_year": people_met_per_year,
    "social_events_per_month": social_events_per_month,
    "personality_type": personality_type,
    "location": location
}])

try:
    # Load model first
    model = joblib.load("soulmate_model.pkl")

    # One-hot encode input_data exactly as during training
    input_data_encoded = pd.get_dummies(input_data, columns=['personality_type', 'location'], drop_first=True)

    # Add any missing columns that model expects
    expected_cols = model.feature_names_in_
    for col in expected_cols:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0

    # Reorder columns to match model's expected order
    input_data_encoded = input_data_encoded[expected_cols]

    # Predict
    prediction = model.predict(input_data_encoded)[0]

    # Display result
    st.subheader("🔮 Predicted Soulmate Probability:")
    st.metric(label="Chance (%)", value=f"{round(prediction * 100, 2)}%")

except FileNotFoundError:
    st.error("Model file 'soulmate_model.pkl' not found. Please ensure the model is trained and saved.")
except Exception as e:
    st.error(f"An error occurred: {e}")
