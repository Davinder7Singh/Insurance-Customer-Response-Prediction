import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from openai import OpenAI  


# --- Groq API Setup ---
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="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"  
)


# --- Load trained model and scaler ---
model = load('random_forest_model.pkl')
scaler = load('scaler.pkl')


st.set_page_config(page_title="Insurance Prediction", layout="centered")
st.markdown("<h1 style='text-align: center;'>Insurance Policy Purchase Prediction</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("Provide the following details to predict if a customer will purchase an insurance policy.")


# --- Create input columns ---
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    age = st.slider('Age', 18, 100, 30)
    vehicle_age = st.selectbox('Vehicle Age', ['< 1 Year', '1-2 Year', '> 2 Years'])
    annual_premium = st.number_input('Annual Premium', min_value=0.0, value=30000.0)
with col2:
    region_code = st.number_input('Region Code', min_value=0, max_value=999, value=28)
    previously_insured = st.selectbox('Previously Insured', ['Yes', 'No'])
    policy_sales_channel = st.number_input('Policy Sales Channel', min_value=0, max_value=999, value=26)
with col3:
    vehicle_damage = st.selectbox('Vehicle Damage', ['Yes', 'No'])
    vintage = st.slider('Vintage (days)', 0, 300, 100)


# --- Map categorical values ---
vehicle_age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
vehicle_damage_map = {'Yes': 1, 'No': 0}
previously_insured_map = {'Yes': 1, 'No': 0}


# --- Create input DataFrame ---
input_data = pd.DataFrame([[ 
    age,
    region_code,
    previously_insured_map[previously_insured],
    vehicle_age_map[vehicle_age],
    vehicle_damage_map[vehicle_damage],
    annual_premium,
    policy_sales_channel,
    vintage
]], columns=[
    'Age',
    'Region_Code',
    'Previously_Insured',
    'Vehicle_Age',
    'Vehicle_Damage',
    'Annual_Premium',
    'Policy_Sales_Channel',
    'Vintage'
])


# --- Scale the data ---
input_data_scaled = scaler.transform(input_data)
if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)[0][1]
    if prediction[0] == 1:
        st.success(f"✅ Likely to purchase the policy (Probability: {prediction_proba:.2f})")
    else:
        st.warning(f"❌ Unlikely to purchase the policy (Probability: {prediction_proba:.2f})")

        # --- Prepare prompt for LLaMA 3 ---
        prompt = f"""
        Customer Details:
        - Age: {age}
        - Region Code: {region_code}
        - Previously Insured: {previously_insured}
        - Vehicle Age: {vehicle_age}
        - Vehicle Damage: {vehicle_damage}
        - Annual Premium: {annual_premium}
        - Policy Sales Channel: {policy_sales_channel}
        - Vintage: {vintage}
        The model predicts that this customer is unlikely to purchase the policy.
        Provide personalized, actionable suggestions to increase the likelihood of policy purchase.
        """


        # --- Generate suggestions using LLaMA 3 ---
        with st.spinner('Generating personalized suggestions...'):
            response = client.chat.completions.create(
                model="llama3-70b-8192",  # Groq's LLaMA 3 70B model
                messages=[
                    {"role": "system", "content": "You are an expert insurance sales advisor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            suggestions = response.choices[0].message.content
        st.info(f"💡 Suggestions:\n\n{suggestions}")
