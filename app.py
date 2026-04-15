import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go

# Load model and scaler
model = pickle.load(open('parkinsons_model_reduced.pkl', 'rb'))
scaler = pickle.load(open('scaler_reduced.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Parkinson's Predictor", layout="centered")

# Title
st.title("🧠 Parkinson's Disease Prediction System")
st.markdown("### Enter key voice parameters")

# Feature labels (user-friendly)
features = [
    "Voice Perturbation Energy (PPE)",
    "Frequency Variation 1 (spread1)",
    "Average Vocal Frequency (Fo)",
    "Frequency Variation 2 (spread2)",
    "Minimum Vocal Frequency (Flo)",
    "Maximum Vocal Frequency (Fhi)"
]

# Input storage
input_data = []

# Input fields
for feature in features:
    val = st.number_input(feature, value=0.0)
    input_data.append(val)

# Buttons
col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🔍 Predict")

with col2:
    demo_btn = st.button("🧪 Load Sample Data")

# Demo sample (Parkinson's case)
if demo_btn:
    input_data = [
        0.284654,   # PPE
        -4.813031,  # spread1
        119.992,    # Fo
        0.266482,   # spread2
        74.997,     # Flo
        157.302     # Fhi
    ]
    st.info("Sample data loaded. Click Predict.")

# Prediction + Risk Meter
if predict_btn:
    input_array = np.asarray(input_data).reshape(1, -1)
    std_data = scaler.transform(input_array)

    prediction = model.predict(std_data)
    confidence = model.decision_function(std_data)

    # Convert confidence → risk %
    risk_score = min(max((confidence[0] + 3) / 6 * 100, 0), 100)

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Parkinson's Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if prediction[0] == 1 else "green"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"},
            ],
        }
    ))

    st.plotly_chart(fig)

    # Result message
    if prediction[0] == 0:
        st.success("🟢 Low Risk (No Parkinson's)")
    else:
        st.error("🔴 High Risk (Parkinson's Detected)")

    # Confidence warning
    if abs(confidence[0]) < 1:
        st.warning("⚠️ Low confidence prediction. Consider further medical evaluation.")
