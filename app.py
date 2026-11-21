import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

MODEL_PATH = "models/best_model.pkl"

st.title("❤️ Heart Disease Prediction System")
st.write(
    "Enter the patient's information below to generate "
    "an educational machine learning prediction."
)
if not os.path.exists(MODEL_PATH):
    st.error("Model files not found. Please train the model first.")
    st.stop()
model = joblib.load(MODEL_PATH)
st.info(f"Selected model: {model.model_name}")
st.subheader("Patient Information")
patient_data = {}
col1, col2 = st.columns(2)

for index, feature in enumerate(model.features):
    container = col1 if index % 2 == 0 else col2
    with container:
        if feature in model.categorical:
            options = model.feature_values[feature]
            if not options:
                st.warning(f"No valid values found for {feature}")
                st.stop()
            default = model.categorical_defaults[feature]
            default_index = options.index(default)
            patient_data[feature] = st.selectbox(
                feature.replace("_", " ").title(),
                options=options,
                index=default_index,
                key=feature
            )
        else:
            default = model.defaults[feature]
            patient_data[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=float(default),
                key=feature
            )
st.divider()
if st.button("Predict Heart Disease", type="primary"):
    try:
        input_df = pd.DataFrame(
            [patient_data],
            columns=model.features
        )
        prediction = int(model.predict(input_df)[0])
        probability = float(
            model.predict_proba(input_df)[0][1]
        )
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("Model Prediction: Heart Disease Indicated")
        else:
            st.success("Model Prediction: No Heart Disease Indicated")

        st.metric(
            "Model-estimated probability for positive class",
            f"{probability * 100:.2f}%"
        )
        st.progress(probability)
        st.warning(
            "This is an educational machine learning output, "
            "not a medical diagnosis. It must not be used to "
            "make clinical decisions. Consult a qualified doctor."
        )
    except Exception as error:
        st.error(f"Prediction failed: {error}")
