
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Model location
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

# App title
st.title("📊 Customer Churn Prediction App")
st.write(
    "Analyze customer information and predict "
    "the risk of customer churn."
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    st.success("Machine learning model loaded successfully!")
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

st.divider()

# Customer information
st.header("👤 Enter Customer Information")

col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col2:
    tenure = st.number_input(
        "Customer Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        value=70.0
    )

st.divider()

st.subheader("🔍 Customer Churn Prediction")


# Additional customer information
st.subheader("Additional Customer Details")

col3, col4 = st.columns(2)

with col3:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )
    security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )
    backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

with col4:
    protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )
    support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )
    tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )
    movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )
    paperless = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=840.0
    )

st.divider()
st.subheader("🔍 Customer Churn Prediction")

if st.button("Predict Customer Churn", type="primary"):

    customer = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_df = pd.DataFrame([customer])

    # Match the trained model's feature order
    input_df = input_df[model.feature_names_in_]

    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Results")

        st.metric(
            "Customer Churn Probability",
            f"{probability * 100:.2f}%"
        )

        if prediction == 1:
            st.error("High Risk: Customer May Churn")
        else:
            st.success("Low Risk: Customer Likely to Stay")

        st.progress(float(probability))

        st.subheader("Business Recommendations")

        if probability >= 0.5:
            st.warning("Customer Retention Action Recommended")

            st.write(
                "• Contact the customer with a personalized retention offer."
            )
            st.write(
                "• Review monthly charges and potential discounts."
            )
            st.write(
                "• Offer technical support and service improvements."
            )
            st.write(
                "• Explore longer-term contract incentives."
            )

        else:
            st.success("Customer Retention Status: Favorable")

            st.write(
                "• Maintain customer satisfaction through regular support."
            )
            st.write(
                "• Offer loyalty rewards when appropriate."
            )
            st.write(
                "• Monitor changes in customer usage and satisfaction."
            )


    except Exception as e:
        st.error(f"Prediction error: {e}")

st.divider()

st.caption(
    "Customer Churn Prediction | "
    "Python • Machine Learning • Streamlit"
)
