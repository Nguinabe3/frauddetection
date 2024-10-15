import streamlit as st
import requests
import pandas as pd
import joblib
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import streamlit.components.v1 as components

# FastAPI backend URL (update this if it's hosted elsewhere)
FASTAPI_URL = "http://127.0.0.1:8000"

# Helper function to authenticate and get JWT token
def get_jwt_token(username, password):
    auth_url = f"{FASTAPI_URL}/token"
    auth_data = {"username": username, "password": password}
    
    try:
        response = requests.post(auth_url, data=auth_data)
        if response.status_code == 200:
            token = response.json().get("access_token")
            return token
        else:
            st.error(f"Authentication failed: {response.json()['detail']}")
            return None
    except Exception as e:
        st.error(f"Error during authentication: {e}")
        return None

# Streamlit layout
st.title("Default Payment Credit Card Prediction")
st.write("### Login to get started")

# Input fields for authentication
username = st.text_input("Username")
password = st.text_input("Password", type="password")

# Create a session state for storing the token
if "jwt_token" not in st.session_state:
    st.session_state.jwt_token = None

# Authentication
if st.button("Login"):
    token = get_jwt_token(username, password)
    if token:
        st.session_state.jwt_token = token
        st.success("Logged in successfully!")

# Show the prediction form only if authenticated
if st.session_state.jwt_token:
    st.write("---")
    st.write("### Enter the details below to predict whether a credit card payment will be defaulted")

    # Input fields for the 23 features
    LIMIT_BAL = st.number_input("Limit Balance", value=50000.0)
    SEX = st.selectbox("Sex (1 = Male, 2 = Female)", [1, 2])
    EDUCATION = st.selectbox("Education (1 = Graduate, 2 = University, 3 = High School)", [1, 2, 3])
    MARRIAGE = st.selectbox("Marriage Status (1 = Married, 2 = Single, 3 = Others)", [1, 2, 3])
    AGE = st.number_input("Age", value=30)
    PAY_0 = st.number_input("PAY_0 (Repayment status in September)", value=0)
    PAY_2 = st.number_input("PAY_2 (Repayment status in August)", value=0)
    PAY_3 = st.number_input("PAY_3 (Repayment status in July)", value=0)
    PAY_4 = st.number_input("PAY_4 (Repayment status in June)", value=0)
    PAY_5 = st.number_input("PAY_5 (Repayment status in May)", value=0)
    PAY_6 = st.number_input("PAY_6 (Repayment status in April)", value=0)
    BILL_AMT1 = st.number_input("Bill Amount (September)", value=5000.0)
    BILL_AMT2 = st.number_input("Bill Amount (August)", value=4000.0)
    BILL_AMT3 = st.number_input("Bill Amount (July)", value=3000.0)
    BILL_AMT4 = st.number_input("Bill Amount (June)", value=2000.0)
    BILL_AMT5 = st.number_input("Bill Amount (May)", value=1000.0)
    BILL_AMT6 = st.number_input("Bill Amount (April)", value=500.0)
    PAY_AMT1 = st.number_input("Payment Amount (September)", value=1000.0)
    PAY_AMT2 = st.number_input("Payment Amount (August)", value=1000.0)
    PAY_AMT3 = st.number_input("Payment Amount (July)", value=1000.0)
    PAY_AMT4 = st.number_input("Payment Amount (June)", value=1000.0)
    PAY_AMT5 = st.number_input("Payment Amount (May)", value=1000.0)
    PAY_AMT6 = st.number_input("Payment Amount (April)", value=1000.0)

    # Button to trigger prediction
    if st.button("Predict"):
        # Prepare the data in the correct format
        input_data = {
            "LIMIT_BAL": LIMIT_BAL,
            "SEX": SEX,
            "EDUCATION": EDUCATION,
            "MARRIAGE": MARRIAGE,
            "AGE": AGE,
            "PAY_0": PAY_0,
            "PAY_2": PAY_2,
            "PAY_3": PAY_3,
            "PAY_4": PAY_4,
            "PAY_5": PAY_5,
            "PAY_6": PAY_6,
            "BILL_AMT1": BILL_AMT1,
            "BILL_AMT2": BILL_AMT2,
            "BILL_AMT3": BILL_AMT3,
            "BILL_AMT4": BILL_AMT4,
            "BILL_AMT5": BILL_AMT5,
            "BILL_AMT6": BILL_AMT6,
            "PAY_AMT1": PAY_AMT1,
            "PAY_AMT2": PAY_AMT2,
            "PAY_AMT3": PAY_AMT3,
            "PAY_AMT4": PAY_AMT4,
            "PAY_AMT5": PAY_AMT5,
            "PAY_AMT6": PAY_AMT6
        }

        try:
            # Send POST request to FastAPI backend
            headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
            response = requests.post(f"{FASTAPI_URL}/predict-fraud/", json=input_data, headers=headers)

            # Check if request is successful
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: {result['prediction']}")
            else:
                st.error(f"Error: {response.status_code}, {response.text}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Adding data drift detection
    st.write("---")
    st.write("### Data Drift Detection")

    # Load the dataset and the model
    df = pd.read_csv('default_of_credit_card_clients.csv', skiprows=1, index_col=0)
    df.drop(columns=['default payment next month'], inplace=True)

    # Load the pre-trained model
    loaded_model = joblib.load('xgb_classifier_model.pkl')

    # Get feature importances from the model
    importances = loaded_model.feature_importances_

    # Create a DataFrame to map feature names to their importance values
    feature_importance_df = pd.DataFrame({'Feature': df.columns, 'Importance': importances})

    # Sort the DataFrame by importance values and select the top features
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    df_import = feature_importance_df[feature_importance_df['Importance'] >= 0.02].Feature
    col_impp = list(df_import)

    # Create data samples for reference and testing
    data = df[col_impp]
    sample_ref = data.iloc[100:200]  # Reference data
    sample_test = data.iloc[800:900]  # Test data (could be new or current data)

    # Initialize the drift report
    report = Report(metrics=[DataDriftPreset()])

    # Run the drift detection on the samples
    report.run(reference_data=sample_ref, current_data=sample_test)

    # Save the report as an HTML file
    report_file = "drift_report.html"
    report.save_html(report_file)

    # Read and display the HTML report in Streamlit
    with open(report_file, "r") as f:
        report_html = f.read()

    # Render the HTML in Streamlit
    st.write("### Data Drift Report")
    components.html(report_html, height=1000)  # Display the report in Streamlit
