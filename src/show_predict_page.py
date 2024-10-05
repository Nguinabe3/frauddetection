import streamlit as st
import pickle
import pandas as pd

def load_model():
    with open('xgb_classifier_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

def show_predict_page():
    st.write("""### Please, fill some information to check on the default paiement of the person""")
    loaded_model = load_model()

    # Apply custom CSS for styling
    st.markdown("""
        <style>
            .stButton button {
                background-color: red;
                color: white;
            }
            .approved {
                color: green;
                font-weight: bold;
            }
            .rejected {
                color: red;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)

    # Collecting input data
    LIMIT_BAL = st.number_input("LIMIT_BAL", value=None, placeholder="Type a number...")  # Credit limit balance
    SEX = st.selectbox("SEX", [1, 2], help="1 for men and 2 for women")  # Gender
    EDUCATION = st.number_input("EDUCATION", min_value=0, max_value=6, placeholder="Type a number between 0 and 6...")  # Education level
    MARRIAGE = st.number_input("MARRIAGE", min_value=0, max_value=3, placeholder="Type 0 for unknown, 1 for married, 2 for single, 3 for others...")  # Marital status
    AGE = st.number_input("AGE", value=None, placeholder="Type your age...")  # Age
    PAY_0 = st.number_input("PAY_0", value=None, placeholder="Enter repayment status for the last month...")  # Repayment status for the last month
    PAY_2 = st.number_input("PAY_2", value=None, placeholder="Enter repayment status for two months ago...")  # Repayment status two months ago
    PAY_3 = st.number_input("PAY_3", value=None, placeholder="Enter repayment status for three months ago...")  # Repayment status three months ago
    PAY_4 = st.number_input("PAY_4", value=None, placeholder="Enter repayment status for four months ago...")  # Repayment status four months ago
    PAY_5 = st.number_input("PAY_5", value=None, placeholder="Enter repayment status for five months ago...")  # Repayment status five months ago
    PAY_6 = st.number_input("PAY_6", value=None, placeholder="Enter repayment status for six months ago...")  # Repayment status six months ago
    BILL_AMT1 = st.number_input("BILL_AMT1", value=None, placeholder="Enter bill statement amount for the last month...")  # Bill statement amount for the last month
    BILL_AMT2 = st.number_input("BILL_AMT2", value=None, placeholder="Enter bill statement amount for two months ago...")  # Bill statement amount two months ago
    BILL_AMT3 = st.number_input("BILL_AMT3", value=None, placeholder="Enter bill statement amount for three months ago...")  # Bill statement amount three months ago
    BILL_AMT4 = st.number_input("BILL_AMT4", value=None, placeholder="Enter bill statement amount for four months ago...")  # Bill statement amount four months ago
    BILL_AMT5 = st.number_input("BILL_AMT5", value=None, placeholder="Enter bill statement amount for five months ago...")  # Bill statement amount five months ago
    BILL_AMT6 = st.number_input("BILL_AMT6", value=None, placeholder="Enter bill statement amount for six months ago...")  # Bill statement amount six months ago
    PAY_AMT1 = st.number_input("PAY_AMT1", value=None, placeholder="Enter payment amount for the last month...")  # Payment amount for the last month
    PAY_AMT2 = st.number_input("PAY_AMT2", value=None, placeholder="Enter payment amount for two months ago...")  # Payment amount two months ago
    PAY_AMT3 = st.number_input("PAY_AMT3", value=None, placeholder="Enter payment amount for three months ago...")  # Payment amount three months ago
    PAY_AMT4 = st.number_input("PAY_AMT4", value=None, placeholder="Enter payment amount for four months ago...")  # Payment amount four months ago
    PAY_AMT5 = st.number_input("PAY_AMT5", value=None, placeholder="Enter payment amount for five months ago...")  # Payment amount five months ago
    PAY_AMT6 = st.number_input("PAY_AMT6", value=None, placeholder="Enter payment amount for six months ago...")  # Payment amount six months ago
    #default_payment_next_month = st.selectbox("Default Payment Next Month", [0, 1], help="0 for no default, 1 for default")  # Target variable (default)

    submit = st.button("Check Default Paiement Status")
    if submit:
        input_data = pd.DataFrame({
        'LIMIT_BAL': [LIMIT_BAL],  # Credit limit balance
        'SEX': [SEX],  # Gender (1 for men, 2 for women)
        'EDUCATION': [EDUCATION],  # Education level
        'MARRIAGE': [MARRIAGE],  # Marital status
        'AGE': [AGE],  # Age
        'PAY_0': [PAY_0],  # Repayment status for the last month
        'PAY_2': [PAY_2],  # Repayment status two months ago
        'PAY_3': [PAY_3],  # Repayment status three months ago
        'PAY_4': [PAY_4],  # Repayment status four months ago
        'PAY_5': [PAY_5],  # Repayment status five months ago
        'PAY_6': [PAY_6],  # Repayment status six months ago
        'BILL_AMT1': [BILL_AMT1],  # Bill statement amount for the last month
        'BILL_AMT2': [BILL_AMT2],  # Bill statement amount for two months ago
        'BILL_AMT3': [BILL_AMT3],  # Bill statement amount for three months ago
        'BILL_AMT4': [BILL_AMT4],  # Bill statement amount for four months ago
        'BILL_AMT5': [BILL_AMT5],  # Bill statement amount for five months ago
        'BILL_AMT6': [BILL_AMT6],  # Bill statement amount for six months ago
        'PAY_AMT1': [PAY_AMT1],  # Payment amount for the last month
        'PAY_AMT2': [PAY_AMT2],  # Payment amount for two months ago
        'PAY_AMT3': [PAY_AMT3],  # Payment amount for three months ago
        'PAY_AMT4': [PAY_AMT4],  # Payment amount for four months ago
        'PAY_AMT5': [PAY_AMT5],  # Payment amount for five months ago
        'PAY_AMT6': [PAY_AMT6],  # Payment amount for six months ago
        #'default payment next month': [default_payment_next_month]  # Target variable (default)
    })

        # Encoding categorical variables
        # le = LabelEncoder()
        # input_data['education'] = le.fit_transform(input_data['education'])
        # input_data['self_employed'] = le.fit_transform(input_data['self_employed'])

        # Prediction using the loaded model
        prediction = loaded_model.predict(input_data)

        # Display the prediction result
        if prediction[0] == 0:
            st.markdown('<p class="approved">Default Paiement: Approved</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="rejected">Default Paiement: Rejected</p>', unsafe_allow_html=True)

        st.write(f"### Client Information")
        st.write(input_data)