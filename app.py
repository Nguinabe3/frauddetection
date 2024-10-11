import streamlit as st
from src.eda import EDA
from src.show_predict_page import show_predict_page
import pandas as pd

# Dummy user database
users_db = {
    "Najlaa": {"password": "password1"},
    "Josue": {"password": "password1"},
    "Reham": {"password": "password1"}
}

# Authentication function
def authenticate(username, password):
    user = users_db.get(username)
    if user and user['password'] == password:
        return True
    return False

# Create session state for authentication status
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# Authentication
if not st.session_state['authenticated']:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    # Check if the user presses the "Login" button
    if st.button("Login"):
        if authenticate(username, password):
            # Update session state upon successful login
            st.session_state['authenticated'] = True
            st.success(f"Welcome, {username}!")
            # Streamlit will automatically re-render after the button is pressed, so no need for experimental_rerun
        else:
            st.error("Invalid username or password")

else:
    # Main app content after successful login
    st.title("Welcome to Default Payment Credit Card System")
    st.write("""
    ### An Interactive Tool for Credit Risk Analysis
    This app provides data exploration and machine learning techniques to analyze credit card customer data and predict payment defaults, enabling data-driven decisions for risk management.
    """)

    with st.sidebar:
        # Ask the user if they want to explore the data or run the model
        st.title("Do you want to explore the data or run the model?")
        action = st.radio("Please, Choose", ('Explore Data', 'Predict Default Credit Card Payment'))

    if action == 'Explore Data':
        # File uploader for the CSV file
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        st.write("Expected format: rows of customer data, including features like 'LIMIT_BAL', 'AGE', 'PAY_0', 'default payment next month'.")

        if uploaded_file is not None:
            # Load the uploaded file into a DataFrame
            df = pd.read_csv(uploaded_file, skiprows=1, index_col=0)
            eda = EDA(df)

            # Ask the user to input the number of rows to display
            num_rows = st.number_input("Enter the number of rows to display:", min_value=1, max_value=len(df), value=5)

            # Display the head of the DataFrame based on user input
            st.write(f"### Showing the first {num_rows} row(s) of the Dataset")
            st.write(df.head(num_rows))

            # Perform the selected EDA functions
            st.write(f"### Summary of the Dataset")
            eda.explore()
            eda.data_visualisation()
            eda.correlation_matrix()

        else:
            st.write("Please upload a CSV file to explore the data.")

    elif action == 'Predict Default Credit Card Payment':
        # Call the prediction page function
        show_predict_page()
