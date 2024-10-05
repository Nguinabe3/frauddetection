import streamlit as st
import matplotlib.pyplot as plt
from src.eda import EDA
from src.show_predict_page import show_predict_page
#from explore_page import Print, Plot, Univ_analysis
import xgboost as xgb
import pickle
import pandas as pd

# current_dir = os.getcwd()
# path = os.path.join(current_dir, '../data/default_of_credit_card_clients.csv')
df = pd.read_csv("default_of_credit_card_clients.csv",skiprows=1,index_col=0)
eda = EDA(df)
# columns,stats=eda.explore()
# print(f"Columns: {columns}")
#eda.data_visualisation()
with st.sidebar:
    # Ask the user if they want to explore the data or run the model
    st.title("Do you want to explore the data or run the model?")
    action = st.radio("""Please, Choose""", ('Explore Data','Predict Default Credit Card Paiement','Data Drift'))
st.title("Welcome to Default Paiement Credit Card System")
st.write("### This system leverages advanced data analysis and machine learning techniques to streamline and enhance the default credit card paiement, ensuring accurate, fair, and efficient decisions.")
if action == 'Explore Data':
    # Ask the user to input the number of rows to display
    num_rows = st.number_input("Enter the number of rows to display:", min_value=1, max_value=len(df), value=5)
    
    # Display the head of the DataFrame based on user input
    # st.write(f"### Showing the first {num_rows} row(s) of the DataFrame")
    st.write(df.head(num_rows))
    #st.write(f"### Unique features in the the data")
    #st.write(df.nunique())
    st.write(f"### Summary of the dataset")
    st.write(df.describe())
    eda.explore()
    eda.data_visualisation()
    eda.correlation_matrix()
    eda.show_percentages()
    #eda.Isolation_Forest_Algorithm()
    

elif action == 'Predict Default Credit Card Paiement':
    # For demonstration purposes, we'll just display a message
    #st.write("### Running the model...")
    # Insert your model code here
    # For example: model_output = my_model.predict(input_data)
    # st.write(model_output)
    show_predict_page()
    #st.write("Model has been run successfully!")