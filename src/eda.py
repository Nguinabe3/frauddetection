import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import streamlit as st

class EDA:
   
   def __init__(self, df):
     self.df = df
   

   def explore(self):
      #columns = self.df.columns
      stats = self.df.describe()
      # st.write(f"### Columns Names")
      # st.write(columns)
      st.write(f"### Somes Statistiques about data")
      st.write(stats)
      #return columns,stats
   
   def data_visualisation(self):
      st.write("### Histogram of All Features")
      fig, ax = plt.subplots(figsize=(20, 20))
      self.df.hist(ax=ax)
      st.pyplot(fig)

   def correlation_matrix(self):
      st.write(f"### Correlation Matrix")
      
      # Compute the correlation matrix
      corrmat = self.df.corr()

      # Check and handle NaN values in the correlation matrix
      if corrmat.isnull().values.any():
         st.write("Warning: The correlation matrix contains NaN values. They will be replaced with zeros.")
         corrmat = corrmat.fillna(0)  # Replace NaN with 0 (or handle as necessary)
      
      # Create a figure for the heatmap
      fig, ax = plt.subplots(figsize=(12, 9))  # Set figure size
      
      # Create the heatmap with seaborn, plotting onto 'ax'
      sns.heatmap(corrmat, vmax=0.8, square=True, annot=True, cmap='coolwarm', ax=ax)
      
      # Render the plot in Streamlit
      st.pyplot(fig)

   def show_percentages(self):
      st.write(f"### Percentages Observation")
      
      # Filter for fraudulent and valid transactions
      fraudulent_transactions = self.df[self.df['default payment next month'] == 1]
      valid_transactions = self.df[self.df['default payment next month'] == 0]
      
      # Calculate the outlier fraction, adding a check to avoid division by zero
      if len(valid_transactions) > 0:
         outlier_fraction = float(len(fraudulent_transactions)) / float(len(valid_transactions))
      else:
         outlier_fraction = float('inf')  # Assign a special value (e.g., infinity) if no valid transactions exist
      
      # Display statistics
      st.write(f"Fraudulent transactions: {len(fraudulent_transactions)}")
      st.write(f"Valid transactions: {len(valid_transactions)}")

      # Handle special case where there are no valid transactions
      if len(valid_transactions) > 0:
         st.write(f"Outlier fraction: {outlier_fraction}")
         st.write(f"Outlier %: {round(outlier_fraction * 100, 2)}")
      else:
         st.write("No valid transactions found, cannot compute outlier fraction or percentage.")

      #return fraudulent_transactions,valid_transactions,outlier_fraction#f"Fraudent transactions: {len(fraudulent_transactions)},Valid transactions: {len(valid_transactions)},Outlier fraction: {outlier_fraction},Outlier %: {round(outlier_fraction*100, 2)}"
   
   def Isolation_Forest_Algorithm(self):
      _,_,outlier_fraction = self.show_percentages()
      columns = self.df.columns.tolist()
      columns = [c for c in columns if c not in ["Class"]]
      target = "Class"
      X = self.df[columns]
      y = self.df[target]
      clf = IsolationForest(max_samples = len(X), contamination = outlier_fraction, random_state = 1)
      clf.fit(X)
      scores_pred = clf.decision_function(X)
      y_pred = clf.predict(X) # Isolation Forest predicts 1 for valid transactions, while it predicts -1 for fraudulent
      print(f"Valid transactions predicted by Isolation Forest: {len(y_pred[y_pred == 1])}") # Valid
      print(f"Fraudulent transactions predicted by Isolation Forest: {len(y_pred[y_pred == -1])}") # Fraudulent

      # A transformation in required in y_pred array, as Isolation Forest predicts 1 for valid transactions, while it predicts -1 for fraudulent, but in the
      # dataset, we have "Class" variable as 0 for valid transactions, 1 for fraudulent transactions.

      y_pred[y_pred == 1] = 0
      y_pred[y_pred == -1] = 1
      print(f"Accuracy score: {accuracy_score(y, y_pred)}")

      n_errors = (y != y_pred).sum()
      print(f"Total incorrect predictions: {n_errors}")
      print(classification_report(y, y_pred))

      clf = LocalOutlierFactor(n_neighbors = 20, contamination = outlier_fraction)
      y_pred = clf.fit_predict(X)
      scores_pred = clf.negative_outlier_factor_

      print(f"Valid transactions predicted by Local Outlier Factor: {len(y_pred[y_pred == 1])}") # Valid
      print(f"Fraudulent transactions predicted by Local Outlier Factor: {len(y_pred[y_pred == -1])}") # Fraudulent
      y_pred[y_pred == 1] = 0
      y_pred[y_pred == -1] = 1
      print(f"Accuracy score: {accuracy_score(y, y_pred)}")
      n_errors = (y != y_pred).sum()
      print(f"Total incorrect predictions: {n_errors}")
      print(classification_report(y, y_pred))  

# df = pd.read_csv("data/default_of_credit_card_clients.csv",skiprows=1,index_col=0)
# eda = EDA(df)
# # columns,stats=eda.explore()
# # print(f"Columns: {columns}")
# eda.data_visualisation()