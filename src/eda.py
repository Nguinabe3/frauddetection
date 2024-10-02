import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class EDA:
   
   def __init__(self, df):
     self.df = df
   

   def explore(self):
      columns = self.df.columns
      stats = self.df.describe()
      return columns,stats
   
   def data_visualisation(self):
      self.df.hist(figsize = (20, 20))
      plt.show()

   def correlation_matrix(self):
      corrmat = self.df.corr()
      fig = plt.figure(figsize = (12, 9))

      sns.heatmap(corrmat, vmax = 0.8, square = True)
      plt.show()

   def show_percentages(self):
      
      fraudulent_transactions = self.df[self.df['Class'] == 1]
      valid_transactions = self.df[self.df['Class'] == 0]

      outlier_fraction = float(len(fraudulent_transactions)) / float(len(valid_transactions))
      print(f"Fraudent transactions: {len(fraudulent_transactions)}")
      print(f"Valid transactions: {len(valid_transactions)}")
      print(f"Outlier fraction: {outlier_fraction}")
      print(f"Outlier %: {round(outlier_fraction*100, 2)}")

      return fraudulent_transactions,valid_transactions,outlier_fraction#f"Fraudent transactions: {len(fraudulent_transactions)},Valid transactions: {len(valid_transactions)},Outlier fraction: {outlier_fraction},Outlier %: {round(outlier_fraction*100, 2)}"
   
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

df = pd.read_csv("/Users/nguinabejosue/Desktop/Fraud_Detection/data/creditcard.csv")
eda = EDA(df)
# columns,stats=eda.explore()
# print(f"Columns: {columns}")
eda.data_visualisation()