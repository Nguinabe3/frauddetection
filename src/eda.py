import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
import streamlit as st

class EDA:
    def __init__(self, df):
        self.df = df

    def explore(self):
        # 1. Distribution of the Target Variable
        st.write(f"### Distribution of the Target Variable")
        st.write(self.df['default payment next month'].value_counts())
        st.write(f"### Distribution (Percentage)")
        st.write(self.df['default payment next month'].value_counts(normalize=True) * 100)
        
        # Summary statistics
        st.write(f"### Some Statistics about Data")
        st.write(self.df.describe())

    def data_visualisation(self):
        # 2. Exploration of Numerical Features
        st.write("### Distribution of Numerical Features")
        numerical_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        self.df[numerical_features].hist(ax=axes.flatten())
        plt.tight_layout()
        st.pyplot(fig)

        # 3. Exploration of Categorical Features
        st.write("### Distribution of Categorical Features (Education, Marital Status)")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.countplot(data=self.df, x='EDUCATION', ax=axes[0])
        sns.countplot(data=self.df, x='MARRIAGE', ax=axes[1])
        axes[0].set_title('Education Distribution')
        axes[1].set_title('Marital Status Distribution')
        plt.tight_layout()
        st.pyplot(fig)

    def correlation_matrix(self):
      # 4. Correlation Matrix
      st.write("### Correlation Matrix")
      corrmat = self.df.corr()
      
      # Handle NaN values in the correlation matrix
      if corrmat.isnull().values.any():
         st.write("Warning: The correlation matrix contains NaN values. They will be replaced with zeros.")
         corrmat = corrmat.fillna(0)
      
      # Create the correlation matrix heatmap
      fig, ax = plt.subplots(figsize=(16, 12))  # Increase the figure size
      sns.heatmap(corrmat, vmax=0.8, square=True, annot=True, cmap='coolwarm', ax=ax, annot_kws={"size": 10}, fmt=".2f")

      # Improve axis labels for readability
      ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)
      ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
      
      # Add a title to the heatmap
      ax.set_title('Feature Correlation Matrix', fontsize=15)

      # Display the plot in Streamlit
      st.pyplot(fig)


 

