import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
import pickle
import matplotlib.pyplot as plt



def train(data_path):
    # Load dataset (assuming it's a CSV file)
    #data_path = 'creditcard.csv'  # Replace with actual path if different
    data = pd.read_csv(data_path)

    # Separate the features and target variable
    X = data.drop(columns=['Class'])
    y = data['Class']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Build a RandomForest model
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train_res, y_train_res)

    # Predict probabilities
    y_probs = model.predict_proba(X_test)[:, 1]

    # Calculate Precision-Recall curve and AUPRC
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    auprc = auc(recall, precision)

    # Save the model
    model_filename = 'fraud_detection_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    # Output the AUPRC value and save the model path
    #auprc, model_filename

    # Adding the plot of the Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, label=f'AUPRC = {auprc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.grid(True)

    # Save the plot to a file
    plot_filename = 'precision_recall_curve.png'
    plt.savefig(plot_filename)

    # Display the plot and return the AUPRC value along with plot file path
    plt.show(), auprc, plot_filename
data_path = "/Users/nguinabejosue/Desktop/Fraud_Detection/data/creditcard.csv"
train(data_path)