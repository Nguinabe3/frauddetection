import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,precision_recall_curve, auc
import pandas as pd
import joblib
import matplotlib.pyplot as plt



def train(data_path):
    # Load dataset (assuming it's a CSV file)
    #data_path = 'creditcard.csv'  # Replace with actual path if different
    df = pd.read_csv(data_path,skiprows=1,index_col=0)

    # Separate the features and target variable
    X = df.drop('default payment next month',axis=1)  # Replace 'target_column' with your actual target column
    y = df['default payment next month']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',  # Since it's a classification task
        eval_metric='logloss',        # Evaluation metric
        use_label_encoder=False       # Disable label encoding
    )

    xgb_clf.fit(X_train, y_train)

        # Make predictions
    y_pred = xgb_clf.predict(X_test)
    y_proba = xgb_clf.predict_proba(X_test)[:, 1]
    

    # Save the model
    joblib.dump(xgb_clf, 'xgb_classifier_model.pkl')
    # Evaluate the results
    # Output the AUPRC value and save the model path
    #auprc, model_filename
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('ROC AUC Score:', roc_auc_score(y_test, y_proba))
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    auprc = auc(recall, precision)

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
# data_path = "/Users/nguinabejosue/Desktop/Fraud_Detection/data/default_of_credit_card_clients.csv"
# train(data_path)