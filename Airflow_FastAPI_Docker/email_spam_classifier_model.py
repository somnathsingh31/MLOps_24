import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

def data_preprocessing(file_path):
    email = pd.read_csv(file_path)
    x = email['text']
    y = email['label']

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.23)
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    xtrain = feature_extraction.fit_transform(xtrain)
    xtest = feature_extraction.transform(xtest)

    ytrain = ytrain.astype('int')
    ytest = ytest.astype('int')

    return xtrain, xtest, ytrain, ytest

def train_model(xtrain, ytrain):
    logistic_reg = LogisticRegression()
    logistic_reg.fit(xtrain, ytrain)
    return logistic_reg

def evaluate_model(trained_model, xtest, ytest):
    predictions = trained_model.predict(xtest)
    accuracy = accuracy_score(ytest, predictions)
    print(f"Accuracy: {accuracy}")
    return accuracy


file_path = f'MLOps\Data\combined_data.csv'

xtrain, xtest, ytrain, ytest = data_preprocessing(file_path)
trained_model = train_model(xtrain, ytrain)
accuracy = evaluate_model(trained_model, xtest, ytest)

# Save the model to a file
joblib.dump(trained_model, 'spam_email_classifier.pkl')