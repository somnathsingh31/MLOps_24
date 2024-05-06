from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def data_preprocessing():
    # Read the list of emails from the text file
    emails = []
    with open(r'/airflow/data/emails.txt', "r") as file:
        for line in file:
            emails.append(line.strip())

    feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)
    sample_features = feature_extraction.transform(emails)
    return sample_features

def load_model():
    model_path = r'/airflow/models/spam_email_classifier.pkl'
    return joblib.load(model_path)

def classification_output(loaded_model, email_samples):
    #email_samples are preprossed email obtained from above
    predictions = loaded_model.predict(email_samples)
    spam_class = {0: 'Not Spam', 1: 'Spam'}
    results = [spam_class[element] for element in predictions]
    return results

with DAG(dag_id='email_spam_classification', start_date=datetime.now(), schedule_interval=None) as dag:
    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing',
        python_callable=data_preprocessing
    )

    load_model_task = PythonOperator(
        task_id='train_model',
        python_callable=load_model,
    )

    classification_output_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=classification_output,
        op_args=[
            '{{ task_instance.xcom_pull(task_ids="load_model") }}',
            '{{ task_instance.xcom_pull(task_ids="data_preprocessing") }}'
        ]
    )

    # Define dependencies
    data_preprocessing_task >> load_model_task >> classification_output_task
