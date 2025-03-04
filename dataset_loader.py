import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import kagglehub

# Data cleaning and pre-processing
def process_data(data, correlation_threshold=0.01):
    label_encoder = LabelEncoder()
    # Removes instances with person with age over 100 years old
    data = data[data['person_age'] < 100.0]

    # Categorical to numerical data
    categoricals = data.select_dtypes(include='object').columns
    for attribute in categoricals:
        data.loc[:, attribute] = label_encoder.fit_transform(data[attribute])

    # Remove attributes uncorrelated towards target attribute
    correlation_matrix = data.corr()
    data_correlation = correlation_matrix['loan_status'].sort_values(ascending=False)
    for attribute, correlation in data_correlation.items():
        if np.abs(correlation) < correlation_threshold:
            data = data.drop(columns=[attribute])

    return data



def load_dataset():
    path = kagglehub.dataset_download("taweilo/loan-approval-classification-data")
    dataset = pd.read_csv(path)
    processed_dataset = process_data(dataset)
    return processed_dataset