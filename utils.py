import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch
from torch.utils.data import TensorDataset

def load_data(train_paths, test_path):
    train_data = pd.concat([pd.read_csv(path) for path in train_paths], ignore_index=True)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def preprocess_data(data):
    X = data.drop(columns=['DATETIME', 'ATT_FLAG'])
    y = data['ATT_FLAG']
    return X, y

def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)
    return X_bal, y_bal

def split_data(X, y, test_size, random_state):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_clients(X, y, num_clients):
    X_split = torch.tensor_split(torch.tensor(X.values, dtype=torch.float32), num_clients)
    y_split = torch.tensor_split(torch.tensor(y.values, dtype=torch.long), num_clients)
    clients = [(TensorDataset(x, y)) for x, y in zip(X_split, y_split)]
    return clients
