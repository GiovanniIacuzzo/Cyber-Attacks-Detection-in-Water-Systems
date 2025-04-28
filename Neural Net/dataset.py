import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_and_preprocess_data(training_dataset_1, training_dataset_2, test_dataset):
    # Unisci i dataset di training
    training_data = pd.concat([training_dataset_1, training_dataset_2], ignore_index=True)

    # Separazione delle features e del target
    X = training_data.drop(columns=['DATETIME', 'ATT_FLAG'])
    y = training_data['ATT_FLAG']

    # Bilanciamento dei dati con SMOTE
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    # Divisione tra train e test
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

    # Conversione in tensori
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Creazione dei DataLoader per il training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, X_train, y_train, X_test, y_test
