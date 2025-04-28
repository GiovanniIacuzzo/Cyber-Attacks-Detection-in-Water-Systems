import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from model import NeuralNet

def train_model(model, train_loader, criterion, optimizer, scheduler, epochs=100, patience=10):
    model.train()
    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step(running_loss)  # Usa la loss per il scheduler

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print("Early stopping triggered")
            break

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)

def evaluate_and_print_metrics(model, test_loader, y_true):
    y_pred, y_true_pred = evaluate_model(model, test_loader)
    accuracy = accuracy_score(y_true, y_true_pred)
    loss = log_loss(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Visualizzazione della confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Attack', 'Attack'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    print(f"Accuracy: {accuracy}")
    print(f"Log Loss: {loss}")
    print(f"Classification Report:\n{class_report}")
    
    return accuracy, loss

def train_local_models(X_train, y_train, train_loader, client_count=5):
    local_accuracies = []
    local_losses = []

    clients_data = np.array_split(X_train, client_count)
    clients_labels = np.array_split(y_train, client_count)

    for i, (client_X, client_y) in enumerate(zip(clients_data, clients_labels)):
        # Creazione del modello per il client
        local_model = NeuralNet(input_dim=X_train.shape[1])
        local_optimizer = optim.AdamW(local_model.parameters(), lr=0.001, weight_decay=1e-5)  # AdamW con weight decay
        local_criterion = torch.nn.CrossEntropyLoss()

        # Creazione dei DataLoader per ciascun client
        client_X_tensor = torch.tensor(client_X.values, dtype=torch.float32)
        client_y_tensor = torch.tensor(client_y.values, dtype=torch.long)
        client_dataset = torch.utils.data.TensorDataset(client_X_tensor, client_y_tensor)
        client_loader = torch.utils.data.DataLoader(client_dataset, batch_size=32, shuffle=True)

        # Learning rate scheduler con ReduceLROnPlateau
        client_scheduler = optim.lr_scheduler.ReduceLROnPlateau(local_optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        # Allenamento del modello del client
        train_model(local_model, client_loader, local_criterion, local_optimizer, client_scheduler, epochs=100)

        # Valutazione del modello del client
        accuracy, loss = evaluate_and_print_metrics(local_model, client_loader, client_y)
        local_accuracies.append(accuracy)
        local_losses.append(loss)

    return local_accuracies, local_losses
