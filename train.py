import torch
import torch.optim as optim
from copy import deepcopy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss import CombinedLoss
import os

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_local(model, train_loader, config, val_loader=None, save_path=None):
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=config['reduce_lr_patience'], factor=config['reduce_lr_factor'], verbose=True)
    criterion = CombinedLoss(ce_weight=config['loss_weights']['ce'], softmax_weight=config['loss_weights']['softmax'])

    early_stopping = EarlyStopping(patience=config['early_stopping_patience'])

    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    best_val_loss = float('inf')

    for epoch in range(config['local_epochs']):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        if val_loader is not None:
            val_loss = evaluate_loss(model, val_loader, criterion)
            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss and save_path is not None:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

    return history

def evaluate_loss(model, val_loader, criterion):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss_total += loss.item()
    return loss_total / len(val_loader)

def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct / total

def average_models(models):
    global_model = deepcopy(models[0])
    for key in global_model.state_dict().keys():
        for i in range(1, len(models)):
            global_model.state_dict()[key] += models[i].state_dict()[key]
        global_model.state_dict()[key] = torch.div(global_model.state_dict()[key], len(models))
    return global_model
