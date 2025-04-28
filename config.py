# config.py

CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'num_clients': 5,
    'local_epochs': 10,
    'global_rounds': 50,
    'batch_size': 64,
    'lr': 0.001,
    'input_size': 43,  
    'hidden_sizes': [128, 64, 32],
    'dropout': 0.3,
    'output_size': 2,
    'early_stopping_patience': 5,
    'reduce_lr_patience': 3,
    'reduce_lr_factor': 0.5,
    'loss_weights': {'ce': 1.0, 'softmax': 0.1}
}
