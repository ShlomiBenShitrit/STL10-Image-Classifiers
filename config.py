import torch

# --- 1. Global Settings ---
RANDOM_SEED = 42
MODEL_TYPE = 'cnn' # Options: 'logistic', 'fc', 'cnn', 'mobilenet_fixed', 'mobilenet_learned'
DATA_ROOT = './data'
VALIDATION_SPLIT = 0.2
IMAGE_SIZE = 64

# --- 2. Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. Model-Specific Hyperparameters ---
MODEL_CONFIGS = {
    'logistic': {
        'batch_size': 256,
        'lr': 1e-2,
        'epochs': 10,
        'weight_decay': 1e-5
    },
    'fc': {
        'batch_size': 64,
        'lr': 1e-3,
        'epochs': 20,
        'hidden_dim': 512,
        'dropout': 0.3,
        'weight_decay': 1e-4
    },
    'cnn': {
        'batch_size': 32,
        'lr': 1e-4,
        'epochs': 1,
        'dropout': 0.5,
        'weight_decay': 1e-4
    },
    'mobilenet_fixed': {
        'batch_size': 64,
        'lr': 1e-3,
        'epochs': 15,
        'weight_decay': 1e-5,
        'dropout': 0.3
        
    },
    'mobilenet_learned': {
        'batch_size': 32,
        'lr': 1e-5,
        'epochs': 50,
        'weight_decay': 1e-4,
        'dropout': 0
    }
}

CURRENT_PARAMS = MODEL_CONFIGS[MODEL_TYPE]