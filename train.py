import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt

# Custom module imports
import config
import get_data
import model_logistic as logistic 
import model_fc as fc             
import model_cnn as cnn          
import mobilenet

def set_seed(seed):
    """
    Ensures reproducibility. 
    We set multiple seeds because PyTorch uses different 'engines' for CPU and GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # torch.manual_seed handles CPU-based random operations
    torch.manual_seed(seed)
    
    # torch.cuda.manual_seed and manual_seed_all handle GPU-based random operations 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Forcing cuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[Info] Seed set to {seed}")

def get_model():
    """Selector function based on config.MODEL_TYPE"""
    MODEL_TYPE = config.MODEL_TYPE
    print(f"[Info] Initializing model: {MODEL_TYPE}")

    if MODEL_TYPE == 'logistic':
        return logistic.LogisticRegressionModel(num_classes=10)
    elif MODEL_TYPE == 'fc':
        return fc.FullyConnectedNet(num_classes=10)
    elif MODEL_TYPE == 'cnn':
        return cnn.SimpleCNN(num_classes=10)
    elif MODEL_TYPE == 'mobilenet_fixed':
        return mobilenet.MobileNetTransfer(fine_tune=False, num_classes=10)
    elif MODEL_TYPE == 'mobilenet_learned':
        return mobilenet.MobileNetTransfer(fine_tune=True, num_classes=10)
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train() # Set model to training mode
    
    running_loss = 0.0
    num_correct_predictions = 0
    num_total_samples = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_size
        _, predicted_indices = outputs.max(1)
        
        num_total_samples += batch_size
        num_correct_predictions += predicted_indices.eq(labels).sum().item()
        
    epoch_loss = running_loss / num_total_samples
    epoch_accuracy = 100. * num_correct_predictions / num_total_samples
    
    return epoch_loss, epoch_accuracy

def validate(model, loader, criterion, device):
    model.eval() # Set model to evaluation mode
    
    running_loss = 0.0
    num_correct_predictions = 0
    num_total_samples = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * batch_size
            _, predicted_indices = outputs.max(1)
            num_total_samples += batch_size
            num_correct_predictions += predicted_indices.eq(labels).sum().item()
            
    val_loss = running_loss / num_total_samples
    val_accuracy = 100. * num_correct_predictions / num_total_samples
    
    return val_loss, val_accuracy

def plot_training_curves(history):
    """
    Plots the training and validation loss and accuracy curves and saves them to 'plots_and_outputs'.
    """
    # Create the output directory if it doesn't exist
    output_folder = 'plots_and_outputs'
    os.makedirs(output_folder, exist_ok=True)

    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy', marker='o')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save using os.path.join for safety with spaces in directory names
    plot_path = os.path.join(output_folder, f'training_curves_{config.MODEL_TYPE}.png')
    plt.savefig(plot_path)
    print(f"[Info] Training curves saved to: {plot_path}")
    plt.close()

def main():
    set_seed(config.RANDOM_SEED)
    device = config.DEVICE

    # Create the output directory at the start
    output_folder = 'plots_and_outputs'
    os.makedirs(output_folder, exist_ok=True)

    print("[Info] Loading STL-10 dataset...")
    train_loader, val_loader, _ = get_data.get_data_loaders()
    
    model = get_model().to(device)
    
    params = config.CURRENT_PARAMS
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=params['lr'], 
                           weight_decay=params.get('weight_decay', 0))
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(params['epochs']):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        
        print(f"Epoch [{epoch+1}/{params['epochs']}] "
              f"Train Loss: {t_loss:.4f}, Acc: {t_acc:.2f}% | "
              f"Val Loss: {v_loss:.4f}, Acc: {v_acc:.2f}%")
        
        # Save best model to the dynamic outputs folder
        if v_acc > best_acc:
            best_acc = v_acc
            checkpoint_filename = f'best_model_{config.MODEL_TYPE}.pth'
            checkpoint_path = os.path.join(output_folder, checkpoint_filename)
            torch.save(model.state_dict(), checkpoint_path)
            print(f" --> Best model saved to {checkpoint_path} with {best_acc:.2f}% accuracy")

    print("[Info] Training complete. Generating performance plots...")
    plot_training_curves(history)

if __name__ == "__main__":
    main()