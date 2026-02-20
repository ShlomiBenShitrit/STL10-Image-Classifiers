import torch
import torch.nn as nn
import get_data             # Accesses the get_data_loaders function
import config               # Accesses hyper-parameters and model type
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Import the model classes exactly as they are used in train.py
import model_logistic as logistic 
import model_fc as fc             
import model_cnn as cnn          
import mobilenet

def get_model():
    """
    Selector function based on config.MODEL_TYPE.
    Mirrors the exact logic used in train.py to ensure consistency.
    """
    model_type = config.MODEL_TYPE
    print(f"[Info] Initializing model for testing: {model_type}")

    if model_type == 'logistic':
        return logistic.LogisticRegressionModel(num_classes=10)
    elif model_type == 'fc':
        return fc.FullyConnectedNet(num_classes=10)
    elif model_type == 'cnn':
        return cnn.SimpleCNN(num_classes=10)
    elif model_type == 'mobilenet_fixed':
        return mobilenet.MobileNetTransfer(fine_tune=False, num_classes=10)
    elif model_type == 'mobilenet_learned':
        return mobilenet.MobileNetTransfer(fine_tune=True, num_classes=10)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def evaluate_model():
    """
    Main evaluation function. Loads the trained model and tests it on the STL-10 test set.
    """
    
    # 1. Set the output directory
    output_folder = 'plots_and_outputs'
    os.makedirs(output_folder, exist_ok=True)

    # 2. Set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Testing on device: {device}")

    # 3. Load the data using eval_transform
    _, _, test_loader = get_data.get_data_loaders()
    
    # 4. Extract class names accurately from the STL-10 dataset
    class_names = test_loader.dataset.subset.classes 

    # 5. Initialize the model architecture
    model = get_model()
    
    # 6. Load the saved weights from the outputs folder
    model_weight_filename = f'best_model_{config.MODEL_TYPE}.pth'
    model_weight_path = os.path.join(output_folder, model_weight_filename)

    try:
        # Added map_location=device to ensure weights load correctly on CPU or GPU
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        print(f"[Info] Successfully loaded weights from '{model_weight_path}'.")
    except FileNotFoundError:
        print(f"[Error] Weight file '{model_weight_path}' not found. Please train {config.MODEL_TYPE} first.")
        return

    model.to(device)
    
    # 7. Set model to evaluation mode
    model.eval()

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    # 8. Disable gradient calculation for efficiency
    print(f"[Info] Starting evaluation...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get the predicted class
            _, predicted = torch.max(outputs.data, 1)
            
            # Update counters
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 9. Calculate and display final Accuracy 
    final_acc = 100 * correct / total
    print(f"\n[Result] Final Test Accuracy ({config.MODEL_TYPE}): {final_acc:.2f}%")

    # 10. Generate detailed metrics and Confusion Matrix 
    print("\n[Info] Generating classification report...")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # --- Plotting and Saving the Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {config.MODEL_TYPE}\nAccuracy: {final_acc:.2f}%')
    
    # Save the plot into the outputs folder
    matrix_filename = f'confusion_matrix_{config.MODEL_TYPE}.png'
    matrix_path = os.path.join(output_folder, matrix_filename)
    
    plt.savefig(matrix_path)
    print(f"[Info] Confusion matrix saved as '{matrix_path}'.")
    plt.show()

if __name__ == "__main__":
    evaluate_model()