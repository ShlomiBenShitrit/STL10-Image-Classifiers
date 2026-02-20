import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import config

class MobileNetTransfer(nn.Module):
    def __init__(self, fine_tune=False, num_classes=10):
        # The super() call initializes the internal engine of nn.Module.
        # It sets up key PyTorch mechanisms like parameter registration 
        # and GPU support so this class can function as a neural network.
        super(MobileNetTransfer, self).__init__()
        
        # Loading the pre-trained MobileNetV2 architecture with ImageNet weights.
        # This model is already a child of nn.Module.
        print(f"[Info] Loading MobileNetV2 (Fine-tune mode: {fine_tune})...")
        weights = MobileNet_V2_Weights.DEFAULT
        self.model = mobilenet_v2(weights=weights)

        # Accessing the uniform dropout rate from config.py
        dropout_p = config.CURRENT_PARAMS.get('dropout', 0.3)

        # Implementation for Task 4 (Fixed): Freezing the weights.
        # If fine_tune is False, we set requires_grad to False for all existing parameters
        # so they won't be updated during the training process.
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Accessing the specific number of features (1280) entering the classifier.
        # model.last_channel is a specific attribute of MobileNetV2 that holds this value.
        num_ftrs = self.model.last_channel 
        
        # Overwriting the original classifier with our specific task requirements.
        # By assigning a new nn.Sequential to self.model.classifier, we replace
        # the original 1000-class output layer with our custom MLP.
        # Requirement: Feature extractor followed by two FC layers and a classification layer.
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),              # Standard dropout from the original model
            
            # First additional Fully Connected (FC) layer
            nn.Linear(num_ftrs, 512),       # num_ftrs (1280) is the input size to this layer
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            
            # Second additional Fully Connected (FC) layer
            nn.Linear(512, 256),            # Takes 512 features and reduces to 256
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            
            # Final classification layer mapping to our 10 target classes
            nn.Linear(256, num_classes)     # Output vector of size 10
        )

    def forward(self, x):
        # This function defines the data flow (Forward Pass).
        # x represents the input image tensor.
        # Passing x through self.model(x) triggers the sequence: 
        # Feature Extractor -> Our custom Classifier.
        return self.model(x)