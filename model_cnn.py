import torch
import torch.nn as nn
import config

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Safely extract dropout rate from the current configuration
        dropout_p = config.CURRENT_PARAMS.get('dropout', 0.5)

        # -----------------------------------------------------------------
        # Feature Extractorג
        # Architecture: 4 Blocks of [Conv -> BatchNorm -> ReLU -> MaxPool]
        # Input Image: 3 x 64 x 64 (defined in config.IMAGE_SIZE)
        # -----------------------------------------------------------------
        self.features = nn.Sequential(
            # Block 1: 
            # Input: (3, 64, 64) -> Conv -> (32, 64, 64) -> MaxPool -> Output: (32, 32, 32)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 
            # Input: (32, 32, 32) -> Conv -> (64, 32, 32) -> MaxPool -> Output: (64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3: 
            # Input: (64, 16, 16) -> Conv -> (128, 16, 16) -> MaxPool -> Output: (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 4: 
            # Input: (128, 8, 8) -> Conv -> (256, 8, 8) -> MaxPool -> Output: (256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # -----------------------------------------------------------------
        # Dynamic Calculation for Linear Layer Input
        # Since we have 4 MaxPool layers (dividing by 2 four times),
        # the spatial dimension reduces by a factor of 16 (2^4).
        # For 64x64 input: 64 / 16 = 4. Final spatial size is 4x4.
        # -----------------------------------------------------------------
        final_spatial_dim = config.IMAGE_SIZE // 16
        self.flattened_size = 256 * final_spatial_dim * final_spatial_dim

        # -----------------------------------------------------------------
        # Classifier (Fully Connected Layers)
        # -----------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 1. Extract Features
        x = self.features(x)
        
        # 2. Classify
        x = self.classifier(x)
        
        return x