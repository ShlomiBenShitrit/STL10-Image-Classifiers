import torch.nn as nn
import config

class FullyConnectedNet(nn.Module):
    def __init__(self, input_dim=3*config.IMAGE_SIZE*config.IMAGE_SIZE, num_classes=10):
        super(FullyConnectedNet, self).__init__()
        
        # We access the config values here safely. 
        # Note: These keys must exist in config.CURRENT_PARAMS when this model is instantiated.
        hidden = config.CURRENT_PARAMS.get('hidden_dim', 512)
        dropout_p = config.CURRENT_PARAMS.get('dropout', 0.5)
        
        self.network = nn.Sequential(
            # Hidden Layer 1
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            
            # Hidden Layer 2
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            
            # Hidden Layer 3
            nn.Linear(hidden // 2, hidden // 4),
            nn.BatchNorm1d(hidden // 4),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            
            # Output Layer
            nn.Linear(hidden // 4, num_classes)
        )

    def forward(self, x):
        # Flatten image: (Batch, 3, 64, 64) -> (Batch, 12288)
        x = x.view(x.size(0), -1)
        return self.network(x)