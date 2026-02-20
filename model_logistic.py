import torch.nn as nn
import config

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim=3*config.IMAGE_SIZE*config.IMAGE_SIZE, num_classes=10):
        super(LogisticRegressionModel, self).__init__()
        # Mapping flattened pixels directly to class scores
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # Flatten the input from (Batch, C, H, W) to (Batch, D)
        x = x.view(x.size(0), -1)
        return self.linear(x)