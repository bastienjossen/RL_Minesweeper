# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """CNN: 4×Conv → Flatten → 2×Dense → Linear output"""
    def __init__(self, in_channels, height, width, conv_units, dense_units, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,   conv_units, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_units,    conv_units, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(conv_units,    conv_units, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(conv_units,    conv_units, kernel_size=3, padding=1)
        # infer flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            conv_size = x.view(1, -1).size(1)
        self.fc1 = nn.Linear(conv_size, dense_units)
        self.fc2 = nn.Linear(dense_units, dense_units)
        self.out = nn.Linear(dense_units, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
