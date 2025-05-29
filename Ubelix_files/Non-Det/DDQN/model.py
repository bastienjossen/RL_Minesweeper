# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingQNetwork(nn.Module):
    """Dueling DQN: separate value and advantage streams."""
    def __init__(self, in_channels, height, width, conv_units, dense_units, n_actions):
        super().__init__()
        # shared conv layers
        self.conv1 = nn.Conv2d(in_channels,    conv_units, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_units,     conv_units, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(conv_units,     conv_units, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(conv_units,     conv_units, kernel_size=3, padding=1)

        # infer conv output size without NumPy
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            conv_size = x.reshape(1, -1).size(1)

        # value stream
        self.value_fc  = nn.Linear(conv_size, dense_units)
        self.value_out = nn.Linear(dense_units, 1)
        # advantage stream
        self.adv_fc    = nn.Linear(conv_size, dense_units)
        self.adv_out   = nn.Linear(dense_units, n_actions)

    def forward(self, x):
        # x: [batch, in_channels, H, W]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # flatten
        x = x.reshape(x.size(0), -1)

        # value head
        v = F.relu(self.value_fc(x))
        v = self.value_out(v)
        # advantage head
        a = F.relu(self.adv_fc(x))
        a = self.adv_out(a)

        # combine into Q
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
