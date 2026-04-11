import torch.nn as nn

class MoodModel(nn.Module):
    def __init__(self, input_size=7):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.out(x)

        return x
