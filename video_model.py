import torch
import torch.nn as nn
from torchvision import models

class VideoDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()

        cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(cnn.children())[:-1])

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        batch, seq, c, h, w = x.shape
        x = x.view(batch * seq, c, h, w)

        with torch.no_grad():
            features = self.cnn(x).squeeze()

        features = features.view(batch, seq, -1)
        lstm_out, _ = self.lstm(features)

        out = self.fc(lstm_out[:, -1, :])
        return out
