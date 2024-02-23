import torch.nn as nn
import torch

class DigitsModel(nn.Module):
    def __init__(self) -> None:
        super(DigitsModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels= 60, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride= 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=60, out_channels= 32, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size= 2, stride= 2)
        )
        self.flatten = nn.Flatten()
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features= 64),
            nn.MaxPool2d(kernel_size= 2, stride= 2)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(in_features= 8 * 8 * 64, out_features= 1024),
            nn.LeakyReLU(negative_slope= 0.01),
            nn.BatchNorm1d(num_features= 1024),
            nn.Dropout(p= 0.2)
        )
        self.fclayer= nn.Linear(in_features= 1024, out_features= 10)
    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        z = self.flatten(y)
        t = self.layer4(z)
        t = self.fclayer(t)
        return t

    
