import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.fc1=nn.Linear(784, 512)
        self.act1=nn.ReLU()
        self.drop=nn.Dropout(0.2)
        self.fc2=nn.Linear(512, 10)
        self.act2=nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        x = self.fc1(images)          #Input(784->512)
        x = self.act1(x)              #ReLU
        x = self.drop(x)              #Dropout(0.2)
        x = self.fc2(x)               #Linear(512->10)
        output = self.act2(x)         #Sigmoid

        return torch.round(output, decimals=4)