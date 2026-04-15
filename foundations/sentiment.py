import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.emb1 = nn.Embedding(vocabulary_size, 16)
        self.fc1 = nn.Linear(16, 1)
        self.act1 = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer
        embedded = self.emb1(x)

        mean_embedded = torch.mean(embedded, dim=1)

        layer_1 = self.fc1(mean_embedded)

        output = self.act1(layer_1)

        # Return a B, 1 tensor and round to 4 decimal places

        return torch.round(output, decimals=4)