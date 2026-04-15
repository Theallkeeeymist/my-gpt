import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.k_proj = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.q_proj = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        B,T,C = embedded.shape

        K = self.k_proj(embedded)
        Q = self.q_proj(embedded)
        V = self.v_proj(embedded)

        d_k = Q.shape[-1]
        scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)

        # Create lower triangular matrix of 1s
        mask = torch.tril(torch.ones(T, T, device=embedded.device))
        # Fill where mask is 0 with -infinity
        scores = scores.masked_fill(mask == 0, float('-inf'))

        Attention = torch.softmax(scores, dim=-1)

        Attention = Attention@V


        return torch.round(Attention, decimals=4)