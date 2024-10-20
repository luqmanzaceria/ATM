import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super().__init__()
        self.condition_network = nn.Linear(condition_dim, input_dim * 2)

    def forward(self, x, condition):
        film_params = self.condition_network(condition)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        return (1 + gamma) * x + beta