
import torch.nn as nn


class DeepSetDiscriminator(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super(DeepSetDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Outputs a probability
        )
    
    def forward(self, x):
        # x: Cashflow (real or generated)
        return self.model(x)
    
