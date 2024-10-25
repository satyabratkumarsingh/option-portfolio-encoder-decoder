import torch.nn as nn
import torch


class DeepSetEncoder(nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim, latent_dim):
        super(DeepSetEncoder, self).__init__()

        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.rho = nn.Sequential(
            nn.Linear(batch_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )
    def forward(self, x):
        phi_output = self.phi(x)
        pooled = torch.mean(phi_output, dim=1)
        latent = self.rho(pooled)
        return latent