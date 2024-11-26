import torch.nn as nn
import torch


class DeepSetEncoder(nn.Module):
    def __init__(self, input_dim=2, batch_size=32, latent_dim=8, hidden_dim=16):
        super(DeepSetEncoder, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
 
        self.rho = nn.Sequential(
            nn.Linear(latent_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)    
        )

    def forward(self, x):
        phi_output = self.phi(x)
        pooled = torch.mean(phi_output, dim=1, keepdim=True)
        pooled_expanded = pooled.expand(-1, x.size(1), -1)
        combined = torch.cat([x, pooled_expanded], dim=2)
        encoded = self.rho(combined)
        return encoded