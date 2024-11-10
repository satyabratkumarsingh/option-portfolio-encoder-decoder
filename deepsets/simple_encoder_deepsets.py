import torch.nn as nn
import torch


class SimpleEncoderDeepset(nn.Module):
    def __init__(self, input_dim=2, latent_dim=8):
        super(SimpleEncoderDeepset, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU()
        )
 
        self.rho = nn.Sequential(
            nn.Linear(latent_dim + input_dim, 32),  # Project to 32 dimensions for rows
            nn.ReLU(),
            nn.Linear(32, 16)          # Project to 16 dimensions for columns
        )
        
    def forward(self, x):
        phi_output = self.phi(x)
        pooled = torch.mean(phi_output, dim=0, keepdim=True)
        pooled_expanded = pooled.expand(x.size(0), -1)
        combined = torch.cat([x, pooled_expanded], dim=1)
        rho = self.rho(combined)
        return rho