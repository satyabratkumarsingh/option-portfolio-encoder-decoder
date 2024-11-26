
import torch.nn as nn
import torch

class DeepSetDecoder(nn.Module):
    def __init__(self, latent_dim=8, hidden_dim=16):
        super(DeepSetDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.ReLU() 
        )
            
    def forward(self, x):
        pooled = torch.sum(x, dim=1)
        output = self.decoder(pooled)
        return output