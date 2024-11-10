import torch.nn as nn
import torch


class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=8):
        super(SimpleDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()  # ReLU ensures non-negative output like max function
        )
            
    def forward(self, x):
        t= self.decoder(x).squeeze(-1)  # Remove last dimension to match target
        return t