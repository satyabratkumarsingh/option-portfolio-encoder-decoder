import torch.nn as nn
import torch


class SimpleEncoder(nn.Module):
    def __init__(self, input_dim=2, latent_dim=8):
        super(SimpleEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        encoder =  self.encoder(x)
        print('++++ Encoder +++++++')
        print(encoder.shape)
        return encoder