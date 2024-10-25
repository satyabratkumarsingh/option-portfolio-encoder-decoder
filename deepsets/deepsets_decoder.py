
import torch.nn as nn
import torch

class DeepSetDecoder(nn.Module):

    def __init__(self, batch_size, latent_dim, hidden_dim, output_dim):
        super(DeepSetDecoder, self).__init__()

        self.latent_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Element generator network
        self.element_generator = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, batch_size)
        )

    def forward(self, x):
        hidden = self.latent_to_hidden(x)
        combined = torch.cat([hidden, x], dim=0)
        output_set = self.element_generator(combined)
        return output_set
    