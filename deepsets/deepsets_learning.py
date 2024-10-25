import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from deepsets_encoder import DeepSetEncoder
from deepsets_decoder import DeepSetDecoder
from deepsets_discriminator import DeepSetDiscriminator
from torch.utils.data import TensorDataset, DataLoader
from portfolio_dataset import PortfolioDataset
from early_stopping import EarlyStopping


def do_padding(orig_portfolio, batch_size, data_type):
    if orig_portfolio.size(0) < batch_size:
        if data_type == 'portfolio':
            rows_to_add = batch_size - orig_portfolio.size(0)
            dummy_rows = torch.zeros(rows_to_add, orig_portfolio.size(1))
            new_portfolio = torch.cat([orig_portfolio, dummy_rows], dim=0)
            return new_portfolio
        else:
            rows_to_add = batch_size - orig_portfolio.size(0)
            padding_tensor = torch.zeros(rows_to_add)
            new_portfolio = torch.cat([orig_portfolio, padding_tensor])
            return new_portfolio
    else:
        return orig_portfolio


batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss functions
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()

# Create models
input_dim = 2  # E.g., input features for each option (like strike, maturity)
hidden_dim = 64
latent_dim = 32
output_dim = 32  # Cashflow

encoder = DeepSetEncoder(batch_size, input_dim, hidden_dim, latent_dim).to(device)
decoder = DeepSetDecoder(batch_size, latent_dim, hidden_dim, output_dim).to(device)



optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001)

# INPUT a tensor for call option with S(T), X 

option_array = np.random.randint(low=101, high=105, size=(200000, 2))
option_tensor = torch.tensor(option_array).float()

#payoff  = max(S(T) - K, 0)
cashflows_array = np.maximum(option_array[:,0] - option_array[:,1], 0)
cashflow_tensor = torch.tensor(cashflows_array).float()



epochs = 1000
option_dataset = TensorDataset(option_tensor, cashflow_tensor)


# Create dataset and data loader
dataset = PortfolioDataset(option_tensor, cashflow_tensor)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

losses = []
best_loss = float('inf')

early_stopping = EarlyStopping(patience=5, min_delta=0.001)


for epoch in range(epochs):
    epoch_losses = []
    for portfolio_real, cashflows_real in data_loader:

        portfolio_real = do_padding(portfolio_real, batch_size, 'portfolio')
        cashflows_real = do_padding(cashflows_real, batch_size, 'cashflow')
    
        latent_rep = encoder(portfolio_real)
        predicted_cashflows = decoder(latent_rep)
   
        loss = mse_loss(predicted_cashflows, cashflows_real)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        epoch_losses.append(loss.item())
    avg_loss = np.mean(epoch_losses)
    losses.append(avg_loss)
    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}')
    early_stopping(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, 'best_model.pt')
        
    if early_stopping.early_stop:
        print(f'Early stopping triggered at epoch {epoch}')
        # Load best model
        checkpoint = torch.load('best_model.pt')
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        break

