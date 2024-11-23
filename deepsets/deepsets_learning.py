import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from deepsets_encoder import DeepSetEncoder
from deepsets_decoder import DeepSetDecoder
from torch.utils.data import TensorDataset, DataLoader
from portfolio_dataset import PortfolioDataset
from early_stopping import EarlyStopping
from option_data import generate_option_prices

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 2
hidden_dim = 16
latent_dim = 8
batch_size = 32
epochs = 1000

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

def main():
    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    encoder = DeepSetEncoder(input_dim=input_dim,
                            batch_size=batch_size,
                            latent_dim=latent_dim,
                            hidden_dim=hidden_dim).to(DEVICE)

    decoder = DeepSetDecoder(latent_dim=latent_dim, 
                            hidden_dim=hidden_dim).to(DEVICE)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                        lr=0.000001)

    # INPUT a tensor for call option with S(T), X 
    s_t_prices, x_prices, cashflows = generate_option_prices(100000, 99, 110)

    option_tensor = torch.tensor(np.column_stack((s_t_prices, x_prices))).float()

    cashflow_tensor = torch.tensor(cashflows).float()

    option_dataset = TensorDataset(option_tensor, cashflow_tensor)

    # Create dataset and data loader
    dataset = PortfolioDataset(option_tensor, cashflow_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    best_loss = float('inf')

    early_stopping = EarlyStopping(patience=21, min_delta=0.000001)


    for epoch in range(epochs):
        epoch_losses = []
        for portfolio_real, cashflows_real in data_loader:
            optimizer.zero_grad()
            portfolio_real = do_padding(portfolio_real, batch_size, 'portfolio')
            cashflows_real = do_padding(cashflows_real, batch_size, 'cashflow')
            latent_rep = encoder(portfolio_real)
            predicted_cashflows = decoder(latent_rep)
            loss = mse_loss(predicted_cashflows, cashflows_real)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}')
        early_stopping(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if early_stopping.early_stop:
            print(f'Early stopping triggered at epoch {epoch}')
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_model.pt')
            break

if __name__ == "__main__":
    main()
