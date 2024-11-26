from comet_ml import start
from comet_ml.integration.pytorch import log_model
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
from portfolio_data import generate_portfolios


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
experiment = start(api_key="iatWnXT4JyBtDQhn7OfgISQoF", project_name="option-portfolio-encoder-decoder", workspace="satyabratkumarsingh")

input_dim = 2
hidden_dim = 16
latent_dim = 8
batch_size = 32
epochs = 1000

def do_padding(orig_portfolio, batch_size, data_type):
    if orig_portfolio.size(0) < batch_size:
        if data_type == 'portfolio':
            rows_to_add = batch_size - orig_portfolio.size(0)
            zero_row = torch.zeros((rows_to_add, orig_portfolio.size(1),
                                    orig_portfolio.size(2))) 
            new_portfolio = torch.cat([orig_portfolio, zero_row], dim=0)
            return new_portfolio
        else:
            rows_to_add = batch_size - orig_portfolio.size(0)
            zero_row = torch.zeros((rows_to_add, orig_portfolio.size(1)))
            new_portfolio = torch.cat([orig_portfolio, zero_row])
            return new_portfolio
    else:
        return orig_portfolio

def main():
    # Loss functions
    mse_loss = nn.MSELoss()

    encoder = DeepSetEncoder(input_dim=input_dim,
                            batch_size=batch_size,
                            latent_dim=latent_dim,
                            hidden_dim=hidden_dim).to(DEVICE)

    decoder = DeepSetDecoder(latent_dim=latent_dim, 
                            hidden_dim=hidden_dim).to(DEVICE)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                        lr=0.0001)

    hyper_params = {
            "learning_rate": 0.0001,
            "steps": 1000,
            "batch_size": 32,
        }
    experiment.log_parameters(hyper_params)

   
    feature_tensor, cashflow_tensor = generate_portfolios(10000, 100)

    # Create dataset and data loader
    dataset = PortfolioDataset(feature_tensor, cashflow_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    best_loss = float('inf')

    early_stopping = EarlyStopping(patience=21, min_delta=0.00001)


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
            print(f'Epoch [{epoch}/{epochs}], Loss: {avg_loss:.6f}')
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
            #log_model(experiment, model=encoder.state_dict(), model_name="Encoder")
            #log_model(experiment, model=decoder.state_dict(), model_name="Decoder")
            #log_model(experiment, model=optimizer.state_dict(), model_name="Optimizer")
            break

if __name__ == "__main__":
    main()
