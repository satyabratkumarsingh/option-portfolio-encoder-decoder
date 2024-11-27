import torch
from deepsets_encoder import DeepSetEncoder
from deepsets_decoder import DeepSetDecoder
import numpy as np
from deepsets_learning import input_dim, latent_dim, hidden_dim, batch_size, DEVICE
from portfolio_data import generate_portfolios

def main():
    print('==========TESTING MOIDDDL=====')
    feature_tensor, cashflow_tensor = generate_portfolios(32, 10)

    encoder = DeepSetEncoder(input_dim=input_dim,
                            batch_size=32,
                            latent_dim=latent_dim,
                            hidden_dim=hidden_dim).to(DEVICE)

    decoder = DeepSetDecoder(latent_dim=latent_dim, 
                            hidden_dim=hidden_dim).to(DEVICE)


    # Load the saved model checkpoint
    checkpoint = torch.load('best_model.pt')

    # Load the saved state_dict into the encoder, decoder, and optimizer
    encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
    decoder.load_state_dict(checkpoint['decoder_state_dict'], strict=False)

    # Set the models to evaluation mode
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        latent = encoder(feature_tensor)

        predicted_cashflows = decoder(latent)

        print('========= Actual Cashflow ========')
        print(cashflow_tensor)
        
        print('=========Predicted Cashflow ========')
        print(predicted_cashflows.shape)
        print("Predicted Cashflows:", predicted_cashflows)

        # 1. Mean Absolute Error (MAE)
        mae = torch.mean(torch.abs(predicted_cashflows - cashflow_tensor))
        print(f"Mean Absolute Error (MAE): {mae.item():.4f}")

        # 2. Mean Squared Error (MSE)
        mse = torch.mean((predicted_cashflows - cashflow_tensor) ** 2)
        print(f"Mean Squared Error (MSE): {mse.item():.4f}")

        # 3. Root Mean Squared Error (RMSE)
        rmse = torch.sqrt(mse)
        print(f"Root Mean Squared Error (RMSE): {rmse.item():.4f}")

        # 4. R-Squared
        ss_total = torch.sum((cashflow_tensor - torch.mean(cashflow_tensor)) ** 2)
        ss_residual = torch.sum((cashflow_tensor - predicted_cashflows) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        print(f"R-Squared: {r2.item():.4f}")

        epsilon = 1e-8  # Small constant to prevent division by zero
        mape = torch.mean(torch.abs((cashflow_tensor - predicted_cashflows) / (cashflow_tensor + epsilon))) * 100
        print(f"MAPE: {mape.item():.4f}")


if __name__ == "__main__":
    main()
