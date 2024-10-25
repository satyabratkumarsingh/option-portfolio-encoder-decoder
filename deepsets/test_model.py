import torch
from deepsets_encoder import DeepSetEncoder
from deepsets_decoder import DeepSetDecoder
import numpy as np


# INPUT a tensor for call option with S(T), X 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

option_array = np.random.randint(low=90, high=110, size=(64, 2))
print(option_array)
option_tensor = torch.tensor(option_array).float()

#payoff  = max(S(T) - K, 0)
cashflows_array = np.maximum(option_array[:,0] - option_array[:,1], 0)
cashflow_tensor = torch.tensor(cashflows_array).float()
print(cashflow_tensor)


# Create models
input_dim = 2  # E.g., input features for each option (like strike, maturity)
hidden_dim = 64
latent_dim = 32
output_dim = 1  # Cashflow
batch_size = 64

# Reinitialize the models (ensure the architecture matches the saved model)
encoder = DeepSetEncoder(batch_size, input_dim, hidden_dim, latent_dim).to(device)
decoder = DeepSetDecoder(batch_size, latent_dim, hidden_dim, output_dim).to(device)

# Load the saved model checkpoint
checkpoint = torch.load('best_model.pt')

# Load the saved state_dict into the encoder, decoder, and optimizer
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

# Set the models to evaluation mode
encoder.eval()
decoder.eval()

# Optionally, you can load the optimizer state if needed for further training
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Test the model
with torch.no_grad():  # No need to track gradients during evaluation
    latent = encoder(option_tensor)  # Encode the test data
    predicted_cashflows = decoder(latent)  # Decode to get predicted cashflows

    # Optionally, compare with actual cashflows if you have them
    print("Predicted Cashflows:", predicted_cashflows)

