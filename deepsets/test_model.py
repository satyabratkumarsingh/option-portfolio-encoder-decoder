import torch
from deepsets_encoder import DeepSetEncoder
from deepsets_decoder import DeepSetDecoder
import numpy as np
from deepsets_learning import input_dim, latent_dim, hidden_dim, batch_size, DEVICE

option_array = np.random.randint(low=90, high=95, size=(32, 2))
print(option_array)
option_tensor = torch.tensor(option_array).float()

#payoff  = max(S(T) - K, 0)
cashflows_array = np.maximum(option_array[:,0] - option_array[:,1], 0)
cashflow_tensor = torch.tensor(cashflows_array).float()
print(cashflow_tensor)

encoder = DeepSetEncoder(input_dim=input_dim,
                         batch_size=batch_size,
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
    latent = encoder(option_tensor)

    predicted_cashflows = decoder(latent)
    print('=========Predicted Cashflow ========')
    print(predicted_cashflows.shape)
    print("Predicted Cashflows:", predicted_cashflows)
