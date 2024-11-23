import numpy as np
import torch


# Parameters for lognormal diffusion
# S_T = S_0 * exp((mu - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
# Where:
# - S_T: Stock price at time T (future price)
# - S_0: Initial stock price (starting value)
# - mu: Drift term (expected return)
# - sigma: Volatility of the stock
# - T: Time to maturity (in years)
# - Z: Standard normal random variable (random noise, Z ~ N(0, 1))

#Let's assume 
MU = 0.05  # Drift (Let's assume stock gives 5% return)
SIGMA = 0.2  # Volatility (20% annualized)
T = 1  # Time to maturity in years (Fixed for now, will need to change)

# Generate a dataset of stock prices of Size n
def generate_option_prices(n, min_price_range, max_price_range):

   
    # Generate random initial stock prices (S_0) in the given range
    S_0_Prices = np.random.uniform(min_price_range, max_price_range, size=n)

    Z = np.random.normal(0, 1, n)

    S_T_Prices = S_0_Prices * np.exp((MU - 0.5 * SIGMA**2) * T + SIGMA * np.sqrt(T) * Z)  # Lognormal diffusion

    min_x_factor = 0.95  # Strike price 95% of S_0
    max_x_factor = 1.05  # Strike price 105% of S_0

    X_prices = S_0_Prices * np.random.uniform(min_x_factor, max_x_factor, size=n) # Strike Prices

    Cashflows = np.maximum(S_T_Prices - X_prices, 0)

    return S_T_Prices, X_prices, Cashflows

