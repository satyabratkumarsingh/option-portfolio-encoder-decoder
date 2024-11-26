import numpy as np
import torch
from option_data import generate_option_prices
import random

# Generate a dataset of portfolio of options with size = n (we later want this to be of variable size)
def generate_portfolios(portfolios_number, portfolio_size):
    random_number = random.randint(100, 101)
    min_range = random_number
    max_range = random_number + 5

    portfolios_feature_list = []
    cashflows_list = []

    for i in range(portfolios_number):

        s_t_s, x_s, cashflow = generate_option_prices(portfolio_size, min_range, max_range)
        option_tensor = torch.tensor(np.column_stack((s_t_s, x_s))).float()

        portfolios_feature_list.append(option_tensor)
        cashflow_tensor = torch.tensor([float(cashflow)])
        cashflows_list.append(cashflow_tensor)

    portfolios_tensor = torch.stack(portfolios_feature_list, dim=0)
    cashflow_tensor = torch.stack(cashflows_list, dim=0)
    return portfolios_tensor, cashflow_tensor