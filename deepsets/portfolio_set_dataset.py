import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import nn, optim
import numpy as np


# Padding function to ensure all batches have consistent dimensions
def do_padding(orig_tensor, batch_size, data_type):
    if orig_tensor.size(0) < batch_size:
        rows_to_add = batch_size - orig_tensor.size(0)
        if data_type == 'portfolio':
            dummy_rows = torch.zeros(rows_to_add, orig_tensor.size(1))
            new_tensor = torch.cat([orig_tensor, dummy_rows], dim=0)
        else:
            padding_tensor = torch.zeros(rows_to_add)
            new_tensor = torch.cat([orig_tensor, padding_tensor])
        return new_tensor
    else:
        return orig_tensor
    

class PortfolioNewDataset(Dataset):
    def __init__(self, portfolios, cashflows, set_size=5):
        self.portfolios = portfolios
        self.cashflows = cashflows
        self.set_size = set_size  

    def __len__(self):
        return len(self.portfolios) // self.set_size

    def __getitem__(self, idx):
        start_idx = idx * self.set_size
        end_idx = start_idx + self.set_size
        portfolios_set = self.portfolios[start_idx:end_idx]
        cashflows_set = self.cashflows[start_idx:end_idx]

        # Apply padding if the set is smaller than set_size
        portfolios_set = do_padding(portfolios_set, self.set_size, 'portfolio')
        cashflows_set = do_padding(cashflows_set, self.set_size, 'cashflow')

        # Add set dimension
        return portfolios_set.unsqueeze(0), cashflows_set.unsqueeze(0)