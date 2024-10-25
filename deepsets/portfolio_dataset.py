from torch.utils.data import  Dataset


class PortfolioDataset(Dataset):
    def __init__(self, portfolios, cashflows):
        self.portfolios = portfolios  
        self.cashflows = cashflows

    def __len__(self):
        return len(self.portfolios)

    def __getitem__(self, idx):
        return self.portfolios[idx], self.cashflows[idx]