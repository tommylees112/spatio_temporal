import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DummyDataset(Dataset):
    """
    Only while testing, ignore otherwise
    """

    def __init__(self, is_train=True):
        self.train = is_train
        self.train_size = 1000
        self.test_size = 1000
        self.input_size = 5
        self.output_size = 1

    def __len__(self):
        if self.train:
            return self.train_size
        return self.test_size

    def __getitem__(self, i):
        max_val = np.random.randint(5, 100)
        x = torch.from_numpy(np.arange(max_val - 5, max_val).reshape(-1, 1)).float()
        y = torch.from_numpy(np.array([max_val + 1]).reshape(-1, 1)).float()

        data = {}
        data["x_d"] = x
        data["y"] = y
        return data
