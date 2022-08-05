import torch
import torch.nn as nn
from transformers import AutoModel


class PercentileRegressor(nn.Module):
    def __init__(self, model_path, dropout=0.1, hidden_dim=768):
        super(PercentileRegressor, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        self.top = nn.Linear(hidden_dim, 1)

    def forward(self, ids, mask):
        x = self.model(ids, mask)[0]
        x = self.dropout(x)
        x = self.top(x[:, 0, :])
        return x


class SlidingWindowPercentileRegressor(nn.Module):
    def __init__(self, model_path, dropout=0.1):
        super(PercentileRegressor, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        self.top = nn.Linear(768 + 2, 1)

    def forward(self, ids, mask, window_index, num_windows):
        x = self.model(ids, mask)[0]
        x = self.dropout(x)
        x = torch.cat((x[:, 0, :], window_index, num_windows), 1)
        x = self.top(x)
        return x
