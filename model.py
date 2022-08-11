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


class RepresExtractor(nn.Module):
    def __init__(self, model_path, dropout=0.1, hidden_dim=768):
        super(RepresExtractor, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids, mask):
        x = self.model(ids, mask)[0]
        x = self.dropout(x)
        return torch.nn.functional.normalize(x[:, 0, :], p=2, dim=1)
