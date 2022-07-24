import torch
import torch.nn as nn
from transformers import AutoModel


class PercentileRegressor(nn.Module):
    def __init__(self, model_path, dropout=0.1):
        super(PercentileRegressor, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        self.top = nn.Linear(768, 1)

    def forward(self, ids, mask):
        x = self.model(ids, mask)[0]
        x = self.dropout(x)
        x = self.top(x[:, 0, :])
        return x
