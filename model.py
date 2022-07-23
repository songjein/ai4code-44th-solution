import torch
import torch.nn as nn
from transformers import AutoModel


class PercentileRegressor(nn.Module):
    def __init__(self, model_path, dropout=0.1):
        super(PercentileRegressor, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        self.top = nn.Linear(768 + 1, 1)

    def forward(self, ids, mask, md_ratio):
        x = self.model(ids, mask)[0]
        x = self.dropout(x)
        x = torch.cat((x[:, 0, :], md_ratio), 1)
        x = self.top(x)
        return x
