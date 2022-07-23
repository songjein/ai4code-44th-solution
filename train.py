import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from dataset import MarkdownDataset
from metrics import kendall_tau
from model import PercentileRegressor

parser = argparse.ArgumentParser(description="Process some arguments")
parser.add_argument("--model_name_or_path", type=str, default="microsoft/codebert-base")
parser.add_argument("--train_md_path", type=str, default="./data/train_md.csv")
parser.add_argument("--train_features_path", type=str, default="./data/train_fts.json")
parser.add_argument("--valid_md_path", type=str, default="./data/valid_md.csv")
parser.add_argument("--valid_features_path", type=str, default="./data/valid_fts.json")
parser.add_argument("--valid_path", type=str, default="./data/valid.csv")

parser.add_argument("--md_max_len", type=int, default=64)
parser.add_argument("--total_max_len", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--accumulation_steps", type=int, default=4)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--n_workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    args = parser.parse_args()

    output_dir = f"outputs_{args.seed}"
    os.makedirs(f"./{output_dir}", exist_ok=True)
    data_dir = Path("./data/")

    seed_everything(args.seed)

    df_train_md = (
        pd.read_csv(args.train_md_path)
        .drop("parent_id", axis=1)
        .dropna()
        .reset_index(drop=True)
    )
    train_fts = json.load(open(args.train_features_path))
    df_valid_md = (
        pd.read_csv(args.valid_md_path)
        .drop("parent_id", axis=1)
        .dropna()
        .reset_index(drop=True)
    )
    valid_fts = json.load(open(args.valid_features_path))
    valid_df = pd.read_csv(args.valid_path)

    order_df = pd.read_csv("./data/train_orders.csv").set_index("id")
    df_orders = pd.read_csv(
        data_dir / "train_orders.csv",
        index_col="id",
        squeeze=True,
    ).str.split()

    train_ds = MarkdownDataset(
        df_train_md,
        model_name_or_path=args.model_name_or_path,
        md_max_len=args.md_max_len,
        total_max_len=args.total_max_len,
        fts=train_fts,
    )
    valid_ds = MarkdownDataset(
        df_valid_md,
        model_name_or_path=args.model_name_or_path,
        md_max_len=args.md_max_len,
        total_max_len=args.total_max_len,
        fts=valid_fts,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        pin_memory=False,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=False,
        drop_last=False,
    )

    def read_data(data):
        return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()

    def validate(model, valid_loader):
        model.eval()

        tbar = tqdm(valid_loader, file=sys.stdout)

        preds = []
        labels = []

        with torch.no_grad():
            for idx, data in enumerate(tbar):
                inputs, target = read_data(data)

                with torch.cuda.amp.autocast():
                    pred = model(*inputs)

                preds.append(pred.detach().cpu().numpy().ravel())
                labels.append(target.detach().cpu().numpy().ravel())

        return np.concatenate(labels), np.concatenate(preds)

    def train(model, train_loader, valid_loader, epochs):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        num_train_optimization_steps = int(
            args.epochs * len(train_loader) / args.accumulation_steps
        )
        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.05 * num_train_optimization_steps,
            num_training_steps=num_train_optimization_steps,
        )

        criterion = torch.nn.L1Loss()
        scaler = torch.cuda.amp.GradScaler()

        for e in range(epochs):
            model.train()
            tbar = tqdm(train_loader, file=sys.stdout)
            loss_list = []
            preds = []
            labels = []

            for idx, data in enumerate(tbar):
                inputs, target = read_data(data)

                with torch.cuda.amp.autocast():
                    pred = model(*inputs)
                    loss = criterion(pred, target)
                scaler.scale(loss).backward()

                if idx % args.accumulation_steps == 0 or idx == len(tbar) - 1:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                loss_list.append(loss.detach().cpu().item())
                preds.append(pred.detach().cpu().numpy().ravel())
                labels.append(target.detach().cpu().numpy().ravel())

                avg_loss = np.round(np.mean(loss_list), 4)

                tbar.set_description(
                    f"Epoch {e + 1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}"
                )

            y_val, y_pred = validate(model, valid_loader)
            valid_df["pred"] = valid_df.groupby(["id", "cell_type"])["rank"].rank(
                pct=True
            )
            valid_df.loc[valid_df["cell_type"] == "markdown", "pred"] = y_pred
            y_dummy = valid_df.sort_values("pred").groupby("id")["cell_id"].apply(list)
            print("Preds score", kendall_tau(df_orders.loc[y_dummy.index], y_dummy))
            torch.save(model.state_dict(), f"./{output_dir}/model_{e}.bin")

        return model, y_pred

    model = PercentileRegressor(args.model_name_or_path)
    model = model.cuda()
    model, y_pred = train(model, train_loader, valid_loader, epochs=args.epochs)
