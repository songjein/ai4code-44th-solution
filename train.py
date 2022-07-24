import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.utils import clip_grad_norm_
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from dataset import PairwiseDataset, PointwiseDataset
from metrics import kendall_tau
from model import PercentileRegressor

parser = argparse.ArgumentParser(description="Process some arguments")
parser.add_argument("--model_name_or_path", type=str, default="microsoft/codebert-base")
parser.add_argument("--train_md_path", type=str, default="./data/train_md.csv")
parser.add_argument("--train_features_path", type=str, default="./data/train_ctx.json")
parser.add_argument("--valid_md_path", type=str, default="./data/valid_md.csv")
parser.add_argument("--valid_features_path", type=str, default="./data/valid_ctx.json")
parser.add_argument("--train_path", type=str, default="./data/train.csv")
parser.add_argument("--valid_path", type=str, default="./data/valid.csv")

parser.add_argument("--md_max_len", type=int, default=64)
parser.add_argument("--total_max_len", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--accumulation_steps", type=int, default=4)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--n_workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train_mode", type=str, default="pointwise")
parser.add_argument("--memo", type=str, default="")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_pairs_with_label(df, mode="train", pos_neg_times=10):
    samples = []
    for id, df_sub in tqdm(df.groupby("id")):
        df_sub_md = df_sub[df_sub["cell_type"] == "markdown"]
        df_sub_code = df_sub[df_sub["cell_type"] == "code"]
        df_sub_code_rank = df_sub_code["rank"].values
        df_sub_code_cell_id = df_sub_code["cell_id"].values
        pos_samples = []
        neg_samples = []
        for md_cell_id, md_rank in df_sub_md[["cell_id", "rank"]].values:
            labels = np.array(
                [((md_rank + 1) == code_rank) for code_rank in df_sub_code_rank]
            ).astype("int")
            for code_cell_id, label in zip(df_sub_code_cell_id, labels):
                if mode == "test":
                    pos_samples.append([md_cell_id, code_cell_id, label])
                elif label == 1:
                    pos_samples.append([md_cell_id, code_cell_id, label])
                elif label == 0:
                    neg_samples.append([md_cell_id, code_cell_id, label])
        random.shuffle(neg_samples)
        _samples = pos_samples + neg_samples[: len(pos_samples) * pos_neg_times]
        random.shuffle(_samples)
        samples += _samples
    return samples


if __name__ == "__main__":
    args = parser.parse_args()

    assert args.train_mode in ["pointwise", "pairwise"]

    if args.train_mode == "pairwise":
        args.total_max_len = 128
        args.md_max_len = 64

    output_dir = f"outputs_{args.train_mode}_{args.memo}_{args.seed}"
    os.makedirs(f"./{output_dir}", exist_ok=True)
    data_dir = Path("./data/")

    seed_everything(args.seed)

    df_train_md = (
        pd.read_csv(args.train_md_path)
        .drop("parent_id", axis=1)
        .dropna()
        .reset_index(drop=True)
    )
    train_ctx = json.load(open(args.train_features_path))
    train_df = pd.read_csv(args.train_path)

    df_valid_md = (
        pd.read_csv(args.valid_md_path)
        .drop("parent_id", axis=1)
        .dropna()
        .reset_index(drop=True)
    )
    valid_ctx = json.load(open(args.valid_features_path))
    valid_df = pd.read_csv(args.valid_path)

    order_df = pd.read_csv("./data/train_orders.csv").set_index("id")
    df_orders = pd.read_csv(
        data_dir / "train_orders.csv",
        index_col="id",
        squeeze=True,
    ).str.split()

    if args.train_mode == "pointwise":
        train_ds = PointwiseDataset(
            df_train_md,
            model_name_or_path=args.model_name_or_path,
            md_max_len=args.md_max_len,
            total_max_len=args.total_max_len,
            ctx=train_ctx,
        )
        valid_ds = PointwiseDataset(
            df_valid_md,
            model_name_or_path=args.model_name_or_path,
            md_max_len=args.md_max_len,
            total_max_len=args.total_max_len,
            ctx=valid_ctx,
        )
    else:
        train_samples = generate_pairs_with_label(train_df, mode="train")
        train_ds = PairwiseDataset(
            train_samples,
            train_df,
            args.model_name_or_path,
            total_max_len=args.total_max_len,
            md_max_len=args.md_max_len,
        )
        valid_samples = generate_pairs_with_label(valid_df, mode="test")
        valid_ds = PairwiseDataset(
            valid_samples,
            valid_df,
            args.model_name_or_path,
            total_max_len=args.total_max_len,
            md_max_len=args.md_max_len,
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

        if args.train_mode == "pointwise":
            criterion = torch.nn.L1Loss()
        else:
            criterion = torch.nn.BCEWithLogitsLoss()

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

            if args.train_mode == "pointwise":
                y_val, y_pred = validate(model, valid_loader)
                valid_df["pred"] = valid_df.groupby(["id", "cell_type"])["rank"].rank(
                    pct=True
                )
                valid_df.loc[valid_df["cell_type"] == "markdown", "pred"] = y_pred
                y_dummy = (
                    valid_df.sort_values("pred").groupby("id")["cell_id"].apply(list)
                )
                print("Preds score", kendall_tau(df_orders.loc[y_dummy.index], y_dummy))
            else:

                def sigmoid(z):
                    return 1 / (1 + np.exp(-z))

                y_val, y_pred = validate(model, valid_loader)
                y_pred = sigmoid(y_pred) > 0.5
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_val, y_pred, average="binary", zero_division=0
                )
                print(f"precision: {precision}")
                print(f"recall: {recall}")
                print(f"f1: {f1}")

            torch.save(model.state_dict(), f"./{output_dir}/model_{e}.bin")

        return model, y_pred

    model = PercentileRegressor(args.model_name_or_path)
    model = model.cuda()
    model, y_pred = train(model, train_loader, valid_loader, epochs=args.epochs)
