import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (PairwiseDataset, PointwiseDataset,
                     SlidingWindowPointwiseDataset)
from metrics import kendall_tau
from model import PercentileRegressor
from preprocess import build_sliding_window_pairs

parser = argparse.ArgumentParser(description="평가 관련 파라미터")
parser.add_argument("--checkpoint-path", type=str, required=True)
parser.add_argument("--model-name-or-path", type=str, default="microsoft/codebert-base")
parser.add_argument("--data-dir", type=str, default="./data/")
parser.add_argument("--train-orders-path", type=str, default="./data/train_orders.csv")
parser.add_argument("--valid-md-path", type=str, default="./data/valid_md.csv")
parser.add_argument(
    "--valid-context-path", type=str, default="./data/valid_ctx_40.json"
)
parser.add_argument("--valid-path", type=str, default="./data/valid.csv")

parser.add_argument("--md-max-len", type=int, default=48)
parser.add_argument("--total-max-len", type=int, default=512)
parser.add_argument("--n-workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-mode", type=str, default="pointwise")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_pairs_with_label(df, mode="train", negative_seletion_ratio=0.05):
    samples = []
    for id, df_sub in tqdm(df.groupby("id")):
        df_sub_md = df_sub[df_sub["cell_type"] == "markdown"]
        df_sub_code = df_sub[df_sub["cell_type"] == "code"]
        df_sub_code_rank = df_sub_code["rank"].values
        df_sub_code_cell_id = df_sub_code["cell_id"].values
        for md_cell_id, md_rank in df_sub_md[["cell_id", "rank"]].values:
            labels = np.array(
                [((md_rank + 1) == code_rank) for code_rank in df_sub_code_rank]
            ).astype("int")
            for code_cell_id, label in zip(df_sub_code_cell_id, labels):
                if mode == "test":
                    samples.append([md_cell_id, code_cell_id, label])
                elif label == 1:
                    samples.append([md_cell_id, code_cell_id, label])
                elif label == 0 and random.uniform(0, 1) < negative_seletion_ratio:
                    samples.append([md_cell_id, code_cell_id, label])
    return samples


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


def valid(model, valid_loader, df_valid, df_orders, args):

    if args.test_mode == "pointwise":
        y_val, y_preds = validate(model, valid_loader)
        df_valid["pred"] = df_valid.groupby(["id", "cell_type"])["rank"].rank(pct=True)
        df_valid.loc[df_valid["cell_type"] == "markdown", "pred"] = y_preds
        pred_orders = df_valid.sort_values("pred").groupby("id")["cell_id"].apply(list)
        _pred_orders = pd.concat([pred_orders, df_orders.loc[pred_orders.index]], 1)
        _pred_orders.to_csv("./pred_gt_orders_40ctx.csv")
        print("Preds score", kendall_tau(df_orders.loc[pred_orders.index], pred_orders))

    elif args.test_mode == "sliding-window-pointwise":
        _, y_preds = validate(model, valid_loader)
        df_valid["pred"] = df_valid.groupby(["id", "cell_type"])["rank"].rank(pct=True)
        id2preds = defaultdict(list)
        for pair, y_pred in zip(valid_sliding_window_pairs, y_preds):
            md_id = pair[0]
            id2preds[md_id].append(y_pred)
        for idx in range(len(df_valid)):
            row = df_valid.iloc[idx]
            df_valid.at[idx, "pred"] = max(id2preds[f"{row.id}-{row.cell_id}"])
        y_dummy = df_valid.sort_values("pred").groupby("id")["cell_id"].apply(list)
        print("Preds score", kendall_tau(df_orders.loc[y_dummy.index], y_dummy))
    else:

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        gt, preds = validate(model, valid_loader)
        preds = sigmoid(preds) > 0.5
        precision, recall, f1, _ = precision_recall_fscore_support(
            gt, preds, average="binary", zero_division=0
        )
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1: {f1}")


if __name__ == "__main__":
    args = parser.parse_args()
    seed_everything(args.seed)

    print(json.dumps(vars(args), indent=2))

    assert args.test_mode in ["pointwise", "sliding-window-pointwise", "pairwise"]
    assert args.model_name_or_path in [
        "microsoft/codebert-base",
        "microsoft/graphcodebert-base",
        "huggingface/CodeBERTa-small-v1",
    ]

    data_dir = Path(args.data_dir)

    df_valid_md = (
        pd.read_csv(args.valid_md_path)
        .drop("parent_id", axis=1)
        .dropna()
        .reset_index(drop=True)
    )
    df_valid = pd.read_csv(args.valid_path)

    df_orders = pd.read_csv(
        args.train_orders_path,
        index_col="id",
        squeeze=True,
    ).str.split()

    if args.test_mode == "pointwise":
        valid_ctx = json.load(open(args.valid_context_path))
        valid_ds = PointwiseDataset(
            df_valid_md,
            model_name_or_path=args.model_name_or_path,
            total_max_len=args.total_max_len,
            md_max_len=args.md_max_len,
            ctx=valid_ctx,
        )
    elif args.test_mode == "sliding-window-pointwise":
        valid_sliding_window_pairs = build_sliding_window_pairs(
            df_valid, args.window_size
        )
        valid_ds = SlidingWindowPointwiseDataset(
            valid_sliding_window_pairs,
            args.model_name_or_path,
            total_max_len=args.total_max_len,
            md_max_len=args.md_max_len,
        )
    elif args.test_mode == "pairwise":
        valid_samples = generate_pairs_with_label(df_valid, mode="test")
        valid_ds = PairwiseDataset(
            valid_samples,
            df_valid,
            args.model_name_or_path,
            total_max_len=args.total_max_len,
            md_max_len=args.md_max_len,
        )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=64,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=False,
        drop_last=False,
    )

    model = PercentileRegressor(args.model_name_or_path)
    model.eval()
    model.load_state_dict(torch.load(args.checkpoint_path))
    model = model.cuda()
    model = valid(model, valid_loader, df_valid, df_orders, args=args)
