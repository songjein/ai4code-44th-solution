import argparse
import json
import os
import random
import sys
from collections import defaultdict
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

from dataset import (PairwiseDataset, PointwiseDataset,
                     SlidingWindowPointwiseDataset)
from metrics import kendall_tau
from model import PercentileRegressor
from preprocess import build_sliding_window_pairs

parser = argparse.ArgumentParser(description="학습 관련 파라미터")
parser.add_argument("--model-name-or-path", type=str, default="microsoft/codebert-base")
parser.add_argument("--data-dir", type=str, default="./data/")
parser.add_argument("--train-orders-path", type=str, default="./data/train_orders.csv")
parser.add_argument("--train-md-path", type=str, default="./data/train_md.csv")
parser.add_argument("--train-context-path", type=str, default="./data/train_ctx.json")
parser.add_argument(
    "--train-sliding-window-pairs-path",
    type=str,
    default="./data/train_sliding_window_30_pairs.json",
)
parser.add_argument("--valid-md-path", type=str, default="./data/valid_md.csv")
parser.add_argument("--valid-context-path", type=str, default="./data/valid_ctx.json")
parser.add_argument(
    "--valid-sliding-window-pairs-path",
    type=str,
    default="./data/valid_sliding_window_30_pairs.json",
)
parser.add_argument("--train-path", type=str, default="./data/train.csv")
parser.add_argument("--valid-path", type=str, default="./data/valid.csv")

parser.add_argument("--learning-rate", type=float, default=3e-5)
parser.add_argument("--md-max-len", type=int, default=48)
parser.add_argument("--total-max-len", type=int, default=512)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--accumulation-steps", type=int, default=4)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--n-workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train-mode", type=str, default="pointwise")
parser.add_argument("--memo", type=str, default="")
parser.add_argument("--window-size", type=int, default=30)
parser.add_argument("--hidden-size", type=int, default=768)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_pairs_with_label(df, mode="train", negative_seletion_ratio=0.05):
    samples = []
    if mode == "test":
        for id, df_sub in tqdm(df.groupby("id")):
            df_sub_md = df_sub[df_sub["cell_type"] == "markdown"]
            df_sub_code = df_sub[df_sub["cell_type"] == "code"]
            df_sub_code_cell_id = df_sub_code["cell_id"].values
            for md_cell_id, md_rank in df_sub_md[["cell_id", "rank"]].values:
                for code_cell_id in df_sub_code_cell_id:
                    samples.append([md_cell_id, code_cell_id, 0])
    else:
        for id, df_sub in tqdm(df.groupby("id")):
            df_sub_md = df_sub[df_sub["cell_type"] == "markdown"]
            df_sub_code = df_sub[df_sub["cell_type"] == "code"]
            md_ranks_set = set(df_sub_md["rank"].values)
            code_ranks = df_sub_code["rank"].values
            df_sub_code_cell_id = df_sub_code["cell_id"].values

            # 3연속 md 까지만 허용(?)
            for md_cell_id, md_rank in df_sub_md[["cell_id", "rank"]].values:
                code_id_label_pairs = []
                for j, code_rank in enumerate(code_ranks):
                    label = 0
                    if (md_rank + 1) == code_rank:
                        label = 1
                    elif (md_rank + 1) in md_ranks_set:
                        if (md_rank + 2) == code_rank:
                            label = 1
                        elif (md_rank + 2) in md_ranks_set:
                            if (md_rank + 3) == code_rank:
                                label = 1
                    code_id = df_sub_code_cell_id[j]
                    code_id_label_pairs.append((code_id, label))

                for code_cell_id, label in code_id_label_pairs:
                    if label == 1:
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


def train(model, train_loader, valid_loader, df_valid, df_orders, args):
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.05 * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )

    if args.train_mode in ["pointwise", "sliding-window-pointwise"]:
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler()

    for e in range(args.epochs):
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

            tbar.set_description(f"Epoch {e + 1} Loss: {avg_loss}")

        if args.train_mode == "pointwise":
            y_val, y_preds = validate(model, valid_loader)
            df_valid["pred"] = df_valid.groupby(["id", "cell_type"])["rank"].rank(
                pct=True
            )
            df_valid.loc[df_valid["cell_type"] == "markdown", "pred"] = y_preds
            y_dummy = df_valid.sort_values("pred").groupby("id")["cell_id"].apply(list)
            print("Preds score", kendall_tau(df_orders.loc[y_dummy.index], y_dummy))

        elif args.train_mode == "sliding-window-pointwise":
            _, y_preds = validate(model, valid_loader)
            df_valid["pred"] = df_valid.groupby(["id", "cell_type"])["rank"].rank(
                pct=True
            )
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

        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            },
            f"./{output_dir}/model_{e}.pt",
        )

    return model


if __name__ == "__main__":
    args = parser.parse_args()
    seed_everything(args.seed)

    print(json.dumps(vars(args), indent=2))

    assert args.train_mode in ["pointwise", "sliding-window-pointwise", "pairwise"]
    assert args.model_name_or_path in [
        "microsoft/codebert-base",
        "microsoft/graphcodebert-base",
        "huggingface/CodeBERTa-small-v1",
        "prajjwal1/bert-small",
    ]

    output_dir = f"outputs_{args.train_mode}_{args.memo}_{args.seed}"
    os.makedirs(f"./{output_dir}", exist_ok=True)
    data_dir = Path(args.data_dir)

    df_train_md = (
        pd.read_csv(args.train_md_path).drop("parent_id", axis=1).reset_index(drop=True)
    )

    df_valid_md = (
        pd.read_csv(args.valid_md_path)
        .drop("parent_id", axis=1)
        .dropna()
        .reset_index(drop=True)
    )
    df_valid = pd.read_csv(args.valid_path)

    #: external 데이터에 대한 정보는 없지만, 벨리데이션 셋은 원본 학습 데이터에서만 나왔기 때문에 상관 없음
    df_orders = pd.read_csv(
        args.train_orders_path,
        index_col="id",
        squeeze=True,
    ).str.split()

    if args.train_mode == "pointwise":
        train_ctx = json.load(open(args.train_context_path))
        train_ds = PointwiseDataset(
            df_train_md,
            model_name_or_path=args.model_name_or_path,
            md_max_len=args.md_max_len,
            total_max_len=args.total_max_len,
            ctx=train_ctx,
        )
        valid_ctx = json.load(open(args.valid_context_path))
        valid_ds = PointwiseDataset(
            df_valid_md,
            model_name_or_path=args.model_name_or_path,
            total_max_len=args.total_max_len,
            md_max_len=args.md_max_len,
            ctx=valid_ctx,
        )
    elif args.train_mode == "sliding-window-pointwise":
        df_train = pd.read_csv(args.train_path)
        train_sliding_window_pairs = build_sliding_window_pairs(
            df_train, args.window_size
        )
        train_ds = SlidingWindowPointwiseDataset(
            train_sliding_window_pairs,
            args.model_name_or_path,
            total_max_len=args.total_max_len,
            md_max_len=args.md_max_len,
        )
        valid_sliding_window_pairs = build_sliding_window_pairs(
            df_valid, args.window_size
        )
        valid_ds = SlidingWindowPointwiseDataset(
            valid_sliding_window_pairs,
            args.model_name_or_path,
            total_max_len=args.total_max_len,
            md_max_len=args.md_max_len,
        )
    elif args.train_mode == "pairwise":
        df_train = pd.read_csv(args.train_path)
        train_samples = generate_pairs_with_label(df_train, mode="train")
        train_ds = PairwiseDataset(
            train_samples,
            df_train,
            args.model_name_or_path,
            total_max_len=args.total_max_len,
            md_max_len=args.md_max_len,
        )
        valid_samples = generate_pairs_with_label(df_valid, mode="train")
        valid_ds = PairwiseDataset(
            valid_samples,
            df_valid,
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

    model = PercentileRegressor(args.model_name_or_path, hidden_dim=args.hidden_size)
    # model = PercentileRegressor("./pretrained_128_prajjwal1/bert-small/", hidden_dim=args.hidden_size)
    model = model.cuda()
    model = train(model, train_loader, valid_loader, df_valid, df_orders, args=args)
