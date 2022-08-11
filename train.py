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

from dataset import (CTPairwiseDataset, PairwiseDataset, PointwiseDataset,
                     SiameseDataset)
from metrics import kendall_tau
from model import PercentileRegressor, RepresExtractor
from preprocess import build_context_dict

parser = argparse.ArgumentParser(description="학습 관련 파라미터")
parser.add_argument("--model-name-or-path", type=str, default="microsoft/codebert-base")
parser.add_argument("--checkpoint-path", type=str)
parser.add_argument("--data-dir", type=str, default="./data/")
parser.add_argument("--train-orders-path", type=str, default="./data/train_orders.csv")
parser.add_argument("--train-path", type=str, default="./data/concat_train.csv")
parser.add_argument("--valid-path", type=str, default="./data/valid.csv")
parser.add_argument("--learning-rate", type=float, default=3e-5)
parser.add_argument("--md-max-len", type=int, default=48)
parser.add_argument("--total-max-len", type=int, default=512)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--accumulation-steps", type=int, default=1)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--n-workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train-mode", type=str, default="pointwise")
parser.add_argument("--memo", type=str, default="")
parser.add_argument("--num-sampled-code-cell", type=int, default=40)
parser.add_argument("--sample-context-randomly", action="store_true")
parser.add_argument("--insert-cell-order", action="store_true")
parser.add_argument("--use-cached-pairs", action="store_true")
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
        for n_id, df_sub in tqdm(df.groupby("id")):
            df_sub_md = df_sub[df_sub["cell_type"] == "markdown"]
            df_sub_code = df_sub[df_sub["cell_type"] == "code"]
            df_sub_code_cell_id = df_sub_code["cell_id"].values
            for md_cell_id, md_rank in df_sub_md[["cell_id", "rank"]].values:
                for code_cell_id in df_sub_code_cell_id:
                    samples.append([n_id, md_cell_id, code_cell_id, 0])
    else:
        for n_id, df_sub in tqdm(df.groupby("id")):
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
                        samples.append([n_id, md_cell_id, code_cell_id, label])
                    elif label == 0 and random.uniform(0, 1) < negative_seletion_ratio:
                        samples.append([n_id, md_cell_id, code_cell_id, label])
    return samples


def generate_pairs_like_kendalltau(df):
    samples = []
    for n_id, df_sub in tqdm(df.groupby("id")):
        for cell_id_1, cell_type_1, rank_1 in df_sub[
            ["cell_id", "cell_type", "rank"]
        ].values:
            for cell_id_2, cell_type_2, rank_2 in df_sub[
                ["cell_id", "cell_type", "rank"]
            ].values:
                if cell_id_1 == cell_id_2:
                    continue
                if cell_type_1 == "code_cell" and cell_type_2 == "code_cell":
                    continue
                samples.append([n_id, cell_id_1, cell_id_2, int(rank_1 < rank_2)])
    print("generated samples:", len(samples))
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


def validate_siamese(model, valid_loader):
    model.eval()
    tbar = tqdm(valid_loader, file=sys.stdout)
    preds = []
    labels = []
    with torch.no_grad():
        for idx, data in enumerate(tbar):
            md_tokens_ids, md_mask, code_tokens_ids, code_mask, label = [
                item.cuda() for item in data
            ]

            md_repres = model(md_tokens_ids, md_mask)
            code_repres = model(code_tokens_ids, code_mask)
            distance = (md_repres - code_repres).norm(1, dim=-1)
            similarity = (distance * -1).exp()

            preds.append(similarity.detach().cpu().numpy().ravel())
            labels.append(label.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)


def train_siamese(model, train_loader, valid_loader, df_valid, df_orders, args):

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

    criterion = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    for e in range(args.epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            md_tokens_ids, md_mask, code_tokens_ids, code_mask, label = [
                item.cuda() for item in data
            ]

            with torch.cuda.amp.autocast():
                md_repres = model(md_tokens_ids, md_mask)
                code_repres = model(code_tokens_ids, code_mask)
                distance = (md_repres - code_repres).norm(1, dim=-1)
                similarity = (distance * -1).exp()
                loss = criterion(similarity.unsqueeze(1), label)

            scaler.scale(loss).backward()

            if idx % args.accumulation_steps == 0 or idx == len(tbar) - 1:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(similarity.detach().cpu().numpy().ravel())
            labels.append(label.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(f"Epoch {e + 1} Loss: {avg_loss}")

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        gt, preds = validate_siamese(model, valid_loader)
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

    assert args.train_mode in [
        "pointwise",
        "sliding-window-pointwise",
        "pairwise",
        "siamese",
        "ct-pairwise",
    ]
    assert args.model_name_or_path in [
        "microsoft/codebert-base",
        "microsoft/graphcodebert-base",
        "huggingface/CodeBERTa-small-v1",
        "prajjwal1/bert-small",
    ]

    output_dir = f"outputs_{args.train_mode}_{args.memo}_{args.seed}"
    os.makedirs(f"./{output_dir}", exist_ok=True)
    data_dir = Path(args.data_dir)

    df_train = pd.read_csv(args.train_path)
    if args.train_model == "ct-pairwise":
        df_train = df_train.groupby("id").filter(lambda x: len(x["cell_type"]) <= 15)
    df_train_md = (
        df_train[df_train["cell_type"] == "markdown"]
        .drop("parent_id", axis=1)
        .reset_index(drop=True)
    )

    df_valid = pd.read_csv(args.valid_path)
    if args.train_model == "ct-pairwise":
        df_valid = df_valid.groupby("id").filter(lambda x: len(x["cell_type"]) <= 15)
    df_valid_md = (
        df_valid[df_valid["cell_type"] == "markdown"]
        .drop("parent_id", axis=1)
        .reset_index(drop=True)
    )

    #: external 데이터에 대한 정보는 없지만, 벨리데이션 셋은 원본 학습 데이터에서만 나왔기 때문에 상관 없음
    df_orders = pd.read_csv(
        args.train_orders_path,
        index_col="id",
        squeeze=True,
    ).str.split()

    if args.train_mode == "pointwise":
        train_ctx = build_context_dict(
            df_train,
            args.num_sampled_code_cell,
            make_sample_randomly=args.sample_context_randomly,
            insert_cell_order=args.insert_cell_order,
        )
        train_ds = PointwiseDataset(
            df_train_md,
            model_name_or_path=args.model_name_or_path,
            md_max_len=args.md_max_len,
            total_max_len=args.total_max_len,
            ctx=train_ctx,
        )
        valid_ctx = build_context_dict(
            df_valid,
            args.num_sampled_code_cell,
            make_sample_randomly=args.sample_context_randomly,
            insert_cell_order=args.insert_cell_order,
        )
        valid_ds = PointwiseDataset(
            df_valid_md,
            model_name_or_path=args.model_name_or_path,
            total_max_len=args.total_max_len,
            md_max_len=args.md_max_len,
            ctx=valid_ctx,
        )
    elif args.train_mode == "ct-pairwise":
        if args.use_cached_pairs:
            with open("./data/train_pairs.json") as f:
                train_samples = json.loads(f.read())
        else:
            train_samples = generate_pairs_like_kendalltau(df_train)
        with open("./data/train_pairs.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(train_samples, ensure_ascii=False))
        train_ds = CTPairwiseDataset(
            train_samples,
            df_train,
            args.model_name_or_path,
            total_max_len=args.total_max_len,
            md_max_len=args.md_max_len,
        )
        if args.use_cached_pairs:
            with open("./data/valid_pairs.json") as f:
                valid_samples = json.loads(f.read())
        else:
            valid_samples = generate_pairs_like_kendalltau(df_valid)

        with open("./data/valid_pairs.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(valid_samples, ensure_ascii=False))
        valid_ds = CTPairwiseDataset(
            valid_samples,
            df_valid,
            args.model_name_or_path,
            total_max_len=args.total_max_len,
            md_max_len=args.md_max_len,
        )
    elif args.train_mode == "siamese":

        if args.use_cached_pairs:
            with open("./data/train_pairs.json") as f:
                train_samples = json.loads(f.read())
        else:
            train_samples = generate_pairs_with_label(df_train, mode="train")

        with open("./data/train_pairs.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(train_samples, ensure_ascii=False))

        train_ds = SiameseDataset(
            train_samples,
            df_train,
            args.model_name_or_path,
            total_max_len=args.total_max_len,
        )

        if args.use_cached_pairs:
            with open("./data/valid_pairs.json") as f:
                valid_samples = json.loads(f.read())
        else:
            valid_samples = generate_pairs_with_label(df_valid, mode="train")

        with open("./data/valid_pairs.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(valid_samples, ensure_ascii=False))

        valid_ds = SiameseDataset(
            valid_samples,
            df_valid,
            args.model_name_or_path,
            total_max_len=args.total_max_len,
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

    dropout = 0.1

    if args.train_mode == "pairwise":
        model = PercentileRegressor(
            "./pretrained_128_prajjwal1/bert-small/",
            hidden_dim=args.hidden_size,
            dropout=dropout,
        )
    elif args.train_mode == "siamese":
        model = RepresExtractor(
            args.model_name_or_path, hidden_dim=args.hidden_size, dropout=dropout
        )
    else:
        model = PercentileRegressor(
            args.model_name_or_path, hidden_dim=args.hidden_size, dropout=dropout
        )

    model = model.cuda()

    if args.train_mode == "siamese":
        model = train_siamese(
            model, train_loader, valid_loader, df_valid, df_orders, args=args
        )
    else:
        model = train(model, train_loader, valid_loader, df_valid, df_orders, args=args)
