import argparse
from typing import List, Dict, Tuple
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

from dataset import (PairwiseDataset, PointwiseDataset, CTPairwiseDataset, SiameseDataset)
from metrics import kendall_tau
from model import PercentileRegressor, RepresExtractor
from preprocess import build_context_dict
from train import generate_pairs_like_kendalltau

parser = argparse.ArgumentParser(description="평가 관련 파라미터")
parser.add_argument("--data-dir", type=str, default="./data/")
parser.add_argument("--train-orders-path", type=str, default="./data/train_orders.csv")
parser.add_argument("--valid-path", type=str, default="./data/valid.csv")
parser.add_argument("--n-workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-mode", type=str, default="pointwise")
parser.add_argument("--output-as-file", action="store_true")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_pairs_with_label(df, mode="train", negative_seletion_ratio=0.05):
    samples = []
    for n_id, df_sub in tqdm(df.groupby("id")):
        df_sub_md = df_sub[df_sub["cell_type"] == "markdown"]
        df_sub_code = df_sub[df_sub["cell_type"] == "code"]
        df_sub_code_cell_id = df_sub_code["cell_id"].values
        for md_cell_id, md_rank in df_sub_md[["cell_id", "rank"]].values:
            for code_cell_id in df_sub_code_cell_id:
                samples.append([n_id, md_cell_id, code_cell_id, 0])
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


def predict_pointwise(
    df_test, model_path, ckpt_path, test_ctx, total_max_len, md_max_len
):

    model = PercentileRegressor(model_path)

    if ".pt" in ckpt_path:
        model.load_state_dict(torch.load(ckpt_path)["model_state"])
    else:
        model.load_state_dict(torch.load(ckpt_path))

    model = model.cuda()
    test_ds = PointwiseDataset(
        df_test[df_test["cell_type"] == "markdown"].reset_index(drop=True),
        model_name_or_path=model_path,
        total_max_len=total_max_len,
        md_max_len=md_max_len,
        ctx=test_ctx,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
        drop_last=False,
    )
    _, y_test = validate(model, test_loader)
    return y_test


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict_pairwise(df_test, model_path, ckpt_path, total_max_len, md_max_len):

    model = PercentileRegressor(model_path, hidden_dim=512)

    if ".pt" in ckpt_path:
        model.load_state_dict(torch.load(ckpt_path)["model_state"])
    else:
        model.load_state_dict(torch.load(ckpt_path))

    model = model.cuda()

    test_samples = generate_pairs_with_label(df_test, mode="test")
    test_ds = PairwiseDataset(
        test_samples,
        df_test,
        model_path,
        total_max_len,
        md_max_len,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
        drop_last=False,
    )

    _, y_test = validate(model, test_loader)

    preds_copy = y_test  # sigmoid(y_test)
    preds = []

    count = 0
    for id, df_tmp in tqdm(df_test.groupby("id")):
        df_tmp_md = df_tmp[df_tmp["cell_type"] == "markdown"]
        df_tmp_code = df_tmp[df_tmp["cell_type"] != "markdown"]
        df_tmp_code_pred = df_tmp_code["pred"].values
        n_code = len(df_tmp_code_pred)
        n_md = len(df_tmp_md)

        preds_tmp = preds_copy[count : count + n_md * n_code]

        # 마크다운 셀 하나하나 마다 코드셀 위치를 결정
        for i in range(n_md):
            pred = preds_tmp[i * n_code : i * n_code + n_code]
            e_pred = np.exp(pred - np.max(pred))
            softmax = e_pred / e_pred.sum()
            idx = np.argmax(softmax)
            rank = df_tmp_code_pred[idx] - 0.001
            preds.append(rank)

        count += n_md * n_code

        del df_tmp_md, df_tmp_code, df_tmp_code_pred

    return preds


def sorted_code_cells(pred_pairs, sorted_code_cell_ids):
    cell_ids = set([pair[0] for pair in pred_pairs])
    # md_cell_id 를 key로 갖는 md-md pair - {md_cell : rank}
    md_md_pairs = {cell_id:0 for cell_id in cell_ids if cell_id not in sorted_code_cell_ids}
    # md_cell_id 를 key로 갖는 md-cd pair - {md_cell : {code_cell : pred}}
    md_cd_pairs = {cell_id:{} for cell_id in md_md_pairs.keys()}
    for pair in pred_pairs:
        if pair[0] in md_md_pairs:
            if pair[1] in md_md_pairs:
                if pair[2] > 0.5:
                    md_md_pairs[pair[1]] +=1
                else:
                    md_md_pairs[pair[0]] +=1
            else:
                md_cd_pairs[pair[0]][pair[1]]=pair[2]
    # md_md_pairs 에서 rank 정보를 이용해 md_cell들을 정렬
    sorted_md_cell_ids = [pair[0] for pair in sorted(list(md_md_pairs.items()), key=lambda x:x[1])]
    sorted_cell_ids = []
    cur_idx = 0
    for md_id in sorted_md_cell_ids:
        # 현재 code_cell이 md_cell보다 위에 있다면 sorted_cell_ids에 계속 저장
        while cur_idx < len(sorted_code_cell_ids) and md_cd_pairs[md_id][sorted_code_cell_ids[cur_idx]] < 0.5:
            sorted_cell_ids.append(sorted_code_cell_ids[cur_idx])
            cur_idx +=1
        sorted_cell_ids.append(md_id)
    # md_cell 배정이 다 끝나고 code_cell이 밑에 남아있는 경우 마저 sorted_cell_ids에 저장
    while cur_idx < len(sorted_code_cell_ids):
        sorted_cell_ids.append(sorted_code_cell_ids[cur_idx])
        cur_idx +=1
    return sorted_cell_ids


def sort_code_cells(n_id, pred_pairs: List[Tuple[str, str, float]], sorted_code_cell_ids: List[str]):

        # md-md, code-md, md-code 셀 정렬
        cell_orders_dict = defaultdict(int)
        for cell_id_a, cell_id_b, prob in pred_pairs:
            if prob > 0.5:
                cell_orders_dict[cell_id_b] += 1
            else:
                cell_orders_dict[cell_id_a] += 1

            if cell_id_a not in cell_orders_dict:
                cell_orders_dict[cell_id_a] = 0

            if cell_id_b not in cell_orders_dict:
                cell_orders_dict[cell_id_b] = 0

        sorted_pred_cell_ids = sorted(list(cell_orders_dict.items()), key=lambda x: x[1])
        sorted_pred_cell_ids = [item[0] for item in sorted_pred_cell_ids]

        # 정렬된 코드셀에 끼워 맞추기
        for code_cell_id in sorted_code_cell_ids:
            assert code_cell_id in sorted_pred_cell_ids, f"{code_cell_id} not exist"

        return sorted_pred_cell_ids


def predict_ct_pairwise(df_test, model_path, ckpt_path, total_max_len, md_max_len):

    model = PercentileRegressor(model_path, hidden_dim=768)

    if ".pt" in ckpt_path:
        model.load_state_dict(torch.load(ckpt_path)["model_state"])
    else:
        model.load_state_dict(torch.load(ckpt_path))

    model = model.cuda()

    test_samples = generate_pairs_like_kendalltau(df_test)
    test_ds = CTPairwiseDataset(
        test_samples,
        df_test,
        model_path,
        total_max_len,
        md_max_len,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
        drop_last=False,
    )
    # 코드셀만 있는 경우가 있는 듯?

    _, y_preds = validate(model, test_loader)

    #: 노트북 아이디 당 정렬된 코드 리스트 리턴
    sorted_code_cell_series = df_test[df_test["cell_type"] == "code"].groupby("id")["cell_id"].apply(list)

    #: 노트북 아이디 리스트
    n_ids = list(sorted_code_cell_series.index.values)

    #: {n_id}-{cell_id} 를 전달 하면 셀 타입을 알려 줄 수 있도록
    unique_ids = [
        f"{n_id}-{cell_id}"
        for n_id, cell_id in zip(df_test["id"].values, df_test["cell_id"].values)
    ]
    id2type = dict(zip(unique_ids, df_test["cell_type"].values))

    #: 노트북 당 (md-md, code-md, md-code) pairs에 대한 추론 정보를 담는 딕셔너리
    preds_dict = defaultdict(list)
    for sample, pred in zip(test_samples, y_preds):
        n_id, cell_a, cell_b, _ = sample
        preds_dict[n_id].append((cell_a, cell_b, sigmoid(pred)))

    sorted_by_n_ids = dict()
    for n_id in n_ids:
        pred_pairs = preds_dict[n_id]
        sorted_code_cell_ids = sorted_code_cell_series.loc[n_id]
        sorted_cell_ids = sorted_code_cells(pred_pairs, sorted_code_cell_ids)
        sorted_by_n_ids[n_id] = sorted_cell_ids

    for idx, row in df_test.iterrows():
        n_id = row["id"]
        cell_id = row.cell_id
        sorted_ids = sorted_by_n_ids[n_id]
        n = len(sorted_ids)
        df_test.at[idx, "pred"] = round(sorted_ids.index(cell_id) / n, 3)

    return df_test

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


def predict_siamese(df_test, model_path, ckpt_path, total_max_len):

    model = RepresExtractor(model_path, hidden_dim=128, dropout=0.1)

    if ".pt" in ckpt_path:
        model.load_state_dict(torch.load(ckpt_path)["model_state"])
    else:
        model.load_state_dict(torch.load(ckpt_path))

    model.eval()
    model.cuda()

    test_samples = generate_pairs_with_label(df_test, mode="test")
    test_ds = SiameseDataset(
        test_samples,
        df_test,
        model_path,
        total_max_len=128,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=512,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
        drop_last=False,
    )

    _, y_test = validate_siamese(model, test_loader)

    preds_copy = y_test
    preds = []

    count = 0
    for id, df_tmp in tqdm(df_test.groupby("id")):
        df_tmp_md = df_tmp[df_tmp["cell_type"] == "markdown"]
        df_tmp_code = df_tmp[df_tmp["cell_type"] != "markdown"]
        df_tmp_code_pred = df_tmp_code["pred"].values
        n_code = len(df_tmp_code_pred)
        n_md = len(df_tmp_md)

        preds_tmp = preds_copy[count : count + n_md * n_code]

        # 마크다운 셀 하나하나 마다 코드셀 위치를 결정
        for i in range(n_md):
            pred = preds_tmp[i * n_code : i * n_code + n_code]
            e_pred = np.exp(pred - np.max(pred))
            softmax = e_pred / e_pred.sum()
            idx = np.argmax(softmax)
            rank = df_tmp_code_pred[idx] - 0.001
            preds.append(rank)

        count += n_md * n_code

        del df_tmp_md, df_tmp_code, df_tmp_code_pred

    return preds


if __name__ == "__main__":
    """
    .. note::
        pointwise 간 weight 구하는 것도 해보면 좋을 듯!
    """

    args = parser.parse_args()
    seed_everything(args.seed)

    print(json.dumps(vars(args), indent=2))

    data_dir = Path(args.data_dir)

    df_valid = pd.read_csv(args.valid_path)
    # df_valid = df_valid[:40232]
    df_valid_md = df_valid[df_valid["cell_type"] == "markdown"]

    df_orders = pd.read_csv(
        args.train_orders_path,
        index_col="id",
        squeeze=True,
    ).str.split()

    if args.test_mode == "ensemble" or args.test_mode == "pointwise":

        model_name_or_path = "microsoft/graphcodebert-base"
        checkpoint_path = "30random-ctx-added-order-graph-10ep/model_9.pt"

        valid_context_dict_1 = build_context_dict(
            df_valid,
            30,
            make_sample_randomly=True,
            insert_cell_order=True,
        )
        valid_context_dict_2 = build_context_dict(
            df_valid,
            40,
            make_sample_randomly=True,
            insert_cell_order=True,
        )

        y_test_1 = predict_pointwise(
            df_valid,
            model_path=model_name_or_path,
            ckpt_path=checkpoint_path,
            test_ctx=valid_context_dict_1,
            total_max_len=512,
            md_max_len=48,
        )
        y_test_2 = predict_pointwise(
            df_valid,
            model_path=model_name_or_path,
            ckpt_path=checkpoint_path,
            test_ctx=valid_context_dict_2,
            total_max_len=512,
            md_max_len=48,
        )

        # model_name_or_path = "microsoft/graphcodebert-base"
        # checkpoint_path = "pointwise-add-data-graphcodebert-40ctx/model_4.bin"

        # valid_context_dict_3 = build_context_dict(
        #     df_valid, 40, make_sample_from_last=False  # 하나만 한다면 False가 나음
        # )

        # y_test_3 = predict_pointwise(
        #     df_valid,
        #     model_path=model_name_or_path,
        #     ckpt_path=checkpoint_path,
        #     test_ctx=valid_context_dict_3,
        #     total_max_len=512,
        #     md_max_len=48,
        # )

        # model_name_or_path = "microsoft/codebert-base"
        # checkpoint_path = "pointwise-add-data-codebert-30ctx/model_4.bin"

        # valid_context_dict_4 = build_context_dict(
        #     df_valid, 30, make_sample_from_last=True
        # )

        # y_test_4 = predict_pointwise(
        #     df_valid,
        #     model_path=model_name_or_path,
        #     ckpt_path=checkpoint_path,
        #     test_ctx=valid_context_dict_4,
        #     total_max_len=512,
        #     md_max_len=48,
        # )

        # preds_pointwise = (y_test_1 + y_test_2 + y_test_3 + y_test_4) / 4
        preds_pointwise = (y_test_1 + y_test_2) / 2

        # preds_pointwise = (
        #     0.8 * (0.65 * (0.4 * y_test_1 + 0.6 * y_test_2) + 0.35 * y_test_3)
        #     + 0.2 * y_test_4
        # )

        # aw = [round(n / 100, 4) for n in range(20, 95)]
        # bw = [round(1.0 - w, 4) for w in aw]

        # a = np.array(preds_pointwise)
        # b = np.array(y_test_4)

        # df_valid["rank"] = df_valid.groupby(["id", "cell_type"]).cumcount()
        # df_valid["pred"] = df_valid.groupby(["id", "cell_type"])["rank"].rank(pct=True)
        # for _aw, _bw in zip(aw, bw):
        #     preds = a * _aw + b * _bw
        #     df_valid.loc[df_valid["cell_type"] == "markdown", "pred"] = preds
        #     pred_orders = (
        #         df_valid.sort_values("pred").groupby("id")["cell_id"].apply(list)
        #     )
        #     print(
        #         f"Preds score ({_aw}, {_bw})",
        #         kendall_tau(df_orders.loc[pred_orders.index], pred_orders),
        #     )

        df_valid["pred"] = df_valid.groupby(["id", "cell_type"])["rank"].rank(pct=True)
        df_valid.loc[df_valid["cell_type"] == "markdown", "pred"] = preds_pointwise
        pred_orders = df_valid.sort_values("pred").groupby("id")["cell_id"].apply(list)

        print("Preds score", kendall_tau(df_orders.loc[pred_orders.index], pred_orders))
        if args.output_as_file:
            _pred_orders = pd.concat([pred_orders, df_orders.loc[pred_orders.index]], 1)
            _pred_orders.to_csv("./output_pointwise.csv")

    if args.test_mode == "ensemble" or args.test_mode == "pairwise":

        model_name_or_path = "prajjwal1/bert-small"
        checkpoint_path = "bert-small-128-pairwise-v6/model_4.pt"

        df_valid["rank"] = df_valid.groupby(["id", "cell_type"]).cumcount()
        df_valid["pred"] = df_valid.groupby(["id", "cell_type"])["rank"].rank(pct=True)

        preds_pairwise = predict_pairwise(
            df_valid,
            model_path=model_name_or_path,
            ckpt_path=checkpoint_path,
            total_max_len=128,
            md_max_len=64,
        )
        df_valid.loc[df_valid["cell_type"] == "markdown", "pred"] = preds_pairwise
        pred_orders = df_valid.sort_values("pred").groupby("id")["cell_id"].apply(list)

        print("Preds score", kendall_tau(df_orders.loc[pred_orders.index], pred_orders))
        if args.output_as_file:
            _pred_orders = pd.concat([pred_orders, df_orders.loc[pred_orders.index]], 1)
            _pred_orders.to_csv("./output_pairwise.csv")

    if args.test_mode == "ensemble" or args.test_mode == "ct-pairwise":

        df_valid = df_valid.groupby("id").filter(lambda x: len(x) <= 15)

        model_name_or_path = "microsoft/graphcodebert-base"
        checkpoint_path = "outputs_pairwise_pairwise-graph--under-15_42/model_2.pt"

        df_valid["rank"] = df_valid.groupby(["id", "cell_type"]).cumcount()
        df_valid["pred"] = df_valid.groupby(["id", "cell_type"])["rank"].rank(pct=True)

        df_valid = predict_ct_pairwise(
            df_valid,
            model_path=model_name_or_path,
            ckpt_path=checkpoint_path,
            total_max_len=128,
            md_max_len=64,
        )
        # df_valid.loc[df_valid["cell_type"] == "markdown", "pred"] = preds_pairwise
        pred_orders = df_valid.sort_values("pred").groupby("id")["cell_id"].apply(list)

        print("Preds score", kendall_tau(df_orders.loc[pred_orders.index], pred_orders))
        if args.output_as_file:
            _pred_orders = pd.concat([pred_orders, df_orders.loc[pred_orders.index]], 1)
            _pred_orders.to_csv("./output_pairwise.csv")

    if args.test_mode == "ensemble" or args.test_mode == "siamese":

        model_name_or_path = "huggingface/CodeBERTa-small-v1"
        checkpoint_path = "./outputs_siamese_siamese_42/model_2.pt"

        df_valid["rank"] = df_valid.groupby(["id", "cell_type"]).cumcount()
        df_valid["pred"] = df_valid.groupby(["id", "cell_type"])["rank"].rank(pct=True)

        preds_pairwise = predict_siamese(
            df_valid,
            model_path=model_name_or_path,
            ckpt_path=checkpoint_path,
            total_max_len=128,
        )
        df_valid.loc[df_valid["cell_type"] == "markdown", "pred"] = preds_pairwise
        pred_orders = df_valid.sort_values("pred").groupby("id")["cell_id"].apply(list)

        print("Preds score", kendall_tau(df_orders.loc[pred_orders.index], pred_orders))
        if args.output_as_file:
            _pred_orders = pd.concat([pred_orders, df_orders.loc[pred_orders.index]], 1)
            _pred_orders.to_csv("./output_pairwise.csv")

    if args.test_mode == "ensemble":

        print("find best_weight for ensemble")

        pointwise_weight = [round(n / 100, 4) for n in range(20, 95)]
        pairwise_weight = [round(1.0 - w, 4) for w in pointwise_weight]

        preds_pointwise = np.array(preds_pointwise)
        preds_pairwise = np.array(preds_pairwise)

        for po_w, pa_w in zip(pointwise_weight, pairwise_weight):
            preds = preds_pointwise * po_w + preds_pairwise * pa_w
            df_valid.loc[df_valid["cell_type"] == "markdown", "pred"] = preds
            pred_orders = (
                df_valid.sort_values("pred").groupby("id")["cell_id"].apply(list)
            )
            print(
                f"Preds score ({po_w}, {pa_w})",
                kendall_tau(df_orders.loc[pred_orders.index], pred_orders),
            )
