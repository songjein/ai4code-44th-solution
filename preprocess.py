import argparse
import json
import math
import os
import random
import re
from glob import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

parser = argparse.ArgumentParser(description="전처리 관련 파라미터")
parser.add_argument("--root-path", type=str, default="./data")
parser.add_argument("--num-sampled-code-cell", type=int, default=30)
parser.add_argument("--window-size", type=int, default=30)
parser.add_argument("--skip-create-from-scratch", action="store_true")


def read_notebook(path):
    id = path.split("/")[-1][: -len(".json")]
    return (
        pd.read_json(path, dtype={"cell_type": "category", "source": "str"})
        .assign(id=id)
        .rename_axis("cell_id")
    )


def get_ranks(gt, derived):
    return [gt.index(d) for d in derived]


def clean_code(cell):
    for char in ["\r\n", "\r", "\n"]:
        cell = cell.replace(char, " ")
    cell = re.sub(r"\s{1,}", " ", cell)
    cell = re.sub(r"#{6,}", "", cell)
    cell = re.sub(r"http\S+[^)]", "", cell)
    cell = "\n".join([sent.strip() for sent in cell.split("\n")])
    return cell


def sample_cells(cells, n):
    """
    .. note::
        더 똑똑하게 추출하기
    """
    cells = [clean_code(cell) for cell in cells]
    if n >= len(cells):
        return cells
    else:
        results = []
        step = len(cells) / n
        idx = 0
        while int(np.round(idx)) < len(cells):
            results.append(cells[int(np.round(idx))])
            idx += step
        assert cells[0] in results
        if cells[-1] not in results:
            results[-1] = cells[-1]
        return results


def build_context_dict(df, num_sampled_code_cell=30):
    """
    .. note::
        더 똑똑하게 추출하기
    """
    features = dict()
    df = df.sort_values("rank").reset_index(drop=True)
    for idx, sub_df in tqdm(df.groupby("id")):
        features[idx] = dict()
        total_md = sub_df[sub_df.cell_type == "markdown"].shape[0]
        code_sub_df = sub_df[sub_df.cell_type == "code"]
        total_code = code_sub_df.shape[0]
        codes = sample_cells(code_sub_df.source.values, num_sampled_code_cell)
        features[idx]["total_code"] = total_code
        features[idx]["total_md"] = total_md
        features[idx]["codes"] = codes
    return features


def make_md_code_pairs_by_sliding_window(md_cells, code_cells, pct_ranks, window_size):
    """
    :return: Tuple(윈도우 인덱스, 전체 윈도우 개수, 마크다운 셀, 코드셀 윈도우, 랭크 퍼센타일)
    """
    pairs = []
    md_cells = [clean_code(cell) for cell in md_cells]
    code_cells = [clean_code(cell) for cell in code_cells]

    if window_size >= len(code_cells):
        window = code_cells
        for md_cell, pct_rank in zip(md_cells, pct_ranks):
            pairs.append((0, 1, md_cell, window, pct_rank))
    else:
        n_windows = math.ceil(len(code_cells) / window_size)
        for w_idx in range(n_windows):
            offset = w_idx * window_size
            window = code_cells[offset : offset + window_size]
            for md_cell, pct_rank in zip(md_cells, pct_ranks):
                pairs.append((w_idx, n_windows, md_cell, window, pct_rank))

    return pairs


def build_sliding_window_pairs(df, window_size=30):
    df = df.sort_values("rank").reset_index(drop=True)
    pairs = []
    for idx, sub_df in tqdm(df.groupby("id")):
        md_sub_df = sub_df[sub_df.cell_type == "markdown"]
        code_sub_df = sub_df[sub_df.cell_type == "code"]
        sub_pairs = make_md_code_pairs_by_sliding_window(
            md_sub_df.source.values,
            code_sub_df.source.values,
            md_sub_df.pct_rank.values,
            window_size,
        )
        pairs += sub_pairs
    return pairs


if __name__ == "__main__":
    """
    .. note::
        dropna를 끝나고 한 번 더 해줘야하는 문제가 있음
    """

    random.seed(42)
    args = parser.parse_args()

    if not args.skip_create_from_scratch:
        os.makedirs(args.root_path, exist_ok=True)

        train_paths = list(glob(f"{args.root_path}/train/*.json"))
        train_notebooks = [
            read_notebook(path)
            for path in tqdm(train_paths, desc="Read Train Notebooks")
        ]

        #: 노트북 아이디, 셀 아이디, 셀 타입, 소스(텍스트)
        df_all = (
            pd.concat(train_notebooks)
            .set_index("id", append=True)
            .swaplevel()
            .sort_index(level="id", sort_remaining=False)
        )

        #: 노트북 아이디별 cell_order(셀 순서)
        df_orders = pd.read_csv(
            f"{args.root_path}/train_orders.csv",
            index_col="id",
            squeeze=True,  # Series로 리턴 됨
        ).str.split()  # string 표현을 리스트로 스플릿

        #: 노트북 아이디별 cell_order(셀 정답 순서), cell_id(주어진 셀 아이디 순서)
        merged_df_orders = df_orders.to_frame().join(
            df_all.reset_index("cell_id").groupby("id")["cell_id"].apply(list),
            how="right",
        )

        ranks = {}
        for id_, cell_order, cell_id in merged_df_orders.itertuples():
            ranks[id_] = {"cell_id": cell_id, "rank": get_ranks(cell_order, cell_id)}

        #: 주어진 cell id에 원래 순서를 매핑한 df를 만듦
        df_ranks = (
            pd.DataFrame.from_dict(ranks, orient="index")
            .rename_axis("id")
            .apply(pd.Series.explode)
            .set_index("cell_id", append=True)
        )

        df_ancestors = pd.read_csv(
            f"{args.root_path}/train_ancestors.csv", index_col="id"
        )
        df_all = (
            df_all.reset_index()
            .merge(df_ranks, on=["id", "cell_id"])
            .merge(df_ancestors, on=["id"])
        )
        df_all = df_all.dropna(subset=["source", "rank"])  # TODO: 추가 문제 체크
        df_all["pct_rank"] = df_all["rank"] / df_all.groupby("id")["cell_id"].transform(
            "count"
        )

        # validation split
        valid_split = 0.1
        splitter = GroupShuffleSplit(n_splits=1, test_size=valid_split, random_state=42)
        train_ind, val_ind = next(splitter.split(df_all, groups=df_all["ancestor_id"]))
        df_train = (
            df_all.loc[train_ind]
            .dropna(subset=["source", "rank"])
            .reset_index(drop=True)
        )
        df_valid = (
            df_all.loc[val_ind].dropna(subset=["source", "rank"]).reset_index(drop=True)
        )
        df_train_md = (
            df_train[df_train["cell_type"] == "markdown"]
            .dropna(subset=["source", "rank"])
            .reset_index(drop=True)
        )
        df_valid_md = (
            df_valid[df_valid["cell_type"] == "markdown"]
            .dropna(subset=["source", "rank"])
            .reset_index(drop=True)
        )
        df_train_md.to_csv(f"{args.root_path}/train_md.csv", index=False)
        df_valid_md.to_csv(f"{args.root_path}/valid_md.csv", index=False)
        df_train.to_csv(f"{args.root_path}/train.csv", index=False)
        df_valid.to_csv(f"{args.root_path}/valid.csv", index=False)

    # 이상한 버그? nan 데이터가 여전히 포함되어 있어 다시 읽은 뒤 없애고 저장
    # 앞서 분명히 source/rank에 대한 nan row를 없앴는데도 매번 남아 있는 문제
    pd.read_csv(f"{args.root_path}/train_md.csv").dropna(
        subset=["source", "rank"]
    ).to_csv(f"{args.root_path}/train_md.csv", index=False)
    pd.read_csv(f"{args.root_path}/valid_md.csv").dropna(
        subset=["source", "rank"]
    ).to_csv(f"{args.root_path}/valid_md.csv", index=False)
    df_train = pd.read_csv(f"{args.root_path}/train.csv").dropna(
        subset=["source", "rank"]
    )
    df_train.to_csv(f"{args.root_path}/train.csv", index=False)
    df_valid = pd.read_csv(f"{args.root_path}/valid.csv").dropna(
        subset=["source", "rank"]
    )
    df_valid.to_csv(f"{args.root_path}/valid.csv", index=False)

    # k 개의 코드셀을 샘플링
    train_context_dict = build_context_dict(df_train, args.num_sampled_code_cell)
    json.dump(
        train_context_dict,
        open(f"{args.root_path}/train_ctx_{args.num_sampled_code_cell}.json", "wt"),
    )

    valid_context_dict = build_context_dict(df_valid, args.num_sampled_code_cell)
    json.dump(
        valid_context_dict,
        open(f"{args.root_path}/valid_ctx_{args.num_sampled_code_cell}.json", "wt"),
    )

    # 슬라이딩 윈도우기반 컨텍스트 생성
    train_sliding_window_pairs = build_sliding_window_pairs(df_train, args.window_size)
    json.dump(
        train_sliding_window_pairs,
        open(
            f"{args.root_path}/train_sliding_window_{args.window_size}_pairs.json", "wt"
        ),
    )

    valid_sliding_window_pairs = build_sliding_window_pairs(df_valid, args.window_size)
    json.dump(
        valid_sliding_window_pairs,
        open(
            f"{args.root_path}/valid_sliding_window_{args.window_size}_pairs.json", "wt"
        ),
    )
