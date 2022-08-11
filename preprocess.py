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
parser.add_argument("--num-sampled-code-cell", type=int, default=40)
parser.add_argument("--window-size", type=int, default=30)
parser.add_argument("--random-context-sample", action="store_true")
parser.add_argument("--insert-cell-order", action="store_true")
parser.add_argument("--skip-create-from-scratch", action="store_true")
parser.add_argument("--memo", type=str, default="")


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
    """
    :param cell: 셀 스트링

    .. note::
        줄바꿈 교정, 6개 초과 # 제거, URL 제거, 연속 공백 1개로 통일
    """
    for char in ["\r\n", "\r", "\n"]:
        cell = cell.replace(char, " ")
    cell = re.sub(r"\s{1,}", " ", cell)
    cell = re.sub(r"#{6,}", "", cell)
    cell = re.sub(r"-{3,}", "", cell)
    cell = re.sub(r"http\S+[^)]", "", cell)
    cell = re.sub(r"<[^>]+>", "", cell)
    cell = re.sub(r"\$\$.*?\$\$", "", cell)
    cell = re.sub(r"\$.*?\$", "", cell)
    cell = re.sub(r"\.*\$", "", cell)
    cell = re.sub(r'\([^)]*?\)', '()', cell)
    cell = re.sub(r'\{[^)]*?\}', '{}', cell)
    cell = re.sub(r'\[[^)]*?\]', '[]', cell)
    cell = "\n".join([sent.strip() for sent in cell.split("\n")])
    return cell


def summary_code_cell(cell: str):
    """
    코드셀을 중요 정보 순서로 요약합니다.
    indentation 이 적을수록 중요한 코드라고 판단합니다.

    :param cell: 코드셀 스트링
    :return : 중요 정보 순서로 요약된 코드셀
    """
    cell = re.sub(r"\n{1,}", "\n", cell)
    cell = re.sub(r"\s+\n", "", cell)
    items = cell.split("\n")
    items.sort(key=lambda x: len(x) - len(x.lstrip()))
    cell = "\n".join(items)
    return cell


def insert_order_to_cell_str(pct: float, cell: str):
    return f"#{round(pct, 2)} {cell}"


def sample_cells(
    cells, n_samples, from_last=False, random_choice=False, insert_cell_order=False
):
    """
    :param cells: 코드셀 스트링 리스트
    :param n_samples: 샘플링 횟수
    :param from_last: 끝을 기준으로 샘플링 (추론시에만 활용)
    :param random_choice: 각 서브 윈도우에서 유니폼 샘플링
    :param insert_cell_order: cell의 순서 정보를 cell str에 주입 (#0.42 cell string...)

    .. note::
        n_samples에 대한 버그가 존재함 len(cells) == 59, n_samples == 30 인 경우
    """
    if len(cells) == 0:
        return []

    cells = [summary_code_cell(cell) for cell in cells]
    if insert_cell_order:
        cells = [
            insert_order_to_cell_str(idx / len(cells), cell)
            for idx, cell in enumerate(cells)
        ]

    if len(cells) <= n_samples:
        return cells

    if from_last:
        cells = cells[::-1]

    sampled_cells = []
    if random_choice:
        choice_prob = n_samples / len(cells)
        for cell in cells:
            if random.random() < choice_prob:
                sampled_cells.append(cell)
    else:
        step = len(cells) / n_samples
        idx = 0
        while idx < len(cells):
            sampled_cells.append(cells[idx])
            idx += step
            idx = int(np.round(idx))

    # 양 끝에 대한 보정
    if cells[0] not in sampled_cells:
        sampled_cells[0] = cells[0]
    if cells[-1] not in sampled_cells:
        sampled_cells[-1] = cells[-1]

    if from_last:
        return sampled_cells[::-1]
    return sampled_cells


def build_context_dict(
    df,
    num_sampled_code_cell=30,
    make_sample_from_last=False,
    make_sample_randomly=False,
    insert_cell_order=False,
):
    features = dict()
    df = df.sort_values("rank").reset_index(drop=True)
    for idx, sub_df in tqdm(df.groupby("id")):
        key = str(idx)
        features[key] = dict()
        total_md = sub_df[sub_df.cell_type == "markdown"].shape[0]
        code_sub_df = sub_df[sub_df.cell_type == "code"]
        total_code = code_sub_df.shape[0]
        codes = sample_cells(
            code_sub_df.source.values.tolist(),
            num_sampled_code_cell,
            from_last=make_sample_from_last,
            random_choice=make_sample_randomly,
            insert_cell_order=insert_cell_order,
        )
        features[key]["total_code"] = total_code
        features[key]["total_md"] = total_md
        features[key]["codes"] = codes
    return features


def make_md_code_pairs_by_sliding_window(
    md_sub_df, code_sub_df, window_size, mode="train"
):
    """
    :return: Tuple(n_id-md_cell_id, 윈도우 인덱스, 전체 윈도우 개수, 마크다운 셀, 코드셀 윈도우, 랭크 퍼센타일)
    """
    pairs = []
    md_cells = [cell for cell in md_sub_df.source.values]
    n_md_ids = [
        f"{a}-{b}" for a, b in zip(md_sub_df.id.values, md_sub_df.cell_id.values)
    ]
    pct_ranks = md_sub_df.pct_rank.values
    code_cells = [cell for cell in code_sub_df.source.values]

    if window_size >= len(code_cells):
        window = code_cells
        for n_md_id, md_cell, pct_rank in zip(n_md_ids, md_cells, pct_ranks):
            pairs.append((n_md_id, float(0), float(1), md_cell, window, pct_rank))
    else:
        n_windows = math.ceil(len(code_cells) / window_size)
        for w_idx in range(n_windows):
            offset = w_idx * window_size
            window = code_cells[offset : offset + window_size]
            range_start = w_idx / n_windows
            range_end = (w_idx + 1) / n_windows
            for n_md_id, md_cell, pct_rank in zip(n_md_ids, md_cells, pct_ranks):
                _pct_rank = -1.0
                if mode == "train" and range_start <= pct_rank < range_end:
                    _pct_rank = pct_rank
                pairs.append(
                    (
                        n_md_id,
                        float(w_idx),
                        float(n_windows),
                        md_cell,
                        window,
                        _pct_rank,
                    )
                )
    return pairs


def build_sliding_window_pairs(df, window_size=30, mode="train"):
    df = df.sort_values("rank").reset_index(drop=True)
    pairs = []
    for idx, sub_df in tqdm(df.groupby("id")):
        md_sub_df = sub_df[sub_df.cell_type == "markdown"]
        code_sub_df = sub_df[sub_df.cell_type == "code"]
        sub_pairs = make_md_code_pairs_by_sliding_window(
            md_sub_df,
            code_sub_df,
            window_size,
            mode,
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

    df_train = None
    df_valid = None
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

        # external 데이터 추가 버전 (odins0n/ai4code-custom-data)
        df_extra_train = pd.read_csv("./data/external_data.csv")
        df_extra_train = df_extra_train.dropna()
        df_extra_train["cell_id"] = df_extra_train["rank"].astype(str)
        df_extra_train["id"] = df_extra_train["notebook_id"].astype(str)
        df_extra_train = df_extra_train[
            ["id", "cell_id", "cell_type", "source", "rank", "pct_rank"]
        ]

        # code만 있거나, md만 있는 경우 체크
        del_n_ids = []
        for idx, sub_df in tqdm(df_extra_train.groupby("id")):
            n_md = sub_df[sub_df.cell_type == "markdown"].shape[0]
            n_code = sub_df[sub_df.cell_type == "code"].shape[0]
            if n_md == 0 or n_code == 0:
                del_n_ids.append(sub_df.id.values[0])

        # code만 있거나, md만 있는 경우 제거 (23분 소요)
        for n_id in tqdm(del_n_ids):
            df_extra_train = df_extra_train[df_extra_train["id"] != n_id]

        df_concat_train = pd.concat([df_train, df_extra_train])
        df_concat_train_md = (
            df_concat_train[df_concat_train["cell_type"] == "markdown"]
            .dropna(subset=["source", "rank"])
            .reset_index(drop=True)
        )
        df_concat_train.to_csv(f"{args.root_path}/concat_train.csv", index=False)
        df_concat_train_md.to_csv(f"{args.root_path}/concat_train_md.csv", index=False)

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
        df_concat_train = pd.read_csv(f"{args.root_path}/concat_train.csv").dropna(
            subset=["source", "rank"]
        )
        df_concat_train.to_csv(f"{args.root_path}/concat_train.csv", index=False)
        df_concat_train_md = pd.read_csv(
            f"{args.root_path}/concat_train_md.csv"
        ).dropna(subset=["source", "rank"])
        df_concat_train_md.to_csv(f"{args.root_path}/concat_train_md.csv", index=False)

    if df_train is None or df_valid is None:
        df_train = pd.read_csv(f"{args.root_path}/train.csv")
        df_valid = pd.read_csv(f"{args.root_path}/valid.csv")

    make_sample_randomly = False
    random_or_not = ""
    if args.random_context_sample:
        make_sample_randomly = True
        random_or_not = "_random"

    # train 컨텍스트 추출
    train_context_dict = build_context_dict(
        df_train,
        args.num_sampled_code_cell,
        make_sample_randomly=make_sample_randomly,
        insert_cell_order=args.insert_cell_order,
    )
    json.dump(
        train_context_dict,
        open(
            f"{args.root_path}/train_{args.memo}{random_or_not}_ctx_{args.num_sampled_code_cell}.json",
            "wt",
        ),
    )

    # validation 컨텍스트 추출
    valid_context_dict = build_context_dict(
        df_valid,
        args.num_sampled_code_cell,
        make_sample_randomly=make_sample_randomly,
        insert_cell_order=args.insert_cell_order,
    )
    json.dump(
        valid_context_dict,
        open(
            f"{args.root_path}/valid_{args.memo}{random_or_not}_ctx_{args.num_sampled_code_cell}.json",
            "wt",
        ),
    )

    # extra 추가 버전 컨텍스트 추출
    df_concat_train = pd.read_csv(f"{args.root_path}/concat_train.csv")
    concat_train_context_dict = build_context_dict(
        df_concat_train,
        args.num_sampled_code_cell,
        make_sample_randomly=make_sample_randomly,
        insert_cell_order=args.insert_cell_order,
    )
    json.dump(
        concat_train_context_dict,
        open(
            f"{args.root_path}/concat_train_{args.memo}{random_or_not}_ctx_{args.num_sampled_code_cell}.json",
            "wt",
        ),
    )

    # # 슬라이딩 윈도우기반 컨텍스트 생성
    # train_sliding_window_pairs = build_sliding_window_pairs(df_train, args.window_size)
    # json.dump(
    #     train_sliding_window_pairs,
    #     open(
    #         f"{args.root_path}/train_sliding_window_{args.window_size}_pairs.json", "wt"
    #     ),
    # )

    # valid_sliding_window_pairs = build_sliding_window_pairs(df_valid, args.window_size)
    # json.dump(
    #     valid_sliding_window_pairs,
    #     open(
    #         f"{args.root_path}/valid_sliding_window_{args.window_size}_pairs.json", "wt"
    #     ),
    # )
