import json
from glob import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm


def read_notebook(path):
    id = path.split("/")[-1][: -len(".json")]
    return (
        pd.read_json(path, dtype={"cell_type": "category", "source": "str"})
        .assign(id=id)
        .rename_axis("cell_id")
    )


if __name__ == "__main__":
    train_paths = list(glob("./data/train/*.json"))
    train_notebooks = [
        read_notebook(path) for path in tqdm(train_paths, desc="Read Train Notebooks")
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
        "./data/train_orders.csv", index_col="id", squeeze=True  # Series로 리턴 됨
    ).str.split()  # string 표현을 리스트로 스플릿

    #: 노트북 아이디별 cell_order(셀 정답 순서), cell_id(주어진 셀 아이디 순서)
    merged_df_orders = df_orders.to_frame().join(
        df_all.reset_index("cell_id").groupby("id")["cell_id"].apply(list),
        how="right",
    )

    def get_ranks(gt, derived):
        return [gt.index(d) for d in derived]

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

    df_ancestors = pd.read_csv("./data/train_ancestors.csv", index_col="id")
    df_all = (
        df_all.reset_index()
        .merge(df_ranks, on=["id", "cell_id"])
        .merge(df_ancestors, on=["id"])
    )
    df_all["pct_rank"] = df_all["rank"] / df_all.groupby("id")["cell_id"].transform(
        "count"
    )

    # validation split
    valid_split = 0.1

    splitter = GroupShuffleSplit(n_splits=1, test_size=valid_split, random_state=42)
    train_ind, val_ind = next(splitter.split(df_all, groups=df_all["ancestor_id"]))
    df_train = df_all.loc[train_ind].reset_index(drop=True)
    df_valid = df_all.loc[val_ind].reset_index(drop=True)

    df_train_md = df_train[df_train["cell_type"] == "markdown"].reset_index(drop=True)
    df_valid_md = df_valid[df_valid["cell_type"] == "markdown"].reset_index(drop=True)
    df_train_md.to_csv("./data/train_md.csv", index=False)
    df_valid_md.to_csv("./data/valid_md.csv", index=False)
    df_train.to_csv("./data/train.csv", index=False)
    df_valid.to_csv("./data/valid.csv", index=False)

    def clean_code(cell):
        return str(cell).replace("\\n", "\n")

    def sample_cells(cells, n):
        cells = [clean_code(cell) for cell in cells]
        if n >= len(cells):
            return [cell[:200] for cell in cells]
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

    def get_features(df):
        """
        .. todo::
            더 똑똑하게 추출하기
        """
        features = dict()
        df = df.sort_values("rank").reset_index(drop=True)
        for idx, sub_df in tqdm(df.groupby("id")):
            features[idx] = dict()
            total_md = sub_df[sub_df.cell_type == "markdown"].shape[0]
            code_sub_df = sub_df[sub_df.cell_type == "code"]
            total_code = code_sub_df.shape[0]
            codes = sample_cells(code_sub_df.source.values, 20)  # 코드셀만 모은 후 거기서 샘플 뽑음
            features[idx]["total_code"] = total_code
            features[idx]["total_md"] = total_md
            features[idx]["codes"] = codes
        return features

    train_feature_transformed_samples = get_features(df_train)
    json.dump(train_feature_transformed_samples, open("./data/train_fts.json", "wt"))

    valid_feature_transformed_samples = get_features(df_valid)
    json.dump(valid_feature_transformed_samples, open("./data/valid_fts.json", "wt"))
