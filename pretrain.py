import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocess import clean_code


def generate_md_code_pairs(df):
    samples = []
    for id, df_sub in tqdm(df.groupby("id")):
        df_sub_md = df_sub[df_sub["cell_type"] == "markdown"]
        df_sub_code = df_sub[df_sub["cell_type"] == "code"]
        df_sub_code_rank = df_sub_code["rank"].values
        df_sub_code_source = df_sub_code["source"].values
        for md_cell_id, md_rank, md_source in df_sub_md[
            ["cell_id", "rank", "source"]
        ].values:
            labels = np.array(
                [((md_rank + 1) == code_rank) for code_rank in df_sub_code_rank]
            ).astype("int")
            for code_source, label in zip(df_sub_code_source, labels):
                if label == 1:
                    samples.append([md_source, code_source])
    return samples


if __name__ == "__main__":

    df = pd.read_csv("./data/concat_train.csv")
    df.source = df.source.apply(clean_code)
    samples = generate_md_code_pairs(df)

    with open("./data/text.txt", "w", encoding="utf-8") as f:
        for md, code in samples:
            sentence = f"{md}</s>{code}"
            f.write(sentence + "\n")
