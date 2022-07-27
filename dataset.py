import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PointwiseDataset(Dataset):
    def __init__(
        self, df, model_name_or_path, total_max_len, md_max_len, code_max_len, ctx
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.md_max_len = md_max_len
        self.code_max_len = code_max_len
        self.total_max_len = total_max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.ctx = ctx

    def __getitem__(self, index):
        row = self.df.iloc[index]
        inputs = self.tokenizer.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            max_length=self.md_max_len,
            return_token_type_ids=True,
            truncation=True,
        )

        # 개별 코드 셀 토큰의 최대 길이를 동적으로 결정
        num_codes = len(self.ctx[row.id]["codes"])
        code_max_len_ = (self.total_max_len - self.md_max_len) // num_codes

        code_inputs = self.tokenizer.batch_encode_plus(
            [str(x) for x in self.ctx[row.id]["codes"]],
            add_special_tokens=False,
            max_length=code_max_len_,
            truncation=True,
        )

        sep_token_id = self.tokenizer.sep_token_id
        pad_token_id = self.tokenizer.pad_token_id

        ids = inputs["input_ids"]
        for x in code_inputs["input_ids"]:
            ids.extend([sep_token_id] + x)
        ids = ids[: self.total_max_len - 1] + [sep_token_id]
        mask = [1] * len(ids)
        if len(ids) != self.total_max_len:
            ids = ids + [pad_token_id] * (self.total_max_len - len(ids))
            mask = mask + [pad_token_id] * (self.total_max_len - len(mask))
        assert len(ids) == self.total_max_len

        ids = torch.LongTensor(ids)
        mask = torch.LongTensor(mask)
        label = torch.FloatTensor([row.pct_rank])

        return ids, mask, label

    def __len__(self):
        return self.df.shape[0]


class PairwiseDataset(Dataset):
    def __init__(
        self,
        samples,
        df,
        model_name_or_path,
        total_max_len=128,
        md_max_len=64,
    ):
        super().__init__()
        self.samples = samples
        self.id2src = dict(zip(df["cell_id"].values, df["source"].values))
        self.total_max_len = total_max_len
        self.md_max_len = md_max_len
        self.code_max_len = total_max_len - md_max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def __getitem__(self, index):
        md_cell_id, code_cell_id, label = self.samples[index]

        md_inputs = self.tokenizer.encode_plus(
            self.id2src[md_cell_id],
            None,
            add_special_tokens=False,
            max_length=self.md_max_len - 2,  # special token
            return_token_type_ids=True,
            truncation=True,
        )
        code_inputs = self.tokenizer.encode_plus(
            self.id2src[code_cell_id],
            None,
            add_special_tokens=False,
            max_length=self.code_max_len - 2,  # special token
            return_token_type_ids=True,
            truncation=True,
        )

        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        pad_token_id = self.tokenizer.pad_token_id

        ids = (
            [cls_token_id]
            + md_inputs["input_ids"]
            + [sep_token_id, sep_token_id]
            + code_inputs["input_ids"]
            + [sep_token_id]
        )
        ids = ids[: self.total_max_len]
        mask = [1] * len(ids)
        if len(ids) != self.total_max_len:
            ids = ids + [pad_token_id] * (self.total_max_len - len(ids))
            mask = mask + [pad_token_id] * (self.total_max_len - len(mask))
        assert len(ids) == self.total_max_len

        ids = torch.LongTensor(ids)
        mask = torch.LongTensor(mask)
        label = torch.FloatTensor([label])
        return ids, mask, label

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    import json

    import pandas as pd

    from train import generate_pairs_with_label

    if False:
        df = pd.read_csv("./data/valid.csv")
        samples = generate_pairs_with_label(df)
        dataset = PairwiseDataset(samples, df, "microsoft/codebert-base")

    if True:
        df_valid_md = (
            pd.read_csv("./data/valid_md.csv")
            .drop("parent_id", axis=1)
            .dropna()
            .reset_index(drop=True)
        )
        valid_ctx = json.load(open("./data/valid_ctx_30.json"))
        dataset = PointwiseDataset(
            df_valid_md,
            model_name_or_path="microsoft/codebert-base",
            md_max_len=64,
            code_max_len=15,
            total_max_len=512,
            ctx=valid_ctx,
        )

    for idx, data in enumerate(dataset):
        print(data)
        if idx > 100:
            break
