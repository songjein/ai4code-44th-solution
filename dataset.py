import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class MarkdownDataset(Dataset):
    def __init__(self, df, model_name_or_path, total_max_len, md_max_len, fts):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.fts = fts

    def __getitem__(self, index):
        row = self.df.iloc[index]

        inputs = self.tokenizer.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            max_length=self.md_max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        code_inputs = self.tokenizer.batch_encode_plus(
            [str(x) for x in self.fts[row.id]["codes"]],
            add_special_tokens=True,
            max_length=22,  # (512-64)//20
            padding="max_length",
            truncation=True,
        )

        ids = inputs["input_ids"]
        for x in code_inputs["input_ids"]:
            ids.extend(x[:-1])  # TODO: </s>를 제거하려고 넣은 건데, 앞에 <s>는 안 없애나?
        ids = ids[: self.total_max_len]
        if len(ids) != self.total_max_len:
            ids = ids + [self.tokenizer.pad_token_id] * (self.total_max_len - len(ids))
        ids = torch.LongTensor(ids)

        mask = inputs["attention_mask"]
        for x in code_inputs["attention_mask"]:
            mask.extend(x[:-1])
        mask = mask[: self.total_max_len]
        if len(mask) != self.total_max_len:
            mask = mask + [self.tokenizer.pad_token_id] * (
                self.total_max_len - len(mask)
            )
        mask = torch.LongTensor(mask)

        assert len(ids) == self.total_max_len

        return ids, mask, torch.FloatTensor([row.pct_rank])

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
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )

        code_inputs = self.tokenizer.encode_plus(
            self.id2src[code_cell_id],
            None,
            add_special_tokens=False,
            max_length=self.code_max_len - 2,  # special token
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )

        cls_token = 0
        sep_token = 2
        ids = (
            [cls_token]
            + md_inputs["input_ids"]
            + [sep_token, sep_token]
            + code_inputs["input_ids"]
            + [sep_token]
        )
        ids = ids[: self.total_max_len]
        mask = [1] * len(ids)
        assert len(ids) == self.total_max_len

        ids = torch.LongTensor(ids)
        mask = torch.LongTensor(mask)
        label = torch.FloatTensor([label])
        return ids, mask, label

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":

    import pandas as pd

    from preprocess import generate_pairs_with_label

    df = pd.read_csv("./data/valid.csv")
    samples = generate_pairs_with_label(df)
    dataset = PairwiseDataset(samples, df, "microsoft/codebert-base")

    print(dataset[0])
