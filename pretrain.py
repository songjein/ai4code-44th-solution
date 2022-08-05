import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling,
                          LineByLineTextDataset, Trainer, TrainingArguments)

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
                    samples.append([md_source[:256], code_source[:256]])
    return samples


if __name__ == "__main__":

    make_dataset = False
    corpus_path = "./data/text.txt"
    model_name = "prajjwal1/bert-small"
    max_seq_len = 128
    output_path = f"./pretrained_{max_seq_len}_{model_name}"
    batch_size = 512
    epochs = 10

    if make_dataset:
        df = pd.read_csv("./data/concat_train.csv")
        df.source = df.source.apply(clean_code)
        samples = generate_md_code_pairs(df)
        with open("./data/text.txt", "w", encoding="utf-8") as f:
            for md, code in samples:
                sentence = f"{md}</s>{code}"
                f.write(sentence + "\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer, file_path=corpus_path, block_size=max_seq_len
    )

    os.makedirs(output_path, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        gradient_accumulation_steps=8,
        fp16=True,
        per_device_train_batch_size=batch_size,
        save_steps=1000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(output_path)
