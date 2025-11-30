import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from datasets import Dataset, load_dataset, ClassLabel, load_metric

# -------------------------
# Utilities
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# -------------------------
# Data loading helpers
# -------------------------
def load_from_hf(split_ratio=(0.8, 0.1, 0.1)):
    """
    Load cgpotts/swda from HuggingFace datasets.
    Returns: train_df, val_df, test_df (pandas DataFrames)
    """
    ds = load_dataset("cgpotts/swda")
    # many HF datasets include 'train' only; check
    if "train" in ds:
        df = pd.DataFrame(ds["train"])
    else:
        # fallback: try the first split
        first_split = list(ds.keys())[0]
        df = pd.DataFrame(ds[first_split])
    # expected columns: 'text' and 'annotation' or 'act_tag' depending on release
    # inspect and adapt
    if "text" not in df.columns:
        # Try common alternatives
        for col in ["utterance", "utterances", "dialogue"]:
            if col in df.columns:
                df = df.rename(columns={col: "text"})
                break
    # act labels are often in 'act' or 'label' or 'act_tag'
    act_col = None
    for col in ["act_tag", "act", "label", "annotation"]:
        if col in df.columns:
            act_col = col
            break
    if act_col is None:
        raise ValueError("No act/label column found in HF dataset. Columns: " + ", ".join(df.columns))
    df = df[["text", act_col]].rename(columns={act_col: "label"})
    # drop NAs
    df = df.dropna().reset_index(drop=True)
    # split
    train, temp = train_test_split(df, test_size=(1 - split_ratio[0]), random_state=42, stratify=df["label"])
    val, test = train_test_split(temp, test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]),
                                 random_state=42, stratify=temp["label"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def load_from_local_csv(csv_path, text_col="text", label_col="label", split_ratio=(0.8,0.1,0.1)):
    """
    Load from a local CSV (e.g., swda-metadata.csv exported from Potts/compprag).
    Ensure this CSV has columns for utterance text and the act label.
    """
    df = pd.read_csv(csv_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain columns: {text_col} and {label_col}. Found: {df.columns.tolist()}")
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df = df.dropna().reset_index(drop=True)
    train, temp = train_test_split(df, test_size=(1 - split_ratio[0]), random_state=42, stratify=df["label"])
    val, test = train_test_split(temp, test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]),
                                 random_state=42, stratify=temp["label"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

# -------------------------
# Preprocess labels (optional grouping)
# -------------------------
def build_label_map(labels: List[str], min_freq=20):
    """
    Build label map: keep only labels with frequency >= min_freq; rare labels -> 'OTHER'
    Returns label2id, id2label
    """
    s = pd.Series(labels)
    freq = s.value_counts()
    kept = set(freq[freq >= min_freq].index.tolist())
    new_labels = [lab if lab in kept else "OTHER" for lab in labels]
    final_labels = sorted(list(set(new_labels)))
    label2id = {lab: i for i, lab in enumerate(final_labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return new_labels, label2id, id2label

# -------------------------
# Tokenization & dataset conversion
# -------------------------
def prepare_hf_dataset(df: pd.DataFrame, tokenizer, label2id):
    texts = df["text"].astype(str).tolist()
    labels = df["label"].tolist()
    mapped = [label2id[l] for l in labels]
    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=128)
    ds = Dataset.from_dict({**enc, "labels": mapped})
    return ds

# -------------------------
# Metrics
# -------------------------
def compute_metrics_fn(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}

# -------------------------
# Main training pipeline
# -------------------------
def main(args):
    # Load data
    if args.mode == "hf":
        train_df, val_df, test_df = load_from_hf()
    else:
        train_df, val_df, test_df = load_from_local_csv(args.local_csv, text_col=args.text_col, label_col=args.label_col)

    # Optional: collapse rare labels
    all_labels = pd.concat([train_df["label"], val_df["label"], test_df["label"]]).tolist()
    new_labels, label2id, id2label = build_label_map(all_labels, min_freq=args.min_label_freq)
    # update dataframes with new labels
    mapping = {}
    for l_old, l_new in zip(all_labels, new_labels):
        # this is per-row mapping; simpler approach: rebuild using s.apply
        pass
    # rebuild label columns using build_label_map's kept set approach:
    # (recompute to be simple)
    freq = pd.Series(all_labels).value_counts()
    kept = set(freq[freq >= args.min_label_freq].index.tolist())
    def collapse_lab(x):
        return x if x in kept else "OTHER"
    train_df["label"] = train_df["label"].apply(collapse_lab)
    val_df["label"] = val_df["label"].apply(collapse_lab)
    test_df["label"] = test_df["label"].apply(collapse_lab)
    final_labels = sorted(list(set(train_df["label"].unique()).union(set(val_df["label"].unique())).union(set(test_df["label"].unique()))))
    label2id = {lab:i for i,lab in enumerate(final_labels)}
    id2label = {i:lab for lab,i in label2id.items()}

    print(f"Using {len(final_labels)} labels.")
    print(final_labels)

    # tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(final_labels),
        id2label=id2label,
        label2id=label2id,
    )

    # prepare datasets
    train_ds = prepare_hf_dataset(train_df, tokenizer, label2id)
    val_ds = prepare_hf_dataset(val_df, tokenizer, label2id)
    test_ds = prepare_hf_dataset(test_df, tokenizer, label2id)

    # training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()

    # Evaluate on test
    print("Evaluating on test set...")
    preds = trainer.predict(test_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 macro:", f1_score(y_true, y_pred, average="macro"))
    print("Classification report:\n", classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(len(id2label))]))

    # Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved model & tokenizer to", args.output_dir)

    # Optional: save label map
    import json
    with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hf", "local"], default="hf",
                        help="hf: use HuggingFace cgpotts/swda ; local: use local CSV with columns (text,label)")
    parser.add_argument("--local_csv", type=str, default="swda-metadata.csv")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./swda-da-model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--min_label_freq", type=int, default=20,
                        help="minimum frequency for a label to be kept; rarer labels mapped to 'OTHER'")
    args = parser.parse_args()
    main(args)
