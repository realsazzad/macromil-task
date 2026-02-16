import os
import numpy as np
import evaluate
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

MODEL_NAME = "distilbert-base-uncased"  # you can switch to a sentiment-pretrained one if you want
MAX_LENGTH = 256  # 256 is usually enough for IMDB; try 512 if you want (slower)
SEED = 42
DATA_PATH = "data/imdb_all.csv"


def train_model(df_train, df_test):
    set_seed(SEED)

    train_dataset = Dataset.from_pandas(df_train)
    test_dataset = Dataset.from_pandas(df_test)

    ds = DatasetDict({"train": train_dataset, "test": test_dataset})

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def tokenize_batch(batch):
        return tokenizer(
            batch["review"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = ds.map(tokenize_batch, batched=True, remove_columns=["review"])
    tokenized.set_format("torch")

    # 3) Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )

    # 4) Collator (dynamic padding)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5) Metrics
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"],
        }

    # 6) Training args
    # Tip: if you have a GPU, fp16=True speeds things up. If not, set fp16=False.
    args = TrainingArguments(
        output_dir="distilbert-imdb",
        run_name="distilbert-imdb",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        num_train_epochs=2,                  # 2-3 epochs is typical
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=16,      # bump to 32 if VRAM allows
        per_device_eval_batch_size=32,
        warmup_ratio=0.06,
        logging_steps=100,
        fp16=True,                           # set False if training on CPU
        seed=SEED,
        report_to="none",
        save_total_limit=2,
    )

    # 7) Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 8) Train
    trainer.train()

    # 9) Final eval on test
    metrics = trainer.evaluate()
    print("Final test metrics:", metrics)

    # 10) Save best model + tokenizer
    trainer.save_model("distilbert-imdb-best")
    tokenizer.save_pretrained("distilbert-imdb-best")
    print("Saved to: distilbert-imdb-best")


