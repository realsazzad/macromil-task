import pandas as pd
from datasets import load_dataset, DatasetDict
import os

SEED = 42
DATA_PATH = "data"
LABEL_MAP = {"negative": 0, "positive": 1}


def load_and_prepare_data():
    """Load the IMDB dataset, convert to pandas and subsample for training/testing.
    Returns
    -------
    train_df : pd.DataFrame
        A dataframe with columns ``review`` and ``sentiment`` for training.
    test_df : pd.DataFrame
        A dataframe with columns ``review`` and ``sentiment`` for evaluation.
    """
    print("Loading the IMDB dataset...")
    ds_all = load_dataset("csv", data_files= os.path.join(DATA_PATH, "imdb_all.csv"))["train"]
    ds_all = ds_all.map(lambda x: { "label": LABEL_MAP[x["sentiment"].lower()]})
    ds_all = ds_all.remove_columns("sentiment")

    # Split into train/test like original IMDB
    split = ds_all.train_test_split(test_size=0.2, seed=SEED)

    ds = DatasetDict({"train": split["train"], "test": split["test"]})
    save_path = os.path.join(DATA_PATH, "train_df.csv")
    train_df = pd.DataFrame(ds['train'])
    train_df.to_csv(save_path, index=False)

    save_path = os.path.join(DATA_PATH, "test_df.csv")
    test_df = pd.DataFrame(ds['test'])
    test_df.to_csv(save_path, index=False)

    return train_df, test_df