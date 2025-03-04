import logging
import os
from typing import Dict, Tuple
from datasets import Dataset, load_dataset

def load_split_data(
    langs_dict: Dict[str, int], 
    test_batch_size: int
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load and split the dataset into train, validation, and test sets.
    """
    ds = load_dataset("hac541309/open-lid-dataset", split="train")
    splits = ds.train_test_split(test_size=0.1, seed=41)
    train, test = splits["train"], splits["test"]
    test = test.train_test_split(test_size=len(langs_dict) * test_batch_size, seed=41)
    valid, test = test["test"], test["train"]

    test_save_path = "data/test_dataset"
    if not os.path.exists(test_save_path):
        logging.info("Saving Test Set...")
        test.save_to_disk(test_save_path)
        logging.info("Test Set Saved.")

    # Rename columns for consistency
    train = train.rename_column("lang", "label")
    test = test.rename_column("lang", "label")
    valid = valid.rename_column("lang", "label")

    return train, valid, test
