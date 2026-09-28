import json
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict

argparser = argparse.ArgumentParser()
argparser.add_argument("-d", "--dataset-dir", type=Path)
argparser.add_argument("-hf", type=str)

if __name__ == "__main__":
    args = argparser.parse_args()

    # train ds
    train_rows = []
    for ds_path in args.dataset_dir.glob("*.train.jsonl"):
        rows = [json.loads(line) for line in open(ds_path)]
        train_rows.extend(rows)
    train_ds = Dataset.from_list(train_rows)

    valid_rows = []
    for ds_path in args.dataset_dir.glob("*.valid.jsonl"):
        rows = [json.loads(line) for line in open(ds_path)]
        valid_rows.extend(rows)
    valid_ds = Dataset.from_list(valid_rows)

    test_rows = []
    for ds_path in args.dataset_dir.glob("*.test.jsonl"):
        rows = [json.loads(line) for line in open(ds_path)]
        test_rows.extend(rows)
    test_ds = Dataset.from_list(test_rows)


    ds = DatasetDict({
        "train": train_ds,
        "valid": valid_ds,
        "test": test_ds
    })

    ds.push_to_hub(args.hf)
