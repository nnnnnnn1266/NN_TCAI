import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a finetuning JSON dataset into train and test sets."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/turtle_dataset_finetune_format.json",
        help="Source dataset JSON path.",
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default="data/turtle_dataset_train.json",
        help="Output path for the train split.",
    )
    parser.add_argument(
        "--test_output",
        type=str,
        default="data/turtle_dataset_test.json",
        help="Output path for the test split.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Fraction of examples to place in the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic splitting.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Input dataset must be a non-empty JSON array.")
    return payload


def save_dataset(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    train_output = Path(args.train_output)
    test_output = Path(args.test_output)

    if not 0 < args.test_ratio < 1:
        raise ValueError("--test_ratio must be between 0 and 1.")
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    dataset = load_dataset(input_path)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)

    test_size = max(1, int(round(len(dataset) * args.test_ratio)))
    test_indices = set(indices[:test_size])

    train_split = [row for index, row in enumerate(dataset) if index not in test_indices]
    test_split = [row for index, row in enumerate(dataset) if index in test_indices]

    if not train_split or not test_split:
        raise RuntimeError("Split produced an empty train or test set.")

    save_dataset(train_output, train_split)
    save_dataset(test_output, test_split)

    print(f"Input examples : {len(dataset)}")
    print(f"Train examples : {len(train_split)}")
    print(f"Test examples  : {len(test_split)}")
    print(f"Seed           : {args.seed}")
    print(f"Train output   : {train_output}")
    print(f"Test output    : {test_output}")


if __name__ == "__main__":
    main()
