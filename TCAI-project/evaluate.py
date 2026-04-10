import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_SYSTEM_PROMPT = (
    "You are a professional turtle care assistant. Provide a clear, accurate, "
    "and practical answer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TCAI and baseline models on the same dataset."
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default="data/turtle_dataset_finetune_format.json",
        help="Evaluation dataset JSON path.",
    )
    parser.add_argument(
        "--models_config",
        type=str,
        default="eval_models.example.json",
        help="JSON file describing the models to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluations/latest",
        help="Directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for model loading.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum generation tokens per answer.",
    )
    parser.add_argument(
        "--include_bertscore",
        action="store_true",
        help="Compute BERTScore if bert-score is installed.",
    )
    parser.add_argument(
        "--include_semantic_similarity",
        action="store_true",
        help="Compute embedding cosine similarity if sentence-transformers is installed.",
    )
    parser.add_argument(
        "--semantic_model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model used for semantic similarity.",
    )
    parser.add_argument(
        "--bertscore_model",
        type=str,
        default="bert-base-chinese",
        help="Model name used when computing BERTScore.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally evaluate only the first N examples for quick checks.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_text(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    # Preserve English words/numbers while splitting CJK into readable units.
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text)


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0

    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index_b, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[index_b - 1] + 1)
            else:
                current.append(max(previous[index_b], current[-1]))
        previous = current
    return previous[-1]


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_text(prediction)
    ref_tokens = tokenize_text(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))

    precision = safe_divide(overlap, len(pred_tokens))
    recall = safe_divide(overlap, len(ref_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_text(prediction)
    ref_tokens = tokenize_text(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = safe_divide(lcs, len(pred_tokens))
    recall = safe_divide(lcs, len(ref_tokens))
    if precision + recall == 0:
        return 0.0
    beta = 1.2
    return ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def format_prompt(instruction: str, input_text: str) -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()

    if input_text:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
        )

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


@dataclass
class Example:
    instruction: str
    input_text: str
    reference: str


def load_examples(path: Path) -> list[Example]:
    payload = load_json(path)
    examples: list[Example] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise TypeError(f"Example {index} must be an object.")
        examples.append(
            Example(
                instruction=str(row.get("instruction", "")),
                input_text=str(row.get("input", "")),
                reference=str(row.get("output", "")),
            )
        )
    if not examples:
        raise ValueError("Evaluation dataset is empty.")
    return examples


class BaseRunner:
    def answer(self, instruction: str, input_text: str) -> str:
        raise NotImplementedError


class MockRunner(BaseRunner):
    def answer(self, instruction: str, input_text: str) -> str:
        from app.services.inference import _mock_answer

        question = input_text.strip() or instruction.strip()
        return _mock_answer(question)


class ApiRunner(BaseRunner):
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def answer(self, instruction: str, input_text: str) -> str:
        question = input_text.strip() or instruction.strip()
        payload = json.dumps({"question": question}).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/ask",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc
        return str(data["answer"])


class UnslothRunner(BaseRunner):
    def __init__(
        self,
        model_name: str,
        max_seq_length: int,
        max_new_tokens: int,
        system_prompt: str,
        load_in_4bit: bool = True,
        load_in_16bit: bool = False,
    ) -> None:
        import torch
        from unsloth import FastLanguageModel

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            load_in_16bit=load_in_16bit,
        )
        FastLanguageModel.for_inference(self.model)

    def answer(self, instruction: str, input_text: str) -> str:
        question = input_text.strip() or instruction.strip()
        prompt = format_prompt(self.system_prompt, question)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.split("### Response:\n", 1)[-1].strip()


class HfRunner(BaseRunner):
    def __init__(
        self,
        model_name: str,
        max_seq_length: int,
        max_new_tokens: int,
        system_prompt: str,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.max_seq_length = max_seq_length

    def answer(self, instruction: str, input_text: str) -> str:
        question = input_text.strip() or instruction.strip()
        prompt = format_prompt(self.system_prompt, question)
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.split("### Response:\n", 1)[-1].strip()


def build_runner(config: dict[str, Any], args: argparse.Namespace) -> BaseRunner:
    backend = config["backend"]
    if backend == "mock":
        return MockRunner()
    if backend == "api":
        return ApiRunner(base_url=config["base_url"])
    if backend == "lora":
        return UnslothRunner(
            model_name=config["adapter_path"],
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
            system_prompt=config.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            load_in_4bit=config.get("load_in_4bit", True),
            load_in_16bit=config.get("load_in_16bit", False),
        )
    if backend == "unsloth":
        return UnslothRunner(
            model_name=config["model_name"],
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
            system_prompt=config.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            load_in_4bit=config.get("load_in_4bit", True),
            load_in_16bit=config.get("load_in_16bit", False),
        )
    if backend == "hf":
        return HfRunner(
            model_name=config["model_name"],
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
            system_prompt=config.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        )
    raise ValueError(f"Unsupported backend: {backend}")


def compute_bertscore(
    predictions: list[str], references: list[str], model_name: str
) -> list[float] | None:
    try:
        from bert_score import score
    except ImportError:
        return None

    _, _, f1 = score(
        predictions,
        references,
        lang="zh",
        model_type=model_name,
        verbose=False,
    )
    return [float(value) for value in f1]


def compute_semantic_similarity(
    predictions: list[str], references: list[str], model_name: str
) -> list[float] | None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    model = SentenceTransformer(model_name)
    pred_embeddings = model.encode(predictions, normalize_embeddings=True)
    ref_embeddings = model.encode(references, normalize_embeddings=True)

    scores: list[float] = []
    for pred_embedding, ref_embedding in zip(pred_embeddings, ref_embeddings):
        cosine = float(sum(float(a) * float(b) for a, b in zip(pred_embedding, ref_embedding)))
        scores.append(cosine)
    return scores


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "f1_score_percent",
        "semantic_similarity_percent",
        "bertscore_percent",
        "rouge_l_percent",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| Model | F1-score(%) | Semantic Similarity(%) | BERTScore(%) | ROUGE-L(%) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['f1_score_percent']} | "
            f"{row['semantic_similarity_percent']} | {row['bertscore_percent']} | "
            f"{row['rouge_l_percent']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def percent_or_na(values: list[float] | None) -> str:
    if values is None:
        return "N/A"
    return f"{mean(values) * 100:.2f}"


def main() -> None:
    args = parse_args()
    eval_data_path = Path(args.eval_data)
    models_config_path = Path(args.models_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(eval_data_path)
    if args.limit is not None:
        examples = examples[: args.limit]
    model_configs = load_json(models_config_path)
    if not isinstance(model_configs, list) or not model_configs:
        raise ValueError("models_config must be a non-empty JSON array.")

    summary_rows: list[dict[str, Any]] = []
    references = [example.reference for example in examples]

    for config in model_configs:
        name = config["name"]
        print(f"Evaluating model: {name}")
        runner = build_runner(config, args)
        predictions: list[str] = []
        records: list[dict[str, Any]] = []

        for index, example in enumerate(examples, start=1):
            prediction = runner.answer(example.instruction, example.input_text)
            predictions.append(prediction)
            records.append(
                {
                    "index": index,
                    "instruction": example.instruction,
                    "input": example.input_text,
                    "reference": example.reference,
                    "prediction": prediction,
                }
            )

        f1_scores = [token_f1(pred, ref) for pred, ref in zip(predictions, references)]
        rouge_scores = [rouge_l(pred, ref) for pred, ref in zip(predictions, references)]

        bert_scores = None
        if args.include_bertscore:
            bert_scores = compute_bertscore(
                predictions, references, model_name=args.bertscore_model
            )

        semantic_scores = None
        if args.include_semantic_similarity:
            semantic_scores = compute_semantic_similarity(
                predictions, references, model_name=args.semantic_model
            )

        save_json(output_dir / f"{name}_predictions.json", records)

        row = {
            "model": name,
            "f1_score_percent": f"{mean(f1_scores) * 100:.2f}",
            "semantic_similarity_percent": percent_or_na(semantic_scores),
            "bertscore_percent": percent_or_na(bert_scores),
            "rouge_l_percent": f"{mean(rouge_scores) * 100:.2f}",
        }
        summary_rows.append(row)

    save_json(output_dir / "summary.json", summary_rows)
    write_summary_csv(output_dir / "summary.csv", summary_rows)
    write_summary_markdown(output_dir / "summary.md", summary_rows)

    print(f"Evaluation finished. Summary written to: {output_dir}")


if __name__ == "__main__":
    main()
