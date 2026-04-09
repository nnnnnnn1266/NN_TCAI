import argparse
import json
from pathlib import Path
from typing import Any

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import Dataset, load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer


MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use Unsloth to fine-tune LLaMA 3.1 8B with LoRA.")
    parser.add_argument("--data_path", type=str, default="data/train_example.json", help="Path to training JSON.")
    parser.add_argument("--output_dir", type=str, default="outputs/llama31-lora", help="Where to save LoRA adapter.")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Base 4-bit model name.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum token length.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device training batch size.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps.")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging frequency.")
    parser.add_argument("--save_steps", type=int, default=50, help="Checkpoint save frequency.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout.")
    return parser.parse_args()


def validate_record(record: dict[str, Any], index: int) -> None:
    required_fields = ("instruction", "input", "output")
    missing = [field for field in required_fields if field not in record]
    if missing:
        raise ValueError(f"Record {index} is missing required fields: {missing}")

    for field in required_fields:
        value = record[field]
        if not isinstance(value, str):
            raise TypeError(f"Record {index} field '{field}' must be a string.")


def ensure_dataset_schema(data_path: Path) -> None:
    with data_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list) or not payload:
        raise ValueError("Training JSON must be a non-empty list of examples.")

    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise TypeError(f"Record {index} must be a JSON object.")
        validate_record(record, index)


def format_prompt(instruction: str, input_text: str, output_text: str = "") -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    output_text = output_text.strip()

    if input_text:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )

    return prompt + output_text


def build_training_dataset(data_path: Path, tokenizer) -> Dataset:
    dataset_dict = load_dataset("json", data_files=str(data_path))
    dataset = dataset_dict["train"]

    def add_text_column(batch: dict[str, list[str]]) -> dict[str, list[str]]:
        texts = []
        eos_token = tokenizer.eos_token or ""
        for instruction, input_text, output_text in zip(
            batch["instruction"], batch["input"], batch["output"]
        ):
            text = format_prompt(instruction, input_text, output_text) + eos_token
            texts.append(text)
        return {"text": texts}

    return dataset.map(add_text_column, batched=True, desc="Formatting instruction dataset")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    ensure_dataset_schema(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_bf16 = is_bfloat16_supported()
    use_fp16 = not use_bf16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    train_dataset = build_training_dataset(data_path, tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args.seed,
            save_strategy="steps",
            save_steps=args.save_steps,
            report_to="none",
        ),
    )

    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"Training finished. LoRA adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
