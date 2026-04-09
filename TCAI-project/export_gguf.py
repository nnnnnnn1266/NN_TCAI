import argparse
from pathlib import Path

from unsloth import FastLanguageModel


DEFAULT_ADAPTER_PATH = "outputs/llama31-turtle-lora"
DEFAULT_OUTPUT_DIR = "exports/llama31-turtle-gguf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained Unsloth LoRA adapter to a GGUF model."
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=DEFAULT_ADAPTER_PATH,
        help="Path to the trained LoRA adapter directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where GGUF files will be written.",
    )
    parser.add_argument(
        "--quantization_method",
        type=str,
        default="q4_k_m",
        help="GGUF quantization method, for example q4_k_m, q5_k_m, q8_0, f16.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length used when loading the model.",
    )
    parser.add_argument(
        "--maximum_memory_usage",
        type=float,
        default=0.75,
        help="Fraction of peak GPU memory Unsloth may use while exporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    adapter_path = Path(args.adapter_path)
    output_dir = Path(args.output_dir)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading adapter from: {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    if not hasattr(model, "save_pretrained_gguf"):
        raise RuntimeError(
            "This model instance does not expose save_pretrained_gguf. "
            "Please verify the Unsloth installation in the current environment."
        )

    print(f"Exporting GGUF to: {output_dir}")
    print(f"Quantization method: {args.quantization_method}")
    model.save_pretrained_gguf(
        str(output_dir),
        tokenizer,
        quantization_method=args.quantization_method,
        maximum_memory_usage=args.maximum_memory_usage,
    )
    print(f"GGUF export finished: {output_dir}")


if __name__ == "__main__":
    main()
