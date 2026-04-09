import argparse

import torch
from unsloth import FastLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a LoRA adapter trained by Unsloth.")
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the saved LoRA adapter directory.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="You are a professional domain QA assistant. Provide a clear and accurate answer.",
        help="Instruction for the model.",
    )
    parser.add_argument("--input_text", type=str, required=True, help="Question or user input.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum generation length.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling.")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    prompt = format_prompt(args.instruction, args.input_text)
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            use_cache=True,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = decoded.split("### Response:\n", 1)[-1].strip()
    print(answer)


if __name__ == "__main__":
    main()
