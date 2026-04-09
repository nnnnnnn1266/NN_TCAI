import os
from pathlib import Path
from threading import Lock
from typing import Any


MOCK_KNOWLEDGE: dict[str, str] = {
    "why do turtles bask": (
        "Turtles bask to regulate body temperature, dry their shells, and "
        "support healthy metabolism with access to heat and UVB light."
    ),
    "what do turtles eat": (
        "A turtle's diet depends on its species, but many pet turtles need a "
        "balance of commercial pellets, leafy greens, and occasional protein."
    ),
    "how often should i clean a turtle tank": (
        "Spot cleaning should happen often, while partial or full tank cleaning "
        "depends on tank size, filtration, and how quickly waste builds up."
    ),
    "why is my turtle not eating": (
        "A turtle may stop eating because of low temperature, stress, poor water "
        "quality, seasonal changes, or illness. Check husbandry conditions first."
    ),
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER_PATH = PROJECT_ROOT / "outputs" / "llama31-turtle-lora"
DEFAULT_INSTRUCTION = (
    "You are a professional turtle care assistant. Provide a clear, accurate, "
    "and practical answer."
)

_runtime_lock = Lock()
_runtime: "AdapterRuntime | None" = None
_runtime_attempted = False
_runtime_status = "mock"
_runtime_reason = "Adapter has not been loaded yet."


def _format_prompt(instruction: str, input_text: str) -> str:
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


def _mock_answer(question: str) -> str:
    normalized = question.strip().lower().rstrip("?.!")

    if not normalized:
        return "Please enter a turtle care question."

    for key, answer in MOCK_KNOWLEDGE.items():
        if key in normalized:
            return answer

    return (
        "The API is currently using the mock inference fallback. Train or connect "
        "a LoRA adapter to get domain-tuned answers for this question: "
        f"'{question.strip()}'."
    )


class AdapterRuntime:
    def __init__(self, model: Any, tokenizer: Any, instruction: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.instruction = instruction

    def answer(self, question: str, max_new_tokens: int = 256) -> str:
        import torch

        prompt = _format_prompt(self.instruction, question)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.split("### Response:\n", 1)[-1].strip()


def _resolve_adapter_path() -> Path:
    configured_path = os.getenv("TCAI_ADAPTER_PATH")
    if configured_path:
        return Path(configured_path).expanduser()
    return DEFAULT_ADAPTER_PATH


def _load_runtime() -> "AdapterRuntime | None":
    global _runtime_status, _runtime_reason

    adapter_path = _resolve_adapter_path()
    if not adapter_path.exists():
        _runtime_status = "mock"
        _runtime_reason = f"Adapter directory not found: {adapter_path}"
        return None

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        _runtime_status = "mock"
        _runtime_reason = "Unsloth is not installed in the current environment."
        return None

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path),
            max_seq_length=int(os.getenv("TCAI_MAX_SEQ_LENGTH", "2048")),
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    except Exception as exc:
        _runtime_status = "mock"
        _runtime_reason = f"Adapter load failed: {exc}"
        return None

    _runtime_status = "lora"
    _runtime_reason = f"Loaded adapter from: {adapter_path}"
    return AdapterRuntime(
        model=model,
        tokenizer=tokenizer,
        instruction=os.getenv("TCAI_SYSTEM_PROMPT", DEFAULT_INSTRUCTION),
    )


def _get_runtime() -> "AdapterRuntime | None":
    global _runtime, _runtime_attempted

    if _runtime_attempted:
        return _runtime

    with _runtime_lock:
        if not _runtime_attempted:
            _runtime = _load_runtime()
            _runtime_attempted = True

    return _runtime


def get_inference_status() -> dict[str, str]:
    _get_runtime()
    return {"mode": _runtime_status, "detail": _runtime_reason}


def generate_answer(question: str) -> str:
    if not question.strip():
        return "Please enter a turtle care question."

    runtime = _get_runtime()
    if runtime is None:
        return _mock_answer(question)

    try:
        return runtime.answer(question)
    except Exception as exc:
        return (
            f"{_mock_answer(question)} "
            f"(Model inference failed, fallback engaged: {exc})"
        )
