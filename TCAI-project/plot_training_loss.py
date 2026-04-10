import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training loss from Hugging Face trainer_state.json."
    )
    parser.add_argument(
        "--trainer_state",
        type=str,
        default="outputs/llama31-turtle-lora/checkpoint-477/trainer_state.json",
        help="Path to trainer_state.json.",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="outputs/plots/training_loss.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/plots/training_loss.csv",
        help="Optional CSV export path for plotted points.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer_state_path = Path(args.trainer_state)
    output_png = Path(args.output_png)
    output_csv = Path(args.output_csv)

    if not trainer_state_path.exists():
        raise FileNotFoundError(f"trainer_state.json not found: {trainer_state_path}")

    with trainer_state_path.open("r", encoding="utf-8") as file:
        state = json.load(file)

    log_history = state.get("log_history", [])
    points = []
    for row in log_history:
        if "loss" in row and "step" in row:
            points.append(
                {
                    "step": row["step"],
                    "epoch": row.get("epoch"),
                    "loss": row["loss"],
                }
            )

    if not points:
        raise RuntimeError("No training loss points found in trainer_state.json.")

    output_png.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as file:
        file.write("step,epoch,loss\n")
        for point in points:
            file.write(f"{point['step']},{point['epoch']},{point['loss']}\n")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to plot training loss. "
            "Install it with: .conda312\\python.exe -m pip install matplotlib"
        ) from exc

    steps = [point["step"] for point in points]
    losses = [point["loss"] for point in points]

    plt.figure(figsize=(9, 5))
    plt.plot(steps, losses, color="#2e7d6b", linewidth=1.8)
    plt.title("TCAI Training Loss Curve")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()

    print(f"Saved loss plot: {output_png}")
    print(f"Saved loss CSV : {output_csv}")
    print(f"Total points   : {len(points)}")


if __name__ == "__main__":
    main()
