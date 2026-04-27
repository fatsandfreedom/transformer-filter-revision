import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
CODE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from sim_main import DEFAULT_EPOCHS, run_experiment


def weighted_prediction_mse(x_seq, y_seq, weights, filter_len, ridge=1e-2):
    gram = np.zeros((filter_len, filter_len), dtype=np.float64)
    cross = np.zeros(filter_len, dtype=np.float64)

    for idx, weight in enumerate(weights):
        if weight <= 0:
            continue
        xk = np.asarray(x_seq[idx], dtype=np.float64)
        yk = np.asarray(y_seq[idx], dtype=np.float64)
        x_matrix = np.lib.stride_tricks.sliding_window_view(xk, filter_len)[: len(yk)]
        gram += weight * (x_matrix.T @ x_matrix)
        cross += weight * (x_matrix.T @ yk[: x_matrix.shape[0]])

    gram += ridge * np.eye(filter_len)
    filt = np.linalg.solve(gram, cross)

    xt = np.asarray(x_seq[len(weights)], dtype=np.float64)
    yt = np.asarray(y_seq[len(weights)], dtype=np.float64)
    x_matrix_t = np.lib.stride_tricks.sliding_window_view(xt, filter_len)[: len(yt)]
    residual = yt[: x_matrix_t.shape[0]] - x_matrix_t @ filt
    return float(np.mean(residual ** 2))


def compare_kernels(result, warmup=10, step_stride=10):
    attention = result["attention"]
    x_raw = result["x_raw"]
    y_raw = result["y_raw"]
    filter_len = result["fps_tensor"].shape[-1]

    learned_losses = []
    uniform_losses = []

    for traj_idx in range(attention.shape[0]):
        attn_traj = attention[traj_idx]
        for t in range(warmup, attn_traj.shape[0], step_stride):
            learned_weights = attn_traj[t, :t].astype(np.float64)
            if np.allclose(learned_weights.sum(), 0.0):
                continue
            learned_weights /= learned_weights.sum()
            uniform_weights = np.ones(t, dtype=np.float64) / float(t)
            learned_losses.append(
                weighted_prediction_mse(
                    x_raw[traj_idx], y_raw[traj_idx], learned_weights, filter_len
                )
            )
            uniform_losses.append(
                weighted_prediction_mse(
                    x_raw[traj_idx], y_raw[traj_idx], uniform_weights, filter_len
                )
            )

    return {
        "num_evaluated_steps": int(len(learned_losses)),
        "attention_weighted_mse": float(np.mean(learned_losses)),
        "uniform_weighted_mse": float(np.mean(uniform_losses)),
        "mse_gap": float(np.mean(uniform_losses) - np.mean(learned_losses)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--step-stride", type=int, default=10)
    args = parser.parse_args()

    result = run_experiment(refresh=args.refresh, epochs=args.epochs, seed=args.seed)
    metrics = compare_kernels(result, step_stride=args.step_stride)

    figure_path = os.path.join("outputs", "figs", "kernel_compare.png")
    json_path = os.path.join("outputs", "logs", "kernel_compare.json")
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Attention weights", "Uniform weights"]
    values = [metrics["attention_weighted_mse"], metrics["uniform_weighted_mse"]]
    ax.bar(labels, values, color=["#d97757", "#8aa0b8"])
    ax.set_ylabel("Mean reconstruction MSE")
    ax.set_title("Kernel Weighting Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved figure to {figure_path}")
    print(f"Saved metrics to {json_path}")


if __name__ == "__main__":
    main()
