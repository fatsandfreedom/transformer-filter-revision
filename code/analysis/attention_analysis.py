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


def percentile_rank(values, target):
    return float(np.mean(values <= target))


def compute_attention_metrics(attention, positions, topk=5, warmup=10):
    spatial_distances = []
    temporal_distances = []
    attention_weights = []
    spatial_percentiles = []
    temporal_percentiles = []

    for traj_idx in range(attention.shape[0]):
        attn_traj = attention[traj_idx]
        pos_traj = positions[traj_idx]
        for t in range(max(warmup, topk), attn_traj.shape[0]):
            hist_weights = attn_traj[t, :t]
            if np.allclose(hist_weights.sum(), 0.0):
                continue
            top_indices = np.argsort(hist_weights)[-topk:]
            spatial_all = np.linalg.norm(pos_traj[:t] - pos_traj[t], axis=1)
            temporal_all = np.abs(np.arange(t) - t).astype(np.float32)

            for k in top_indices:
                spatial = float(spatial_all[k])
                temporal = float(temporal_all[k])
                weight = float(hist_weights[k])
                spatial_distances.append(spatial)
                temporal_distances.append(temporal)
                attention_weights.append(weight)
                spatial_percentiles.append(percentile_rank(spatial_all, spatial))
                temporal_percentiles.append(percentile_rank(temporal_all, temporal))

    spatial_distances = np.array(spatial_distances, dtype=np.float64)
    temporal_distances = np.array(temporal_distances, dtype=np.float64)
    attention_weights = np.array(attention_weights, dtype=np.float64)
    spatial_percentiles = np.array(spatial_percentiles, dtype=np.float64)
    temporal_percentiles = np.array(temporal_percentiles, dtype=np.float64)

    metrics = {
        "num_pairs": int(attention_weights.size),
        "topk": int(topk),
        "mean_attention_weight": float(np.mean(attention_weights)),
        "attention_spatial_corr": float(
            np.corrcoef(attention_weights, spatial_distances)[0, 1]
        ),
        "attention_temporal_corr": float(
            np.corrcoef(attention_weights, temporal_distances)[0, 1]
        ),
        "mean_spatial_distance": float(np.mean(spatial_distances)),
        "mean_temporal_gap": float(np.mean(temporal_distances)),
        "mean_spatial_percentile": float(np.mean(spatial_percentiles)),
        "mean_temporal_percentile": float(np.mean(temporal_percentiles)),
    }
    return metrics, spatial_distances, temporal_distances, attention_weights


def plot_attention_relationships(
    spatial_distances, temporal_distances, attention_weights, output_path
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(spatial_distances, attention_weights, s=8, alpha=0.35)
    axes[0].set_xlabel("Spatial Distance")
    axes[0].set_ylabel("Attention Weight")
    axes[0].set_title("Attention vs Spatial Distance")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(temporal_distances, attention_weights, s=8, alpha=0.35)
    axes[1].set_xlabel("Temporal Gap")
    axes[1].set_ylabel("Attention Weight")
    axes[1].set_title("Attention vs Temporal Gap")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    result = run_experiment(refresh=args.refresh, epochs=args.epochs, seed=args.seed)
    metrics, spatial_distances, temporal_distances, attention_weights = (
        compute_attention_metrics(result["attention"], result["z_raw"], topk=args.topk)
    )

    figure_path = os.path.join("outputs", "figs", "attention_distance_analysis.png")
    json_path = os.path.join("outputs", "logs", "attention_metrics.json")
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    plot_attention_relationships(
        spatial_distances, temporal_distances, attention_weights, figure_path
    )
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved figure to {figure_path}")
    print(f"Saved metrics to {json_path}")


if __name__ == "__main__":
    main()
