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


def pairwise_distances(features):
    squared_norms = np.sum(features * features, axis=1, keepdims=True)
    distances = squared_norms + squared_norms.T - 2.0 * (features @ features.T)
    np.maximum(distances, 0.0, out=distances)
    return distances


def knn_indices(features, k):
    distances = pairwise_distances(features.astype(np.float32))
    np.fill_diagonal(distances, np.inf)
    return np.argpartition(distances, kth=k, axis=1)[:, :k]


def knn_overlap(reference_features, candidate_features, k=10):
    ref_neighbors = knn_indices(reference_features, k)
    cand_neighbors = knn_indices(candidate_features, k)

    overlaps = []
    for ref, cand in zip(ref_neighbors, cand_neighbors):
        overlaps.append(len(set(ref.tolist()) & set(cand.tolist())) / float(k))
    return float(np.mean(overlaps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    result = run_experiment(refresh=args.refresh, epochs=args.epochs, seed=args.seed)
    physical = result["z_raw"].reshape(-1, 2)
    raw_pca = result["fp_2d"]
    emb_pca = result["emb_2d"]

    raw_score = knn_overlap(physical, raw_pca, k=args.k)
    emb_score = knn_overlap(physical, emb_pca, k=args.k)
    metrics = {
        "k": int(args.k),
        "raw_fingerprint_pca_knn_overlap": raw_score,
        "latent_embedding_pca_knn_overlap": emb_score,
        "improvement": emb_score - raw_score,
    }

    figure_path = os.path.join("outputs", "figs", "knn_consistency.png")
    json_path = os.path.join("outputs", "logs", "knn_metrics.json")
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Raw fingerprint", "Latent embedding"]
    values = [raw_score, emb_score]
    ax.bar(labels, values, color=["#8aa0b8", "#d97757"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(f"Mean k-NN overlap (k={args.k})")
    ax.set_title("Local Geometry Consistency")
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
