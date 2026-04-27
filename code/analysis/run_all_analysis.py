import json
import os
import sys
import argparse

CURRENT_DIR = os.path.dirname(__file__)
CODE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from attention_analysis import compute_attention_metrics, plot_attention_relationships
from kernel_compare import compare_kernels
from knn_metrics import knn_overlap
from sim_main import DEFAULT_EPOCHS, run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    result = run_experiment(refresh=args.refresh, epochs=args.epochs, seed=args.seed)

    attn_metrics, spatial_distances, temporal_distances, attention_weights = (
        compute_attention_metrics(result["attention"], result["z_raw"], topk=5)
    )
    plot_attention_relationships(
        spatial_distances,
        temporal_distances,
        attention_weights,
        os.path.join("outputs", "figs", "attention_distance_analysis.png"),
    )

    physical = result["z_raw"].reshape(-1, 2)
    knn_metrics = {
        "k": 10,
        "raw_fingerprint_pca_knn_overlap": knn_overlap(physical, result["fp_2d"], k=10),
        "latent_embedding_pca_knn_overlap": knn_overlap(physical, result["emb_2d"], k=10),
    }
    knn_metrics["improvement"] = (
        knn_metrics["latent_embedding_pca_knn_overlap"]
        - knn_metrics["raw_fingerprint_pca_knn_overlap"]
    )

    kernel_metrics = compare_kernels(result, step_stride=10)

    os.makedirs(os.path.join("outputs", "logs"), exist_ok=True)
    with open(
        os.path.join("outputs", "logs", "attention_metrics.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(attn_metrics, handle, indent=2)
    with open(
        os.path.join("outputs", "logs", "knn_metrics.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(knn_metrics, handle, indent=2)
    with open(
        os.path.join("outputs", "logs", "kernel_compare.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(kernel_metrics, handle, indent=2)

    print(
        json.dumps(
            {
                "attention": attn_metrics,
                "knn": knn_metrics,
                "kernel": kernel_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
