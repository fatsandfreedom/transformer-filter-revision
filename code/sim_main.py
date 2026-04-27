import argparse
import json
import math
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from tqdm import tqdm


DEFAULT_FILTER_LEN = 128
DEFAULT_TRAJ_LEN = 150
DEFAULT_NUM_TRAJS = 15
DEFAULT_EPOCHS = 150
DEFAULT_DATA_FILE = os.path.join("outputs", "logs", "isfo_final_v5.pkl")
DEFAULT_SUMMARY_FIG = os.path.join("outputs", "figs", "simulation_results.png")
DEFAULT_LOSS_FIG = os.path.join("outputs", "figs", "training_loss_curve.png")
DEFAULT_ATTN_DISTANCE_FIG = os.path.join(
    "outputs", "figs", "attention_distance_analysis.png"
)
DEFAULT_ATTN_DISTANCE_JSON = os.path.join(
    "outputs", "logs", "attention_distance_metrics.json"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


class AcousticEnvironment:
    def __init__(self, n_samples=500, fs=16000, filter_len=128):
        self.n_samples = n_samples
        self.fs = fs
        self.L = filter_len
        self.room_dim = [6, 4]
        self.mic_pos = np.array([[3.5], [1.5]])

    def generate_diverse_trajectories(self, num_trajs=15, points_per_traj=150):
        print(
            f"Generating {num_trajs} trajectories "
            f"({num_trajs * points_per_traj} physical samples)..."
        )
        all_samples = []
        total_len_needed = self.n_samples + self.L + 10
        x_signal_base = np.random.randn(total_len_needed)

        for traj_idx in tqdm(range(num_trajs)):
            start_angle = np.random.uniform(0, 2 * np.pi)
            direction = np.random.choice([-1, 1])
            t_steps = np.linspace(0, 2 * np.pi, points_per_traj)
            r_list = 0.5 + 2.0 * t_steps / (2 * np.pi)
            theta_list = start_angle + direction * t_steps

            for step_idx in range(points_per_traj):
                curr_r = r_list[step_idx]
                curr_theta = theta_list[step_idx]
                sx = np.clip(
                    self.mic_pos[0][0] + curr_r * np.cos(curr_theta), 0.2, 5.8
                )
                sy = np.clip(
                    self.mic_pos[1][0] + curr_r * np.sin(curr_theta), 0.2, 3.8
                )

                room = pra.ShoeBox(
                    self.room_dim, fs=self.fs, max_order=2, absorption=0.2
                )
                room.add_microphone_array(pra.MicrophoneArray(self.mic_pos, self.fs))
                room.add_source([sx, sy])
                room.compute_rir()

                rir = room.rir[0][0]
                h_eff = np.zeros(self.L)
                peak = np.argmax(np.abs(rir))
                start_h = max(0, peak - 2)
                valid_len = min(len(rir) - start_h, self.L)
                h_eff[:valid_len] = rir[start_h : start_h + valid_len]

                y_conv = np.convolve(x_signal_base, h_eff, mode="valid")
                y_store = y_conv[: self.n_samples]
                x_store = x_signal_base[: self.n_samples + self.L - 1]

                fp = np.zeros(self.L)
                for lag in range(self.L):
                    x_lag = x_store[self.L - 1 - lag : self.L - 1 - lag + self.n_samples]
                    fp[lag] = np.dot(y_store, x_lag)
                norm = np.linalg.norm(fp)
                if norm > 0:
                    fp /= norm

                all_samples.append(
                    {
                        "fp": fp.astype(np.float32),
                        "x": x_store.astype(np.float32),
                        "y": y_store.astype(np.float32),
                        "z": np.array([sx, sy], dtype=np.float32),
                        "traj_id": traj_idx,
                        "time_idx": step_idx,
                    }
                )
        return all_samples


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + 0.05 * self.pe[: x.size(1), :].unsqueeze(0)


class ISFOTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, filter_len=32, device="cpu"):
        super().__init__()
        self.L = filter_len
        self.device = device
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, d_model),
            nn.Tanh(),
        )
        self.pos = PositionalEncoding(d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.ridge = 1e-2

    def forward(self, fps, x_list, y_list):
        batch_size, seq_len, _ = fps.shape
        e_orig = self.enc(fps)
        e = self.pos(e_orig)

        query = self.W_q(e)
        key = self.W_k(e)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key.size(-1))
        mask = torch.triu(torch.ones(seq_len, seq_len), 1).bool().to(self.device)
        scores = scores.masked_fill(mask, -1e9)
        alpha = torch.softmax(scores * 2.0, dim=-1)

        total_loss = 0.0
        valid_samples = 0
        sample_steps = torch.randint(10, seq_len, (5,), device=self.device)

        for batch_idx in range(batch_size):
            attn_b = alpha[batch_idx]
            for t in sample_steps:
                w_t = attn_b[t]
                gram = torch.zeros(self.L, self.L, device=self.device)
                cross = torch.zeros(self.L, device=self.device)
                valid_idx = torch.where(w_t > 0.005)[0]
                if len(valid_idx) == 0:
                    continue
                for k in valid_idx:
                    xk = torch.from_numpy(x_list[batch_idx][k]).float().to(self.device)
                    yk = torch.from_numpy(y_list[batch_idx][k]).float().to(self.device)
                    x_matrix = xk.unfold(0, self.L, 1)[: len(yk)]
                    gram += w_t[k] * (x_matrix.t() @ x_matrix)
                    cross += w_t[k] * (x_matrix.t() @ yk[: x_matrix.size(0)])
                gram += self.ridge * torch.eye(self.L, device=self.device)
                try:
                    wt_hat = torch.linalg.solve(gram, cross)
                    xt = torch.from_numpy(x_list[batch_idx][t]).float().to(self.device)
                    yt = torch.from_numpy(y_list[batch_idx][t]).float().to(self.device)
                    x_matrix_t = xt.unfold(0, self.L, 1)[: len(yt)]
                    total_loss += torch.mean(
                        (yt[: x_matrix_t.size(0)] - x_matrix_t @ wt_hat) ** 2
                    )
                    valid_samples += 1
                except RuntimeError:
                    continue

        return total_loss / max(valid_samples, 1), alpha, e_orig


def load_or_generate_data(
    filter_len=DEFAULT_FILTER_LEN,
    traj_len=DEFAULT_TRAJ_LEN,
    num_trajs=DEFAULT_NUM_TRAJS,
    data_file=DEFAULT_DATA_FILE,
    refresh=False,
):
    ensure_parent_dir(data_file)
    if os.path.exists(data_file) and not refresh:
        with open(data_file, "rb") as handle:
            data = pickle.load(handle)
        if len(data[0]["fp"]) == filter_len:
            return data
        print("Cached data shape mismatch. Regenerating.")

    env = AcousticEnvironment(filter_len=filter_len)
    data = env.generate_diverse_trajectories(
        num_trajs=num_trajs, points_per_traj=traj_len
    )
    with open(data_file, "wb") as handle:
        pickle.dump(data, handle)
    return data


def prepare_sequences(data, num_trajs=DEFAULT_NUM_TRAJS, traj_len=DEFAULT_TRAJ_LEN):
    fps_all = np.array([item["fp"] for item in data], dtype=np.float32)
    fps_tensor = torch.tensor(fps_all, dtype=torch.float32).view(
        num_trajs, traj_len, -1
    )
    x_raw = [
        [item["x"] for item in data[i * traj_len : (i + 1) * traj_len]]
        for i in range(num_trajs)
    ]
    y_raw = [
        [item["y"] for item in data[i * traj_len : (i + 1) * traj_len]]
        for i in range(num_trajs)
    ]
    z_raw = np.array([item["z"] for item in data], dtype=np.float32).reshape(
        num_trajs, traj_len, 2
    )
    return fps_all, fps_tensor, x_raw, y_raw, z_raw


def train_model(model, fps_tensor, x_raw, y_raw, epochs=DEFAULT_EPOCHS, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    fps_tensor = fps_tensor.to(model.device)

    print("\nTraining model...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss, _, _ = model(fps_tensor, x_raw, y_raw)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        loss_history.append(loss_value)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: loss={loss_value:.6f}")
    return loss_history


def collect_outputs(model, fps_tensor, x_raw, y_raw):
    model.eval()
    fps_tensor = fps_tensor.to(model.device)

    with torch.no_grad():
        _, all_attn, _ = model(fps_tensor, x_raw, y_raw)
        _, _, all_emb = model(fps_tensor, x_raw, y_raw)

    attention = all_attn.detach().cpu().numpy()
    embeddings = all_emb.detach().cpu().numpy()
    return attention, embeddings


def compute_pca_views(fps_all, embeddings):
    pca_phys = PCA(n_components=2)
    fp_2d = pca_phys.fit_transform(fps_all)
    pca_emb = PCA(n_components=2)
    emb_2d = pca_emb.fit_transform(embeddings.reshape(-1, embeddings.shape[-1]))
    return fp_2d, emb_2d


def plot_summary(attention, z_raw, fp_2d, emb_2d, summary_path):
    ensure_parent_dir(summary_path)
    num_trajs, traj_len, _ = z_raw.shape
    colors = np.tile(np.arange(traj_len), num_trajs)

    fig = plt.figure(figsize=(20, 14))

    ax1 = fig.add_subplot(221)
    image = ax1.imshow(attention[0], cmap="hot", origin="lower")
    ax1.set_title("Causal Attention Map (Trajectory 0)")
    plt.colorbar(image, ax=ax1)

    ax2 = fig.add_subplot(222)
    z_flat = z_raw.reshape(-1, 2)
    ax2.scatter(z_flat[:, 0], z_flat[:, 1], c=colors, s=10, cmap="viridis", alpha=0.6)
    ax2.set_title("Physical Space Trajectories (X, Y)")
    ax2.set_xlabel("Room X")
    ax2.set_ylabel("Room Y")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(223)
    ax3.scatter(fp_2d[:, 0], fp_2d[:, 1], c=colors, s=10, cmap="viridis", alpha=0.6)
    ax3.set_title("Fingerprint Geometry (PCA)")
    ax3.set_xlabel("PC 1")
    ax3.set_ylabel("PC 2")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(224)
    ax4.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=10, cmap="viridis", alpha=0.8)
    ax4.set_title("Learned Latent Geometry (PCA)")
    ax4.set_xlabel("Latent Axis 1")
    ax4.set_ylabel("Latent Axis 2")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(loss_history, loss_path):
    ensure_parent_dir(loss_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(loss_history) + 1), loss_history, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def percentile_rank(values, target):
    return float(np.mean(values <= target))


def analyze_attention_distance(attention, z_raw, topk=5, warmup=10):
    spatial_distances = []
    temporal_distances = []
    attention_weights = []
    spatial_percentiles = []
    temporal_percentiles = []

    for traj_idx in range(attention.shape[0]):
        attn_traj = attention[traj_idx]
        pos_traj = z_raw[traj_idx]
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


def plot_attention_distance_analysis(
    spatial_distances, temporal_distances, attention_weights, output_path
):
    ensure_parent_dir(output_path)
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


def run_experiment(
    refresh=False,
    epochs=DEFAULT_EPOCHS,
    seed=0,
    data_file=DEFAULT_DATA_FILE,
    summary_path=DEFAULT_SUMMARY_FIG,
    loss_path=DEFAULT_LOSS_FIG,
    attn_distance_fig_path=None,
    attn_distance_json_path=None,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_or_generate_data(data_file=data_file, refresh=refresh)
    fps_all, fps_tensor, x_raw, y_raw, z_raw = prepare_sequences(data)

    model = ISFOTransformer(
        input_dim=fps_tensor.shape[-1],
        d_model=64,
        filter_len=fps_tensor.shape[-1],
        device=device,
    ).to(device)
    loss_history = train_model(model, fps_tensor, x_raw, y_raw, epochs=epochs)
    attention, embeddings = collect_outputs(model, fps_tensor, x_raw, y_raw)
    fp_2d, emb_2d = compute_pca_views(fps_all, embeddings)

    plot_summary(attention, z_raw, fp_2d, emb_2d, summary_path)
    plot_loss_curve(loss_history, loss_path)

    attention_distance_metrics = None
    if attn_distance_fig_path or attn_distance_json_path:
        (
            attention_distance_metrics,
            spatial_distances,
            temporal_distances,
            attention_weights,
        ) = analyze_attention_distance(attention, z_raw)

        if attn_distance_fig_path:
            plot_attention_distance_analysis(
                spatial_distances,
                temporal_distances,
                attention_weights,
                attn_distance_fig_path,
            )
        if attn_distance_json_path:
            ensure_parent_dir(attn_distance_json_path)
            with open(attn_distance_json_path, "w", encoding="utf-8") as handle:
                json.dump(attention_distance_metrics, handle, indent=2)

    return {
        "model": model,
        "data": data,
        "fps_all": fps_all,
        "fps_tensor": fps_tensor,
        "x_raw": x_raw,
        "y_raw": y_raw,
        "z_raw": z_raw,
        "attention": attention,
        "embeddings": embeddings,
        "fp_2d": fp_2d,
        "emb_2d": emb_2d,
        "loss_history": loss_history,
        "device": device,
        "attention_distance_metrics": attention_distance_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--analyze-attention-distance", action="store_true")
    args = parser.parse_args()

    run_experiment(
        refresh=args.refresh,
        epochs=args.epochs,
        seed=args.seed,
        attn_distance_fig_path=(
            DEFAULT_ATTN_DISTANCE_FIG if args.analyze_attention_distance else None
        ),
        attn_distance_json_path=(
            DEFAULT_ATTN_DISTANCE_JSON if args.analyze_attention_distance else None
        ),
    )


if __name__ == "__main__":
    main()
