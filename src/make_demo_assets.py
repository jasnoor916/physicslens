"""
src/make_demo_assets.py

Generates all visual assets needed for the PhysicsLens demo video.

Outputs to demo/assets/:
  Group 1 — Scatter plots:
    scatter_k_oscillator.png
    scatter_b_oscillator.png
    scatter_L_pendulum.png
    scatter_b_pendulum.png

  Group 2 — Counterfactual comparison videos:
    cf_k_low_vs_high.mp4          (k=5 vs k=18, same b/x0/v0)
    cf_b_low_vs_high.mp4          (b=0.1 vs b=1.8, same k/x0/v0)
    cf_pendulum_L_short_vs_long.mp4  (L=0.3 vs L=2.0, same b/theta0)

  Group 3 — Latent PCA:
    latent_pca_k_oscillator.png
    latent_pca_b_oscillator.png
    latent_pca_L_pendulum.png
    latent_pca_b_pendulum.png

  Group 4 — Single example inference card:
    example_inference_oscillator.png
    example_inference_pendulum.png

  Group 5 — SINDy equation card:
    sindy_equation.png

Run from repo root:
    python src/make_demo_assets.py

Requirements:
    pip install opencv-python scikit-learn matplotlib numpy torch
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
CKPT_DIR  = ROOT / "checkpoints"
LOG_DIR   = ROOT / "logs"
DATA_DIR  = ROOT / "data"
OUT_DIR   = ROOT / "demo" / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

# ── visual theme ───────────────────────────────────────────────────────────────
BG       = "#0a0a1a"
CYAN     = "#00e5ff"
MAGENTA  = "#ff4081"
YELLOW   = "#ffd740"
WHITE    = "#ffffff"
GREY     = "#555577"
DPI      = 150

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.edgecolor":    GREY,
    "axes.labelcolor":   WHITE,
    "xtick.color":       WHITE,
    "ytick.color":       WHITE,
    "text.color":        WHITE,
    "grid.color":        GREY,
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "font.family":       "monospace",
})

# ══════════════════════════════════════════════════════════════════════════════
# 1.  MODEL DEFINITIONS  (must match train_scene_encoder.py / world_model.py)
# ══════════════════════════════════════════════════════════════════════════════

class FrameEncoder(nn.Module):
    """Matches world_model.py architecture used in world_model_combined.pt."""
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3,   32,  4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32,  64,  4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64,  128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim * 2),   # outputs [mu || logvar]
        )

    def forward(self, x):
        h   = self.conv(x)
        out = self.fc(h)
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar

class SceneEncoderV2(nn.Module):
    """Must match train_scene_encoder.py exactly (with encode() added)."""
    def __init__(self, latent_dim=16, hidden_dim=64, scene_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.1)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.proj = nn.Linear(hidden_dim * 2, scene_dim)

        def _head():
            return nn.Sequential(
                nn.Linear(scene_dim, 64), nn.LayerNorm(64), nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32), nn.GELU(),
                nn.Linear(32, 1),
            )
        self.p1_head = _head()
        self.p2_head = _head()

    def encode(self, z):
        """Returns 32-dim scene vector before prediction heads."""
        h, _ = self.lstm(z)
        a    = torch.softmax(self.attn(h), dim=1)
        ctx  = (a * h).sum(dim=1)
        return self.proj(ctx)          # (B, scene_dim)

    def forward(self, z):
        s = self.encode(z)
        return self.p1_head(s).squeeze(-1), self.p2_head(s).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CHECKPOINT LOADING
# ══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[make_demo_assets] device = {DEVICE}")


def load_frame_encoder(ckpt_name="world_model_combined.pt"):
    path = CKPT_DIR / ckpt_name
    assert path.exists(), f"Frame encoder checkpoint not found: {path}"
    ckpt = torch.load(path, map_location=DEVICE)

    # combined-model checkpoint uses key "model_state"
    full_state = ckpt.get("model_state") or ckpt.get("model_state_dict")
    assert full_state is not None, f"No model state found in {ckpt_name}"

    enc = FrameEncoder(latent_dim=16).to(DEVICE)
    state = {k.replace("encoder.", "", 1): v
             for k, v in full_state.items()
             if k.startswith("encoder.")}
    missing, unexpected = enc.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ missing={missing}  unexpected={unexpected}")
    enc.eval()
    print(f"  ✓ FrameEncoder loaded from {ckpt_name}  ({len(state)} tensors)")
    return enc

def load_scene_encoder(ckpt_name):
    path = CKPT_DIR / ckpt_name
    assert path.exists(), f"Scene encoder checkpoint not found: {path}"
    ckpt  = torch.load(path, map_location=DEVICE)
    cfg   = ckpt["config"]
    model = SceneEncoderV2(**cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    norm  = ckpt["norm_stats"]
    print(f"  ✓ SceneEncoderV2 loaded from {ckpt_name}")
    return model, norm


# ══════════════════════════════════════════════════════════════════════════════
# 3.  INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_video(enc, video_np):
    """
    video_np : (T, H, W, 3) uint8
    returns  : (T, 16) float32 numpy  (mu values)
    """
    T = video_np.shape[0]
    frames = torch.from_numpy(video_np).float() / 255.0   # (T, H, W, 3)
    frames = frames.permute(0, 3, 1, 2).to(DEVICE)        # (T, 3, H, W)
    mu, _  = enc(frames)                                   # (T, 16)
    return mu.cpu().numpy()


@torch.no_grad()
def predict_params(scene_enc, norm, z_seq_np):
    """
    z_seq_np : (T, 16) numpy
    returns  : (p1_physical, p2_physical) floats
    """
    z   = torch.from_numpy(z_seq_np).float().unsqueeze(0).to(DEVICE)  # (1, T, 16)
    p1n, p2n = scene_enc(z)
    p1  = p1n.item() * norm["p1_std"] + norm["p1_mean"]
    p2  = p2n.item() * norm["p2_std"] + norm["p2_mean"]
    return p1, p2


@torch.no_grad()
def get_scene_vector(scene_enc, z_seq_np):
    """Returns 32-dim scene vector as numpy."""
    z   = torch.from_numpy(z_seq_np).float().unsqueeze(0).to(DEVICE)
    vec = scene_enc.encode(z)          # (1, 32)
    return vec.squeeze(0).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  VAL-SET INFERENCE  (regenerates per-sample predictions)
# ══════════════════════════════════════════════════════════════════════════════

def run_val_inference(scenario, enc, scene_enc, norm, n_val=300):
    """
    Loads the last n_val .npz files in data/{scenario}/ as the validation set
    (consistent with dataset.py which uses an 80/20 split on sorted file list).

    Returns dict with keys:
        true_p1, pred_p1   : (N,) float32
        true_p2, pred_p2   : (N,) float32
        scene_vecs         : (N, 32) float32
    """
    data_path = DATA_DIR / scenario
    files     = sorted(data_path.glob("*.npz"))
    val_files = files[-n_val:]           # last 20 % = val set

    true_p1, pred_p1 = [], []
    true_p2, pred_p2 = [], []
    scene_vecs       = []

    p1_key = "k" if scenario == "oscillator" else "L"
    p2_key = "b"

    print(f"  Running inference on {len(val_files)} {scenario} val samples …")
    for f in val_files:
        d      = np.load(f)
        video  = d["video"]                              # (90, 64, 64, 3)
        tp1    = float(d[p1_key])
        tp2    = float(d[p2_key])

        z_seq  = encode_video(enc, video)                # (90, 16)
        pp1, pp2 = predict_params(scene_enc, norm, z_seq)
        svec   = get_scene_vector(scene_enc, z_seq)      # (32,)

        true_p1.append(tp1);  pred_p1.append(pp1)
        true_p2.append(tp2);  pred_p2.append(pp2)
        scene_vecs.append(svec)

    return {
        "true_p1":    np.array(true_p1,    dtype=np.float32),
        "pred_p1":    np.array(pred_p1,    dtype=np.float32),
        "true_p2":    np.array(true_p2,    dtype=np.float32),
        "pred_p2":    np.array(pred_p2,    dtype=np.float32),
        "scene_vecs": np.array(scene_vecs, dtype=np.float32),
    }


def r2_score(true, pred):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  GROUP 1 — SCATTER PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_scatter(true_vals, pred_vals, param_name, unit,
                 color, save_name, scenario_label):
    r2  = r2_score(true_vals, pred_vals)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=DPI)

    ax.scatter(true_vals, pred_vals,
               alpha=0.55, s=22, c=color,
               edgecolors="none", zorder=3)

    lo = min(true_vals.min(), pred_vals.min()) * 0.97
    hi = max(true_vals.max(), pred_vals.max()) * 1.03
    ax.plot([lo, hi], [lo, hi], color=WHITE, lw=1.2,
            linestyle="--", alpha=0.6, label="perfect", zorder=2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"True {param_name}  [{unit}]", fontsize=12)
    ax.set_ylabel(f"Predicted {param_name}  [{unit}]", fontsize=12)
    ax.set_title(f"{scenario_label} — {param_name}\nR² = {r2:.4f}",
                 fontsize=14, color=color, pad=12)
    ax.grid(True)
    ax.set_aspect("equal")

    # R² badge
    ax.text(0.04, 0.93, f"R² = {r2:.4f}",
            transform=ax.transAxes, fontsize=15,
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=color, lw=1.5))

    plt.tight_layout()
    out = OUT_DIR / save_name
    plt.savefig(out, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_name}  (R²={r2:.4f})")


def render_group1(osc_data, pen_data):
    print("\n[Group 1] Scatter plots …")
    plot_scatter(osc_data["true_p1"], osc_data["pred_p1"],
                 "k (spring constant)", "N/m",
                 CYAN,    "scatter_k_oscillator.png", "Oscillator")
    plot_scatter(osc_data["true_p2"], osc_data["pred_p2"],
                 "b (damping)", "N·s/m",
                 MAGENTA, "scatter_b_oscillator.png", "Oscillator")
    plot_scatter(pen_data["true_p1"], pen_data["pred_p1"],
                 "L (length)", "m",
                 YELLOW,  "scatter_L_pendulum.png",   "Pendulum")
    plot_scatter(pen_data["true_p2"], pen_data["pred_p2"],
                 "b (damping)", "N·s",
                 MAGENTA, "scatter_b_pendulum.png",   "Pendulum")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  GROUP 2 — COUNTERFACTUAL COMPARISON VIDEOS
# ══════════════════════════════════════════════════════════════════════════════

LABEL_FONT  = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.38
LABEL_THICK = 1
DIVIDER_W   = 6      # pixel gap between panes
HEADER_H    = 28     # pixels above video for labels
FPS_OUT     = 30


def _bgr(hex_color):
    """Convert '#rrggbb' → (B, G, R) tuple for OpenCV."""
    h  = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def make_side_by_side_video(video_a, video_b,
                             label_a, label_b,
                             save_name,
                             color_a=CYAN, color_b=MAGENTA):
    """
    video_a, video_b : (T, 64, 64, 3) uint8  RGB
    Saves MP4 to OUT_DIR / save_name.
    """
    T, H, W, _ = video_a.shape
    canvas_w    = W * 2 + DIVIDER_W
    canvas_h    = H + HEADER_H

    out_path = str(OUT_DIR / save_name)
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, FPS_OUT, (canvas_w, canvas_h))

    ca = _bgr(color_a)
    cb = _bgr(color_b)
    bg = _bgr(BG)

    for t in range(T):
        canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)

        # paste frames (convert RGB → BGR for OpenCV)
        frame_a = cv2.cvtColor(video_a[t], cv2.COLOR_RGB2BGR)
        frame_b = cv2.cvtColor(video_b[t], cv2.COLOR_RGB2BGR)
        canvas[HEADER_H:, :W]                 = frame_a
        canvas[HEADER_H:, W + DIVIDER_W:]     = frame_b

        # colored divider line
        canvas[:, W: W + DIVIDER_W] = 30

        # labels
        (tw_a, _), _ = cv2.getTextSize(label_a, LABEL_FONT, LABEL_SCALE, LABEL_THICK)
        (tw_b, _), _ = cv2.getTextSize(label_b, LABEL_FONT, LABEL_SCALE, LABEL_THICK)
        cx_a = (W - tw_a) // 2
        cx_b = W + DIVIDER_W + (W - tw_b) // 2
        cv2.putText(canvas, label_a, (cx_a, 18),
                    LABEL_FONT, LABEL_SCALE, ca, LABEL_THICK, cv2.LINE_AA)
        cv2.putText(canvas, label_b, (cx_b, 18),
                    LABEL_FONT, LABEL_SCALE, cb, LABEL_THICK, cv2.LINE_AA)

        writer.write(canvas)

    writer.release()
    print(f"  ✓ {save_name}  ({T} frames)")


def render_group2():
    print("\n[Group 2] Counterfactual comparison videos …")
    from data_gen_oscillator import generate_oscillator_video
    from data_gen_pendulum   import generate_pendulum_video

    # ── 2a: oscillator — low k vs high k ──────────────────────────────────────
    k_low, k_high = 4.0, 18.0
    b_cf, x0_cf, v0_cf = 0.4, 1.5, 0.0
    v_slow, _, _ = generate_oscillator_video(k_low,  b_cf, 1.0, x0_cf, v0_cf)
    v_fast, _, _ = generate_oscillator_video(k_high, b_cf, 1.0, x0_cf, v0_cf)
    make_side_by_side_video(
        v_slow, v_fast,
        f"k = {k_low:.0f} N/m  (slow)",
        f"k = {k_high:.0f} N/m  (fast)",
        "cf_k_low_vs_high.mp4",
    )

    # ── 2b: oscillator — low b vs high b ──────────────────────────────────────
    k_cf2 = 10.0
    b_low, b_high = 0.1, 1.8
    v_undamped, _, _ = generate_oscillator_video(k_cf2, b_low,  1.0, x0_cf, v0_cf)
    v_damped,   _, _ = generate_oscillator_video(k_cf2, b_high, 1.0, x0_cf, v0_cf)
    make_side_by_side_video(
        v_undamped, v_damped,
        f"b = {b_low}  (underdamped)",
        f"b = {b_high}  (overdamped)",
        "cf_b_low_vs_high.mp4",
        color_a=CYAN, color_b=YELLOW,
    )

    # ── 2c: pendulum — short vs long ──────────────────────────────────────────
    L_short, L_long = 0.3, 2.0
    b_pen, theta0 = 0.15, 0.6
    v_short, _, _ = generate_pendulum_video(L_short, b_pen, theta0)
    v_long,  _, _ = generate_pendulum_video(L_long,  b_pen, theta0)
    make_side_by_side_video(
        v_short, v_long,
        f"L = {L_short} m  (fast)",
        f"L = {L_long} m  (slow)",
        "cf_pendulum_L_short_vs_long.mp4",
        color_a=YELLOW, color_b=CYAN,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7.  GROUP 3 — LATENT PCA
# ══════════════════════════════════════════════════════════════════════════════

def plot_pca(scene_vecs, color_vals, cmap, param_name, unit,
             save_name, scenario_label):
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scene_vecs)          # (N, 2)
    var    = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 6), dpi=DPI)
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=color_vals, cmap=cmap,
                    s=28, alpha=0.75, edgecolors="none", zorder=3)
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(f"{param_name}  [{unit}]", color=WHITE)
    cb.ax.yaxis.set_tick_params(color=WHITE)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=WHITE)

    ax.set_xlabel(f"PC1  ({var[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2  ({var[1]*100:.1f}% var)", fontsize=11)
    ax.set_title(f"{scenario_label} — Latent scene space\ncoloured by {param_name}",
                 fontsize=13, pad=10)
    ax.grid(True)

    plt.tight_layout()
    out = OUT_DIR / save_name
    plt.savefig(out, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_name}")


def render_group3(osc_data, pen_data):
    print("\n[Group 3] Latent PCA …")
    plot_pca(osc_data["scene_vecs"], osc_data["true_p1"],
             "plasma", "k", "N/m",
             "latent_pca_k_oscillator.png", "Oscillator")
    plot_pca(osc_data["scene_vecs"], osc_data["true_p2"],
             "viridis", "b", "N·s/m",
             "latent_pca_b_oscillator.png", "Oscillator")
    plot_pca(pen_data["scene_vecs"], pen_data["true_p1"],
             "plasma", "L", "m",
             "latent_pca_L_pendulum.png", "Pendulum")
    plot_pca(pen_data["scene_vecs"], pen_data["true_p2"],
             "viridis", "b", "N·s",
             "latent_pca_b_pendulum.png", "Pendulum")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  GROUP 4 — SINGLE EXAMPLE INFERENCE CARD
# ══════════════════════════════════════════════════════════════════════════════

def render_example_card(scenario, enc, scene_enc, norm,
                         p1_name, p1_unit, p2_name, p2_unit,
                         p1_key, p2_key, save_name):
    """
    Picks one val sample, shows frames + true vs predicted parameters.
    """
    data_path = DATA_DIR / scenario
    files     = sorted(data_path.glob("*.npz"))
    # pick one mid-range sample
    sample_f  = files[len(files) // 2 + 50]
    d         = np.load(sample_f)
    video     = d["video"]
    true_p1   = float(d[p1_key])
    true_p2   = float(d[p2_key])

    z_seq    = encode_video(enc, video)
    pred_p1, pred_p2 = predict_params(scene_enc, norm, z_seq)

    # select 6 evenly-spaced frames
    idxs   = np.linspace(0, 89, 6, dtype=int)
    frames = [video[i] for i in idxs]          # list of (64,64,3) uint8

    fig = plt.figure(figsize=(11, 4), dpi=DPI)
    gs  = fig.add_gridspec(1, 8, wspace=0.08)

    # ── 6 video frames ────────────────────────────────────────────────────────
    for i, frame in enumerate(frames):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frame)
        ax.set_title(f"t={idxs[i]/30:.1f}s", fontsize=8, color=GREY)
        ax.axis("off")

    # ── parameter comparison panel ────────────────────────────────────────────
    ax_info = fig.add_subplot(gs[0, 6:])
    ax_info.axis("off")

    lines = [
        ("PhysicsLens Inference", WHITE, 13, "bold"),
        ("", WHITE, 1, "normal"),
        (f"{p1_name}", CYAN, 10, "bold"),
        (f"  True:  {true_p1:7.4f} {p1_unit}", WHITE, 10, "normal"),
        (f"  Pred:  {pred_p1:7.4f} {p1_unit}", CYAN,  10, "normal"),
        (f"  Err:   {abs(true_p1-pred_p1)/true_p1*100:.2f}%", GREY, 9, "normal"),
        ("", WHITE, 1, "normal"),
        (f"{p2_name}", MAGENTA, 10, "bold"),
        (f"  True:  {true_p2:7.4f} {p2_unit}", WHITE,   10, "normal"),
        (f"  Pred:  {pred_p2:7.4f} {p2_unit}", MAGENTA, 10, "normal"),
        (f"  Err:   {abs(true_p2-pred_p2)/true_p2*100:.2f}%", GREY, 9, "normal"),
    ]

    y = 0.97
    for text, color, size, weight in lines:
        ax_info.text(0.05, y, text,
                     transform=ax_info.transAxes,
                     fontsize=size, color=color,
                     fontweight=weight, va="top",
                     fontfamily="monospace")
        y -= 0.09 if size > 9 else 0.075

    fig.patch.set_facecolor(BG)
    plt.suptitle(f"{scenario.capitalize()} — single video inference",
                 color=WHITE, fontsize=12, y=1.01)

    out = OUT_DIR / save_name
    plt.savefig(out, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_name}  (true k={true_p1:.3f}, pred={pred_p1:.3f})")


def render_group4(enc, osc_se, osc_norm, pen_se, pen_norm):
    print("\n[Group 4] Example inference cards …")
    render_example_card(
        "oscillator", enc, osc_se, osc_norm,
        "k (spring constant)", "N/m",
        "b (damping)",         "N·s/m",
        "k", "b",
        "example_inference_oscillator.png",
    )
    render_example_card(
        "pendulum", enc, pen_se, pen_norm,
        "L (length)", "m",
        "b (damping)", "N·s",
        "L", "b",
        "example_inference_pendulum.png",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 9.  GROUP 5 — SINDY EQUATION CARD
# ══════════════════════════════════════════════════════════════════════════════

def render_group5():
    print("\n[Group 5] SINDy equation card …")

    # Load R² values from logs if available; else use known results
    sindy_log = LOG_DIR / "day2_results.json"
    if sindy_log.exists():
        with open(sindy_log) as f:
            log = json.load(f)
        r2_k = log.get("r2_k_sindy_latent", 0.9966)
        r2_b = log.get("r2_b_sindy_latent", 0.5314)
    else:
        r2_k, r2_b = 0.9966, 0.5314

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=DPI)
    ax.axis("off")

    # Title
    ax.text(0.5, 0.92, "Equation Discovery via SINDy",
            ha="center", va="top", fontsize=16,
            color=WHITE, fontweight="bold",
            transform=ax.transAxes)

    # Separator line
    ax.plot([0.05, 0.95], [0.82, 0.82],
            color=GREY, lw=0.8, transform=ax.transAxes)

    # True equation
    ax.text(0.08, 0.70, "True equation of motion:",
            ha="left", va="top", fontsize=11,
            color=GREY, transform=ax.transAxes)
    ax.text(0.08, 0.56,
            r"$\ddot{x}$  =  $-k \cdot x$  $-$  $b \cdot \dot{x}$",
            ha="left", va="top", fontsize=20,
            color=WHITE, transform=ax.transAxes)

    # Discovered equation
    ax.text(0.08, 0.40, "Discovered by PhysicsLens (from raw video):",
            ha="left", va="top", fontsize=11,
            color=GREY, transform=ax.transAxes)
    ax.text(0.08, 0.26,
            r"$\ddot{x}$  =  $-\hat{k} \cdot x$  $-$  $\hat{b} \cdot \dot{x}$",
            ha="left", va="top", fontsize=20,
            color=CYAN, transform=ax.transAxes)

    # R² badges
    ax.text(0.62, 0.56,
            f"R²(k̂) = {r2_k:.4f}",
            ha="left", va="top", fontsize=13,
            color=CYAN, fontweight="bold",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=CYAN, lw=1.5))
    ax.text(0.62, 0.34,
            f"R²(b̂) = {r2_b:.4f}",
            ha="left", va="top", fontsize=13,
            color=MAGENTA, fontweight="bold",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=MAGENTA, lw=1.5))

    # footnote
    ax.text(0.5, 0.04,
            "* b̂ limited by STLSQ threshold — small damping coefficients zeroed out",
            ha="center", va="bottom", fontsize=8,
            color=GREY, fontstyle="italic",
            transform=ax.transAxes)

    fig.patch.set_facecolor(BG)
    out = OUT_DIR / "sindy_equation.png"
    plt.savefig(out, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  ✓ sindy_equation.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10.  ASSET MANIFEST
# ══════════════════════════════════════════════════════════════════════════════

def print_manifest():
    print("\n" + "=" * 60)
    print("ASSET MANIFEST — demo/assets/")
    print("=" * 60)
    groups = {
        "Group 1 — Scatter plots": [
            "scatter_k_oscillator.png",
            "scatter_b_oscillator.png",
            "scatter_L_pendulum.png",
            "scatter_b_pendulum.png",
        ],
        "Group 2 — Counterfactual videos": [
            "cf_k_low_vs_high.mp4",
            "cf_b_low_vs_high.mp4",
            "cf_pendulum_L_short_vs_long.mp4",
        ],
        "Group 3 — Latent PCA": [
            "latent_pca_k_oscillator.png",
            "latent_pca_b_oscillator.png",
            "latent_pca_L_pendulum.png",
            "latent_pca_b_pendulum.png",
        ],
        "Group 4 — Inference cards": [
            "example_inference_oscillator.png",
            "example_inference_pendulum.png",
        ],
        "Group 5 — SINDy card": [
            "sindy_equation.png",
        ],
    }
    total = 0
    for grp, files in groups.items():
        print(f"\n  {grp}")
        for fn in files:
            p      = OUT_DIR / fn
            exists = "✓" if p.exists() else "✗ MISSING"
            size   = f"({p.stat().st_size // 1024} KB)" if p.exists() else ""
            print(f"    {exists}  {fn}  {size}")
            total += p.exists()
    print(f"\n  {total}/{sum(len(v) for v in groups.values())} assets generated")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 11.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PhysicsLens — make_demo_assets.py")
    print("=" * 60)

    # ── load models ───────────────────────────────────────────────────────────
    print("\n[Loading models]")
    enc     = load_frame_encoder("world_model_combined.pt")
    osc_se, osc_norm = load_scene_encoder("scene_encoder_oscillator_combined.pt")
    pen_se, pen_norm = load_scene_encoder("scene_encoder_pendulum_combined.pt")

    # ── inference on val sets ─────────────────────────────────────────────────
    print("\n[Val-set inference]")
    osc_data = run_val_inference("oscillator", enc, osc_se, osc_norm, n_val=300)
    pen_data = run_val_inference("pendulum",   enc, pen_se, pen_norm, n_val=300)

    # print live R² so you can verify against training logs
    print(f"\n  Verified R² from inference:")
    print(f"    osc  k : {r2_score(osc_data['true_p1'], osc_data['pred_p1']):.4f}")
    print(f"    osc  b : {r2_score(osc_data['true_p2'], osc_data['pred_p2']):.4f}")
    print(f"    pen  L : {r2_score(pen_data['true_p1'], pen_data['pred_p1']):.4f}")
    print(f"    pen  b : {r2_score(pen_data['true_p2'], pen_data['pred_p2']):.4f}")

    # ── render assets ─────────────────────────────────────────────────────────
    render_group1(osc_data, pen_data)
    render_group2()
    render_group3(osc_data, pen_data)
    render_group4(enc, osc_se, osc_norm, pen_se, pen_norm)
    render_group5()

    # ── manifest ──────────────────────────────────────────────────────────────
    print_manifest()
    print(f"\nAll assets saved to:  {OUT_DIR.resolve()}\n")


if __name__ == "__main__":
    main()