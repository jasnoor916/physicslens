"""
train_scene_encoder.py — Reusable scene encoder trainer (any scenario)

Usage:
    python src/train_scene_encoder.py --scenario oscillator
    python src/train_scene_encoder.py --scenario pendulum
    python src/train_scene_encoder.py --scenario pendulum --ckpt world_model_combined.pt
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import r2_score

sys.path.insert(0, "src")
from world_model import FrameEncoder

CKPT_DIR = Path("checkpoints")
LOG_DIR  = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Scenario configs ──────────────────────────────────────────────
SCENARIO_CONFIG = {
    "oscillator": {
        "data_dir":   "data/oscillator",
        "param_keys": ["k", "b"],
        "labels":     ["k (spring)", "b (damping)"],
        "equation":   "ẍ = -k·x - b·ẋ",
    },
    "pendulum": {
        "data_dir":   "data/pendulum",
        "param_keys": ["L", "b"],
        "labels":     ["L (length)", "b (damping)"],
        "equation":   "θ̈ = -(g/L)·sin(θ) - b·θ̇",
    },
}


# ── Architecture ──────────────────────────────────────────────────
class SceneEncoderV2(nn.Module):
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
                nn.Linear(32, 1)
            )
        self.p1_head = _head()
        self.p2_head = _head()

    def forward(self, z):
        h, _ = self.lstm(z)
        a    = torch.softmax(self.attn(h), dim=1)
        ctx  = (a * h).sum(dim=1)
        s    = self.proj(ctx)
        return self.p1_head(s).squeeze(-1), self.p2_head(s).squeeze(-1)


# ── Dataset ───────────────────────────────────────────────────────
from torch.utils.data import Dataset, DataLoader

class PhysicsDataset(Dataset):
    def __init__(self, data_dir, param_keys, indices):
        self.files       = sorted(Path(data_dir).glob("sample_*.npz"))
        self.param_keys  = param_keys
        self.indices     = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        d = np.load(self.files[self.indices[idx]])
        video  = torch.from_numpy(d["video"]).float() / 255.0
        video  = video.permute(0, 3, 1, 2)
        params = torch.tensor([float(d[k]) for k in self.param_keys],
                              dtype=torch.float32)
        return video, params


def get_loaders(data_dir, param_keys, n_total=1500, train_frac=0.8, batch_size=32):
    n_train  = int(n_total * train_frac)
    train_ds = PhysicsDataset(data_dir, param_keys, list(range(n_train)))
    val_ds   = PhysicsDataset(data_dir, param_keys, list(range(n_train, n_total)))

    def collate(batch):
        videos, params = zip(*batch)
        T = min(v.shape[0] for v in videos)
        return torch.stack([v[:T] for v in videos]), torch.stack(list(params))

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=2, collate_fn=collate, pin_memory=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate, pin_memory=True))


# ── Frozen frame encoder ──────────────────────────────────────────
def load_frozen_encoder(latent_dim=16, ckpt_name="world_model_best.pt"):
    ckpt_path = CKPT_DIR / ckpt_name
    print(f"Loading frame encoder from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("model_state_dict") or ckpt.get("model_state")

    encoder = FrameEncoder(latent_dim=latent_dim).to(DEVICE)
    enc_state = {
        k.replace("encoder.", "").replace("frame_encoder.", ""): v
        for k, v in state.items()
        if k.startswith("encoder.") or k.startswith("frame_encoder.")
    }
    encoder.load_state_dict(enc_state, strict=False)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    return encoder


@torch.no_grad()
def extract_latents(encoder, loader):
    Zs, Ps = [], []
    for videos, params in loader:
        B, T, C, H, W = videos.shape
        flat = videos.reshape(B * T, C, H, W).to(DEVICE)
        mu, _ = encoder(flat)
        Zs.append(mu.reshape(B, T, -1).cpu())
        Ps.append(params)
    return torch.cat(Zs), torch.cat(Ps)


# ── Training loop ─────────────────────────────────────────────────
def train(scenario, epochs=60, batch_size=64, lr=2e-3,
          ckpt_name="world_model_best.pt"):
    cfg = SCENARIO_CONFIG[scenario]
    print(f"\n{'='*60}")
    print(f"Training scene encoder: {scenario}")
    print(f"  Equation:        {cfg['equation']}")
    print(f"  Targets:         {cfg['labels']}")
    print(f"  Frame encoder:   {ckpt_name}")
    print(f"{'='*60}")

    train_loader, val_loader = get_loaders(
        cfg["data_dir"], cfg["param_keys"], batch_size=32
    )

    encoder = load_frozen_encoder(ckpt_name=ckpt_name)
    print("Frame encoder loaded & frozen ✓")

    print("Extracting latents...")
    Z_tr, P_tr = extract_latents(encoder, train_loader)
    Z_vl, P_vl = extract_latents(encoder, val_loader)
    print(f"  Z_tr={tuple(Z_tr.shape)}  Z_vl={tuple(Z_vl.shape)}")

    p1_mean, p1_std = P_tr[:, 0].mean().item(), P_tr[:, 0].std().item()
    p2_mean, p2_std = P_tr[:, 1].mean().item(), P_tr[:, 1].std().item()
    print(f"  {cfg['labels'][0]}: mean={p1_mean:.3f} std={p1_std:.3f}")
    print(f"  {cfg['labels'][1]}: mean={p2_mean:.3f} std={p2_std:.3f}")

    p1_tr_n = ((P_tr[:, 0] - p1_mean) / p1_std).to(DEVICE)
    p2_tr_n = ((P_tr[:, 1] - p2_mean) / p2_std).to(DEVICE)
    Z_tr    = Z_tr.to(DEVICE)
    Z_vl    = Z_vl.to(DEVICE)

    model = SceneEncoderV2(latent_dim=16, hidden_dim=64, scene_dim=32).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=20, T_mult=1, eta_min=1e-5
    )

    N = Z_tr.shape[0]
    best_r2 = -999.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N)
        ep_loss, n_b = 0.0, 0

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            p1_b, p2_b = model(Z_tr[idx])
            loss = (2.0 * nn.functional.mse_loss(p1_b, p1_tr_n[idx]) +
                    1.0 * nn.functional.mse_loss(p2_b, p2_tr_n[idx]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); n_b += 1

        sched.step()
        model.eval()
        with torch.no_grad():
            p1_v, p2_v = model(Z_vl)
        p1_pred = p1_v.cpu().numpy() * p1_std + p1_mean
        p2_pred = p2_v.cpu().numpy() * p2_std + p2_mean
        r2_p1   = r2_score(P_vl[:, 0].numpy(), p1_pred)
        r2_p2   = r2_score(P_vl[:, 1].numpy(), p2_pred)

        history.append({"epoch": epoch, "loss": ep_loss/n_b,
                        "r2_p1": r2_p1, "r2_p2": r2_p2})

        if r2_p1 > best_r2:
            best_r2 = r2_p1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch:02d} | loss={ep_loss/n_b:.4f} | "
              f"R²({cfg['labels'][0][:1]})={r2_p1:.3f}  "
              f"R²({cfg['labels'][1][:1]})={r2_p2:.3f}")

    model.load_state_dict(best_state)

    # Save with suffix if combined ckpt was used
    suffix = "_combined" if "combined" in ckpt_name else ""
    save_path = CKPT_DIR / f"scene_encoder_{scenario}{suffix}.pt"

    norm_stats = {
        "p1_mean": p1_mean, "p1_std": p1_std,
        "p2_mean": p2_mean, "p2_std": p2_std,
        "scenario": scenario,
        "param_labels": cfg["labels"],
        "frame_encoder_ckpt": ckpt_name,
    }
    torch.save({
        "model_state_dict": model.state_dict(),
        "norm_stats":        norm_stats,
        "config": {"latent_dim": 16, "hidden_dim": 64, "scene_dim": 32}
    }, save_path)

    log_name = f"scene_encoder_{scenario}{suffix}.json"
    with open(LOG_DIR / log_name, "w") as f:
        json.dump({
            "scenario": scenario,
            "frame_encoder_ckpt": ckpt_name,
            "best_r2_p1": float(best_r2),
            "final_r2_p2": float(r2_p2),
            "history": history
        }, f, indent=2)

    print(f"\nBest R²({cfg['labels'][0][:1]}) = {best_r2:.4f}")
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="oscillator",
                        choices=list(SCENARIO_CONFIG.keys()))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--ckpt", default="world_model_best.pt",
                        help="Frame encoder checkpoint to load (e.g. world_model_combined.pt)")
    args = parser.parse_args()
    train(args.scenario, epochs=args.epochs, ckpt_name=args.ckpt)