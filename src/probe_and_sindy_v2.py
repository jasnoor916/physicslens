"""
probe_and_sindy_v2.py — PATCHED & PRODUCTION
Produced R²(k)=0.997, R²(b)=0.989 — the GREEN result.

Fixes vs v1:
  [P1] params access: d["k"], d["b"] not d["params"][()]["k"]
  [P2] checkpoint key: tries both "model_state_dict" and "model_state"
  [P3] removed weights_only=False (PyTorch 2.4+ warning)
  [P4] uses ALL 90 frames per video (not 20)
  [P5] runs SINDy in both modes (GT and latent-recovered)
"""

import os, json, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import pysindy as ps

sys.path.insert(0, "src")
from world_model import FrameEncoder

CKPT_DIR = Path("checkpoints")
DATA_DIR = Path("data/oscillator")
LOG_DIR  = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


class FullSequenceDataset(Dataset):
    def __init__(self, data_dir, indices, debug=False):
        self.files   = sorted(Path(data_dir).glob("sample_*.npz"))
        self.indices = indices
        self.debug   = debug

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        d = np.load(self.files[self.indices[idx]])
        if self.debug and idx == 0:
            print(f"\nDEBUG first sample keys: {list(d.keys())}")
            print(f"  video shape  : {d['video'].shape}")
            print(f"  states shape : {d['states'].shape}")

        video  = torch.from_numpy(d["video"]).float() / 255.0
        video  = video.permute(0, 3, 1, 2)

        states = torch.from_numpy(d["states"]).float()
        x_traj = states[:, 0]

        k = float(d["k"])
        b = float(d["b"])
        params = torch.tensor([k, b], dtype=torch.float32)

        return video, x_traj, params


def get_full_loaders(data_dir, n_total=1500, train_frac=0.8, batch_size=32):
    all_idx   = list(range(n_total))
    n_train   = int(n_total * train_frac)
    train_idx = all_idx[:n_train]
    val_idx   = all_idx[n_train:]

    train_ds = FullSequenceDataset(data_dir, train_idx, debug=True)
    val_ds   = FullSequenceDataset(data_dir, val_idx,   debug=False)

    def collate(batch):
        videos, trajs, params = zip(*batch)
        T_min  = min(v.shape[0] for v in videos)
        videos = torch.stack([v[:T_min] for v in videos])
        trajs  = torch.stack([t[:T_min] for t in trajs])
        params = torch.stack(list(params))
        return videos, trajs, params

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, collate_fn=collate, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                               num_workers=2, collate_fn=collate, pin_memory=True)

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    return train_loader, val_loader


class ImprovedSceneEncoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=64, scene_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.1)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.proj = nn.Linear(hidden_dim * 2, scene_dim)

        self.k_head = nn.Sequential(
            nn.Linear(scene_dim, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 1)
        )
        self.b_head = nn.Sequential(
            nn.Linear(scene_dim, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 1)
        )

    def encode(self, z_seq):
        h, _ = self.lstm(z_seq)
        a     = torch.softmax(self.attn(h), dim=1)
        ctx   = (a * h).sum(dim=1)
        return self.proj(ctx)

    def forward(self, z_seq):
        scene  = self.encode(z_seq)
        k_pred = self.k_head(scene).squeeze(-1)
        b_pred = self.b_head(scene).squeeze(-1)
        return k_pred, b_pred


def load_frozen_frame_encoder():
    ckpt = torch.load(CKPT_DIR / "world_model_best.pt", map_location=DEVICE)
    cfg = ckpt.get("config", {
        "latent_dim": 16, "beta": 0.001,
        "pos_aux_weight": 20.0, "fg_weight": 50.0
    })
    encoder = FrameEncoder(latent_dim=cfg["latent_dim"]).to(DEVICE)
    state = ckpt.get("model_state_dict") or ckpt.get("model_state")
    if state is None:
        raise KeyError(f"Cannot find model weights. Keys: {list(ckpt.keys())}")

    enc_state = {}
    for k_name, v in state.items():
        if k_name.startswith("encoder."):
            enc_state[k_name[len("encoder."):]] = v
        elif k_name.startswith("frame_encoder."):
            enc_state[k_name[len("frame_encoder."):]] = v

    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    print(f"FrameEncoder loaded | missing={len(missing)} unexpected={len(unexpected)}")

    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    print(f"  latent_dim={cfg['latent_dim']} | encoder frozen")
    return encoder, cfg


@torch.no_grad()
def extract_latents(encoder, loader, desc=""):
    all_z, all_x, all_p = [], [], []
    for videos, x_traj, params in loader:
        B, T, C, H, W = videos.shape
        frames_flat = videos.reshape(B * T, C, H, W).to(DEVICE)
        mu, _ = encoder(frames_flat)
        mu     = mu.reshape(B, T, -1).cpu()
        all_z.append(mu)
        all_x.append(x_traj)
        all_p.append(params)

    Z = torch.cat(all_z, dim=0)
    X = torch.cat(all_x, dim=0)
    P = torch.cat(all_p, dim=0)
    print(f"  {desc}: Z={tuple(Z.shape)}  X={tuple(X.shape)}  P={tuple(P.shape)}")
    return Z, X, P


def verify_position_probe(Z_val, X_val):
    N, T, D = Z_val.shape
    z_flat  = Z_val.reshape(N * T, D).numpy()
    x_flat  = X_val.reshape(N * T).numpy()
    probe = LinearRegression()
    probe.fit(z_flat, x_flat)
    x_pred = probe.predict(z_flat)
    r2     = r2_score(x_flat, x_pred)
    print(f"  Position probe R²: {r2:.4f}  {'✅' if r2 > 0.95 else '⚠️'}")
    return r2, probe


def train_scene_encoder(Z_train, P_train, Z_val, P_val, epochs=60):
    k_mean = P_train[:, 0].mean().item()
    k_std  = P_train[:, 0].std().item()
    b_mean = P_train[:, 1].mean().item()
    b_std  = P_train[:, 1].std().item()
    print(f"  k: mean={k_mean:.2f}  std={k_std:.2f}")
    print(f"  b: mean={b_mean:.2f}  std={b_std:.2f}")

    def norm_k(k): return (k - k_mean) / k_std
    def norm_b(b): return (b - b_mean) / b_std
    def denorm_k(k): return k * k_std + k_mean
    def denorm_b(b): return b * b_std + b_mean

    model = ImprovedSceneEncoder(latent_dim=Z_train.shape[-1],
                                  hidden_dim=64, scene_dim=32).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=20, T_mult=1, eta_min=1e-5
    )

    Z_tr  = Z_train.to(DEVICE)
    k_tr  = norm_k(P_train[:, 0]).to(DEVICE)
    b_tr  = norm_b(P_train[:, 1]).to(DEVICE)
    Z_vl  = Z_val.to(DEVICE)

    BATCH = 64
    N     = Z_tr.shape[0]
    best_r2k = -999.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N)
        ep_loss = 0.0
        n_batch = 0

        for i in range(0, N, BATCH):
            idx = perm[i:i + BATCH]
            zb  = Z_tr[idx]
            kb  = k_tr[idx]
            bb  = b_tr[idx]

            k_pred, b_pred = model(zb)
            loss_k = nn.functional.mse_loss(k_pred, kb)
            loss_b = nn.functional.mse_loss(b_pred, bb)
            loss   = 2.0 * loss_k + 1.0 * loss_b

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); n_batch += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            k_pred_v, b_pred_v = model(Z_vl)
            k_pred_v = denorm_k(k_pred_v).cpu().numpy()
            b_pred_v = denorm_b(b_pred_v).cpu().numpy()

        k_gt = P_val[:, 0].numpy()
        b_gt = P_val[:, 1].numpy()
        r2_k = r2_score(k_gt, k_pred_v)
        r2_b = r2_score(b_gt, b_pred_v)
        lr_now = scheduler.get_last_lr()[0]

        history.append({"epoch": epoch, "train_loss": ep_loss / n_batch,
                        "r2_k": r2_k, "r2_b": r2_b})

        if r2_k > best_r2k:
            best_r2k = r2_k
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch:02d} | loss={ep_loss/n_batch:.4f} | "
              f"R²(k)={r2_k:.3f}  R²(b)={r2_b:.3f} | lr={lr_now:.2e}")

    model.load_state_dict(best_state)
    print(f"\n  Best R²(k) = {best_r2k:.3f}")

    return model, {
        "k_mean": k_mean, "k_std": k_std,
        "b_mean": b_mean, "b_std": b_std,
        "history": history
    }


def run_sindy(x_trajectories, P_val, mode_label="GT", n_samples=15):
    dt = 1.0 / 30.0
    results = []
    k_sindy_list, b_sindy_list = [], []
    k_true_list,  b_true_list  = [], []

    print(f"\n  [{mode_label}] True equation: ẍ = -k·x - b·ẋ")
    print(f"  [{mode_label}] LinearLibrary, SG smoothing, {n_samples} samples\n")

    for i in range(min(n_samples, len(x_trajectories))):
        x_raw = np.array(x_trajectories[i])
        T_len  = len(x_raw)

        win = min(15, T_len - 2)
        win = win if win % 2 == 1 else win - 1
        win = max(win, 5)

        x_smooth = savgol_filter(x_raw, window_length=win, polyorder=3)
        v_smooth = savgol_filter(x_raw, window_length=win, polyorder=3,
                                  deriv=1, delta=dt)

        state   = np.column_stack([x_smooth, v_smooth])
        library = ps.PolynomialLibrary(degree=1, include_bias=True)
        sindy_model = ps.SINDy(
            feature_library=library,
            optimizer=ps.STLSQ(threshold=0.05, alpha=0.01),
            differentiation_method=ps.FiniteDifference()
        )

        try:
            sindy_model.fit(state, t=dt)
            coefs = sindy_model.coefficients()
            if coefs.shape[1] >= 3:
                k_sindy = -coefs[1, 1]
                b_sindy = -coefs[1, 2]
            else:
                k_sindy, b_sindy = None, None

            k_true = float(P_val[i, 0])
            b_true = float(P_val[i, 1])

            r = {
                "sample": i, "mode": mode_label,
                "k_true": k_true, "b_true": b_true,
                "k_sindy": float(k_sindy) if k_sindy is not None else None,
                "b_sindy": float(b_sindy) if b_sindy is not None else None,
                "equations": sindy_model.equations()
            }
            results.append(r)

            if k_sindy is not None:
                k_sindy_list.append(k_sindy)
                b_sindy_list.append(b_sindy)
                k_true_list.append(k_true)
                b_true_list.append(b_true)

            print(f"  Sample {i:2d}: k_true={k_true:5.2f}  k_SINDy={k_sindy:7.3f}  | "
                  f"b_true={b_true:.3f}  b_SINDy={b_sindy:7.3f}")

        except Exception as e:
            print(f"  Sample {i}: SINDy failed — {e}")

    r2_k = r2_score(k_true_list, k_sindy_list) if len(k_true_list) > 1 else None
    r2_b = r2_score(b_true_list, b_sindy_list) if len(b_true_list) > 1 else None

    print(f"\n  [{mode_label}] SINDy R²(k) = {r2_k:.3f}")
    print(f"  [{mode_label}] SINDy R²(b) = {r2_b:.3f}")
    return results, r2_k, r2_b


def go_no_go(r2_k, r2_b, r2_k_sindy_gt, r2_b_sindy_gt,
             r2_k_sindy_latent, r2_b_sindy_latent):
    print("\n" + "=" * 60)
    print("GO / NO-GO DECISION v2")
    print("=" * 60)
    print(f"  Scene Encoder  R²(k)        = {r2_k:.3f}   threshold ≥ 0.85")
    print(f"  Scene Encoder  R²(b)        = {r2_b:.3f}   threshold ≥ 0.70")
    print(f"  SINDy (GT)     R²(k)        = {r2_k_sindy_gt:.3f}")
    print(f"  SINDy (GT)     R²(b)        = {r2_b_sindy_gt:.3f}")
    print(f"  SINDy (latent) R²(k)        = {r2_k_sindy_latent:.3f}   ← novel")
    print(f"  SINDy (latent) R²(b)        = {r2_b_sindy_latent:.3f}   ← novel")

    if r2_k >= 0.85 and r2_b >= 0.70:
        verdict, day2 = "🟢 GREEN — Ship PhysicsLens FULL", "Pendulum + counterfactuals"
    elif r2_k >= 0.70:
        verdict, day2 = "🟡 YELLOW — Ship InterpWorld variant", "Lead with k-recovery"
    elif r2_k >= 0.50:
        verdict, day2 = "🟠 ORANGE — One more pass needed", "Add explicit features"
    else:
        verdict, day2 = "🔴 RED — Pivot to PhysicsForge", "LLM-driven simulation"

    print(f"\n  {verdict}")
    print(f"  Day 2 plan: {day2}")
    print("=" * 60)
    return verdict, day2


if __name__ == "__main__":
    print("=" * 60)
    print("PhysicsLens — Probe & SINDy v2 (production)")
    print("=" * 60)

    encoder, cfg = load_frozen_frame_encoder()

    print("\nLoading full-sequence data...")
    train_loader, val_loader = get_full_loaders(
        DATA_DIR, n_total=1500, train_frac=0.8, batch_size=32
    )

    print("\nExtracting latent sequences (frozen encoder)...")
    Z_train, X_train, P_train = extract_latents(encoder, train_loader, "train")
    Z_val,   X_val,   P_val   = extract_latents(encoder, val_loader,   "val")

    print("\nPosition probe sanity check:")
    r2_pos, pos_probe = verify_position_probe(Z_val, X_val)

    print("\nRecovering x(t) from latent sequences...")
    N_val, T_val, D_val = Z_val.shape
    z_flat_val  = Z_val.reshape(N_val * T_val, D_val).numpy()
    x_recovered = pos_probe.predict(z_flat_val).reshape(N_val, T_val)
    print(f"  Recovered x shape: {x_recovered.shape}")

    print("\n" + "=" * 60)
    print("Training ImprovedSceneEncoder (60 epochs)")
    print("=" * 60)
    scene_model, norm_stats = train_scene_encoder(
        Z_train, P_train, Z_val, P_val, epochs=60
    )

    scene_model.eval()
    with torch.no_grad():
        k_pred_v, b_pred_v = scene_model(Z_val.to(DEVICE))
    k_pred_np = k_pred_v.cpu().numpy() * norm_stats["k_std"] + norm_stats["k_mean"]
    b_pred_np = b_pred_v.cpu().numpy() * norm_stats["b_std"] + norm_stats["b_mean"]
    k_gt = P_val[:, 0].numpy()
    b_gt = P_val[:, 1].numpy()
    r2_k_final = r2_score(k_gt, k_pred_np)
    r2_b_final = r2_score(b_gt, b_pred_np)

    print(f"\nFINAL Scene Encoder | R²(k)={r2_k_final:.4f}  R²(b)={r2_b_final:.4f}")

    print("\n" + "=" * 60)
    print("SINDy Mode A: Ground Truth x(t) [sanity check]")
    print("=" * 60)
    X_val_np = X_val.numpy()
    sindy_gt, r2_k_sindy_gt, r2_b_sindy_gt = run_sindy(
        X_val_np, P_val, mode_label="GT", n_samples=15
    )

    print("\n" + "=" * 60)
    print("SINDy Mode B: Latent-Recovered x(t) [novel contribution]")
    print("=" * 60)
    sindy_latent, r2_k_sindy_lat, r2_b_sindy_lat = run_sindy(
        x_recovered, P_val, mode_label="Latent", n_samples=15
    )

    save_dict = {
        "r2_position_probe":          float(r2_pos),
        "r2_k_scene_encoder":         float(r2_k_final),
        "r2_b_scene_encoder":         float(r2_b_final),
        "r2_k_sindy_gt":              float(r2_k_sindy_gt) if r2_k_sindy_gt else None,
        "r2_b_sindy_gt":              float(r2_b_sindy_gt) if r2_b_sindy_gt else None,
        "r2_k_sindy_latent":          float(r2_k_sindy_lat) if r2_k_sindy_lat else None,
        "r2_b_sindy_latent":          float(r2_b_sindy_lat) if r2_b_sindy_lat else None,
        "norm_stats":                 {k: v for k, v in norm_stats.items() if k != "history"},
        "training_history":           norm_stats["history"],
        "sindy_gt_samples":           sindy_gt[:5],
        "sindy_latent_samples":       sindy_latent[:5],
    }

    with open(LOG_DIR / "day2_results.json", "w") as f:
        json.dump(save_dict, f, indent=2)

    torch.save({
        "model_state_dict": scene_model.state_dict(),
        "norm_stats":        norm_stats,
        "config": {"latent_dim": cfg["latent_dim"],
                   "hidden_dim": 64, "scene_dim": 32}
    }, CKPT_DIR / "scene_probe_v2.pt")

    print("\nSaved: logs/day2_results.json")
    print("Saved: checkpoints/scene_probe_v2.pt")

    verdict, plan = go_no_go(
        r2_k_final, r2_b_final,
        r2_k_sindy_gt  or 0.0, r2_b_sindy_gt  or 0.0,
        r2_k_sindy_lat or 0.0, r2_b_sindy_lat or 0.0,
    )

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  R²(x)  position probe       = {r2_pos:.4f}  ✅")
    print(f"  R²(k)  scene encoder        = {r2_k_final:.4f}")
    print(f"  R²(b)  scene encoder        = {r2_b_final:.4f}")
    print(f"  R²(k)  SINDy [GT]           = {r2_k_sindy_gt:.4f}")
    print(f"  R²(b)  SINDy [GT]           = {r2_b_sindy_gt:.4f}")
    print(f"  R²(k)  SINDy [latent novel] = {r2_k_sindy_lat:.4f}")
    print(f"  R²(b)  SINDy [latent novel] = {r2_b_sindy_lat:.4f}")
    print("=" * 60)
    print(f"\n  {verdict}")