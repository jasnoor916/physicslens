"""
Day 1 Decision Script — v6 with scene encoder + train scene encoder via probe.
This is the v1 script — produced R²(k)=0.494 (ORANGE).
Kept for reference. Use probe_and_sindy_v2.py for the production pipeline.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import pysindy as ps
import json
import sys
sys.path.insert(0, '.')

for mod in list(sys.modules.keys()):
    if 'src.' in mod:
        del sys.modules[mod]

from src.world_model import WorldModel
from src.dataset import OscillatorDataset
from torch.utils.data import DataLoader, ConcatDataset


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt['config']
    model = WorldModel(
        latent_dim=cfg['latent_dim'],
        beta=cfg['beta'],
        pos_aux_weight=cfg['pos_aux_weight'],
        fg_weight=cfg.get('fg_weight', 50.0)
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt['model_state'], strict=False)
    print(f"Loaded model from {ckpt_path}")
    print(f"  Missing keys (will train): {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    return model


def train_scene_encoder(model, data_dir, device, n_epochs=15, batch_size=32, lr=1e-3):
    print(f"\n{'='*60}")
    print("Training scene encoder (frozen frame encoder)")
    print(f"{'='*60}")

    for p in model.parameters():
        p.requires_grad = False
    for p in model.scene_encoder.parameters():
        p.requires_grad = True

    probe_head = nn.Sequential(
        nn.Linear(model.scene_latent_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    ).to(device)

    params = list(model.scene_encoder.parameters()) + list(probe_head.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    train_ds = OscillatorDataset(data_dir, n_frames=20, split='train')
    val_ds = OscillatorDataset(data_dir, n_frames=20, split='val')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    all_k, all_b = [], []
    for _, params_, _ in train_loader:
        all_k.append(params_['k'])
        all_b.append(params_['b'])
    K_mean, K_std = torch.cat(all_k).mean().item(), torch.cat(all_k).std().item()
    B_mean, B_std = torch.cat(all_b).mean().item(), torch.cat(all_b).std().item()
    print(f"K stats: mean={K_mean:.2f}, std={K_std:.2f}")
    print(f"B stats: mean={B_mean:.2f}, std={B_std:.2f}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        probe_head.train()
        train_loss = []

        for video, params_, _ in train_loader:
            video = video.to(device)
            k = ((params_['k'] - K_mean) / K_std).to(device)
            b = ((params_['b'] - B_mean) / B_std).to(device)

            with torch.no_grad():
                mu, _, _ = model.encode_video(video)

            z_scene = model.scene_encoder(mu)
            pred = probe_head(z_scene)

            loss_k = nn.functional.mse_loss(pred[:, 0], k)
            loss_b = nn.functional.mse_loss(pred[:, 1], b)
            loss = loss_k + loss_b

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss.append(loss.item())

        model.eval()
        probe_head.eval()
        val_k_pred, val_k_gt, val_b_pred, val_b_gt = [], [], [], []
        with torch.no_grad():
            for video, params_, _ in val_loader:
                video = video.to(device)
                mu, _, _ = model.encode_video(video)
                z_scene = model.scene_encoder(mu)
                pred = probe_head(z_scene)

                val_k_pred.append(pred[:, 0].cpu() * K_std + K_mean)
                val_k_gt.append(params_['k'])
                val_b_pred.append(pred[:, 1].cpu() * B_std + B_mean)
                val_b_gt.append(params_['b'])

        val_k_pred = torch.cat(val_k_pred).numpy()
        val_k_gt = torch.cat(val_k_gt).numpy()
        val_b_pred = torch.cat(val_b_pred).numpy()
        val_b_gt = torch.cat(val_b_gt).numpy()

        r2_k = r2_score(val_k_gt, val_k_pred)
        r2_b = r2_score(val_b_gt, val_b_pred)

        print(f"  Epoch {epoch:02d} | train_loss={np.mean(train_loss):.4f} | "
              f"R²(k)={r2_k:.3f}  R²(b)={r2_b:.3f}")

    return probe_head, (K_mean, K_std, B_mean, B_std), (r2_k, r2_b, val_k_pred, val_k_gt, val_b_pred, val_b_gt)


def run_sindy(X_pred, States, n_samples=5):
    print(f"\n{'='*60}")
    print("SINDy: Discover equation of motion from recovered x(t)")
    print(f"{'='*60}")

    dt = 1.0 / 30
    results = []

    for i in range(min(n_samples, len(X_pred))):
        x_recovered = X_pred[i]
        x_gt = States[i, :, 0]
        v_gt = States[i, :, 1]
        v_recovered = np.gradient(x_recovered, dt)

        traj = np.stack([x_recovered, v_recovered], axis=-1)

        feature_lib = ps.PolynomialLibrary(degree=2)
        optimizer = ps.STLSQ(threshold=0.1, alpha=0.05)
        sindy_model = ps.SINDy(
            feature_library=feature_lib,
            optimizer=optimizer,
        )

        try:
            sindy_model.fit(traj, t=dt)
            equations = sindy_model.equations(precision=3)
            results.append({'sample': i, 'equations': equations})

            print(f"\n  Sample {i}:")
            print(f"    Recovered x[0:3]: {x_recovered[:3].round(3)}")
            print(f"    Discovered:")
            for j, eq in enumerate(equations):
                var = ['x', 'v'][j]
                print(f"      d{var}/dt = {eq}")
        except Exception as e:
            print(f"  Sample {i}: SINDy failed — {e}")

    return results


def extract_position_predictions(model, data_dir, device, batch_size=32):
    train_ds = OscillatorDataset(data_dir, n_frames=20, split='train')
    val_ds = OscillatorDataset(data_dir, n_frames=20, split='val')
    all_ds = ConcatDataset([train_ds, val_ds])
    loader = DataLoader(all_ds, batch_size=batch_size, shuffle=False)

    all_x_pred, all_states = [], []
    with torch.no_grad():
        for video, _, states in loader:
            video = video.to(device)
            x_pred = model.predict_position(video)
            all_x_pred.append(x_pred.cpu().numpy())
            all_states.append(states.numpy())

    return np.concatenate(all_x_pred), np.concatenate(all_states)


def decision_gate(r2_k, r2_b, sindy_results):
    sindy_ok = False
    if sindy_results:
        eqs = sindy_results[0].get('equations', [])
        if len(eqs) >= 2:
            sindy_ok = any(('x' in eq and len(eq) > 5) for eq in eqs[1:])

    print(f"\n{'='*60}")
    print("GO / NO-GO DECISION")
    print(f"{'='*60}")
    print(f"R²(k) = {r2_k:.3f}   threshold: 0.85")
    print(f"R²(b) = {r2_b:.3f}   threshold: 0.70")
    print(f"SINDy meaningful: {sindy_ok}")

    if r2_k >= 0.85 and r2_b >= 0.70 and sindy_ok:
        decision = "GREEN"
        print(f"\n✅ GREEN — SHIP FULL PhysicsLens")
    elif r2_k >= 0.7 or (r2_k >= 0.6 and sindy_ok):
        decision = "YELLOW"
        print(f"\n⚠️ YELLOW — Ship InterpWorld (probing version)")
    elif r2_k >= 0.4:
        decision = "ORANGE"
        print(f"\n🟠 ORANGE — Real signal, refine architecture")
    else:
        decision = "RED"
        print(f"\n❌ RED — Pivot to PhysicsForge")

    print(f"{'='*60}\n")
    return decision


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model("checkpoints/world_model_best.pt", device)

    probe_head, stats, results = train_scene_encoder(
        model, "data/oscillator", device, n_epochs=15
    )
    r2_k, r2_b, k_pred, k_gt, b_pred, b_gt = results

    print(f"\n{'='*60}")
    print("FINAL Probe Predictions (Sample)")
    print(f"{'='*60}")
    print(f"\nSpring constant k:")
    for gt, pr in zip(k_gt[:5], k_pred[:5]):
        print(f"  GT={gt:.3f}  Pred={pr:.3f}  Err={abs(gt-pr):.3f}")
    print(f"\nDamping b:")
    for gt, pr in zip(b_gt[:5], b_pred[:5]):
        print(f"  GT={gt:.3f}  Pred={pr:.3f}  Err={abs(gt-pr):.3f}")

    X_pred, States = extract_position_predictions(model, "data/oscillator", device)
    sindy_results = run_sindy(X_pred, States, n_samples=5)

    decision = decision_gate(r2_k, r2_b, sindy_results)

    save_data = {
        'decision': decision,
        'r2_k': float(r2_k),
        'r2_b': float(r2_b),
        'sindy_results': sindy_results
    }
    import os
    os.makedirs("logs", exist_ok=True)
    with open("logs/day1_results.json", 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to logs/day1_results.json")

    torch.save({
        'scene_encoder_state': model.scene_encoder.state_dict(),
        'probe_head_state': probe_head.state_dict(),
        'stats': stats,
    }, "checkpoints/scene_probe.pt")
    print("Scene encoder + probe saved to checkpoints/scene_probe.pt")