import torch
import numpy as np
import json
import time
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
sys.path.insert(0, '.')

from src.world_model import WorldModel
from src.dataset import get_dataloaders


def train(
    data_dir       = "data/oscillator",
    n_frames       = 20,
    batch_size     = 32,
    latent_dim     = 16,
    beta           = 0.001,
    pos_aux_weight = 20.0,
    fg_weight      = 50.0,
    lr             = 5e-4,
    n_epochs       = 30,
    save_dir       = "checkpoints",
    log_every      = 20,
    num_workers    = 2,
    early_stop_patience = 8
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"PhysicsLens v5 — Position-only probe")
    print(f"{'='*60}")
    print(f"Latent: {latent_dim}, beta={beta}, pos_aux={pos_aux_weight}, fg={fg_weight}")
    print(f"Epochs: {n_epochs}, batch={batch_size}")
    print(f"{'='*60}\n")

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    train_loader, val_loader = get_dataloaders(
        data_dir, n_frames=n_frames,
        batch_size=batch_size, num_workers=num_workers
    )

    model = WorldModel(
        latent_dim=latent_dim, beta=beta,
        pos_aux_weight=pos_aux_weight, fg_weight=fg_weight
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)

    history = []
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = {'recon': [], 'kl': [], 'dyn': [], 'pos': [], 'active_dims': []}
        t0 = time.time()

        for batch_idx, (video, params, states) in enumerate(train_loader):
            video  = video.to(device)
            states = states.to(device)

            loss, losses = model(video, states=states)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            for k in ['recon', 'kl', 'dyn', 'pos']:
                train_losses[k].append(losses[k])
            train_losses['active_dims'].append(losses['active_dims'])

            if (batch_idx + 1) % log_every == 0:
                ar  = np.mean(train_losses['recon'][-log_every:])
                ad  = np.mean(train_losses['dyn'][-log_every:])
                ap  = np.mean(train_losses['pos'][-log_every:])
                act = train_losses['active_dims'][-1]
                print(f"  E{epoch:02d} B{batch_idx+1:03d} | "
                      f"recon={ar:.4f} dyn={ad:.4f} pos={ap:.4f} active={act}/{latent_dim}")

        model.eval()
        val_losses = {'recon': [], 'kl': [], 'dyn': [], 'pos': []}
        val_active = []
        with torch.no_grad():
            for video, params, states in val_loader:
                video  = video.to(device)
                states = states.to(device)
                loss, losses = model(video, states=states)
                for k in ['recon', 'kl', 'dyn', 'pos']:
                    val_losses[k].append(losses[k])
                val_active.append(losses['active_dims'])

        train_avg = {k: float(np.mean(v)) for k, v in train_losses.items()}
        val_avg   = {k: float(np.mean(v)) for k, v in val_losses.items()}
        val_avg['active_dims'] = float(np.mean(val_active))
        elapsed   = time.time() - t0

        print(f"\n{'-'*60}")
        print(f"Epoch {epoch:02d}/{n_epochs} ({elapsed:.1f}s)")
        print(f"  Train: recon={train_avg['recon']:.4f} dyn={train_avg['dyn']:.4f} "
              f"pos={train_avg['pos']:.4f} active={train_avg['active_dims']:.1f}")
        print(f"  Val:   recon={val_avg['recon']:.4f} dyn={val_avg['dyn']:.4f} "
              f"pos={val_avg['pos']:.4f} active={val_avg['active_dims']:.1f}")
        print(f"  pos=1.0 means predicting mean; pos=0.1 means R²≈0.9")

        sched.step()

        val_score = val_avg['pos']
        if val_score < best_val_loss:
            best_val_loss = val_score
            epochs_since_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_loss':    best_val_loss,
                'config': {
                    'latent_dim': latent_dim,
                    'beta': beta,
                    'pos_aux_weight': pos_aux_weight,
                    'fg_weight': fg_weight
                }
            }, save_dir / "world_model_best.pt")
            print(f"  ✓ Saved best (val_pos={best_val_loss:.4f})")
        else:
            epochs_since_improvement += 1
            print(f"  No improvement ({epochs_since_improvement}/{early_stop_patience})")

        history.append({'epoch': epoch, 'train': train_avg, 'val': val_avg})
        print(f"{'-'*60}\n")

        if epochs_since_improvement >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    with open(save_dir / "train_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val pos loss: {best_val_loss:.4f}")
    print(f"R²(position) ≈ {1 - best_val_loss:.3f}")
    return model, history


if __name__ == "__main__":
    model, history = train()