import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


class OscillatorDataset(Dataset):
    def __init__(self, data_dir, n_frames=180, split='train', train_frac=0.8):
        self.data_dir = Path(data_dir)
        self.n_frames = n_frames

        all_files = sorted(self.data_dir.glob("*.npz"))
        assert len(all_files) > 0, f"No .npz files found in {data_dir}"

        n_train = int(len(all_files) * train_frac)
        if split == 'train':
            self.files = all_files[:n_train]
        else:
            self.files = all_files[n_train:]

        print(f"Dataset [{split}]: {len(self.files)} samples, {n_frames} frames each")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(str(self.files[idx]))
        video = d['video'][:self.n_frames].astype(np.float32) / 255.0
        video = torch.tensor(video).permute(0, 3, 1, 2)

        params = {
            'k':  torch.tensor(float(d['k']),  dtype=torch.float32),
            'b':  torch.tensor(float(d['b']),  dtype=torch.float32),
            'x0': torch.tensor(float(d['x0']), dtype=torch.float32),
            'v0': torch.tensor(float(d['v0']), dtype=torch.float32),
        }
        states = torch.tensor(d['states'][:self.n_frames], dtype=torch.float32)
        return video, params, states


def get_dataloaders(data_dir, n_frames=180, batch_size=8, num_workers=2):
    train_ds = OscillatorDataset(data_dir, n_frames=n_frames, split='train')
    val_ds   = OscillatorDataset(data_dir, n_frames=n_frames, split='val')

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader



# ══════════════════════════════════════════════════════════════════
# COMBINED DATASET — Day 2: oscillator + pendulum joint training
# ══════════════════════════════════════════════════════════════════
class CombinedPhysicsDataset(Dataset):
    """
    Loads oscillator + pendulum samples into one dataset.
    Frame encoder learns generic 'red-blob-on-dark-background' features
    across both scenarios.
    """
    def __init__(self, n_frames=180, split='train', train_frac=0.8):
        self.n_frames = n_frames

        osc_dir = Path("data/oscillator")
        pen_dir = Path("data/pendulum")
        assert osc_dir.exists(), f"Missing {osc_dir}"
        assert pen_dir.exists(), f"Missing {pen_dir}"

        osc_files = sorted(osc_dir.glob("*.npz"))
        pen_files = sorted(pen_dir.glob("*.npz"))

        n_osc_train = int(len(osc_files) * train_frac)
        n_pen_train = int(len(pen_files) * train_frac)

        if split == 'train':
            self.files     = osc_files[:n_osc_train] + pen_files[:n_pen_train]
            self.scenarios = (['oscillator'] * n_osc_train +
                              ['pendulum']   * n_pen_train)
        else:
            self.files     = osc_files[n_osc_train:] + pen_files[n_pen_train:]
            self.scenarios = (['oscillator'] * (len(osc_files) - n_osc_train) +
                              ['pendulum']   * (len(pen_files) - n_pen_train))

        n_osc = self.scenarios.count('oscillator')
        n_pen = self.scenarios.count('pendulum')
        print(f"Combined [{split}]: {len(self.files)} samples "
              f"({n_osc} osc + {n_pen} pen)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(str(self.files[idx]))
        scenario = self.scenarios[idx]

        video = d['video'][:self.n_frames].astype(np.float32) / 255.0
        video = torch.tensor(video).permute(0, 3, 1, 2)
        states = torch.tensor(d['states'][:self.n_frames], dtype=torch.float32)

        # Scenario tag as integer (0=oscillator, 1=pendulum)
        scenario_id = 0 if scenario == 'oscillator' else 1
        return video, states, torch.tensor(scenario_id, dtype=torch.long)


def get_combined_dataloaders(n_frames=180, batch_size=32, num_workers=2):
    train_ds = CombinedPhysicsDataset(n_frames=n_frames, split='train')
    val_ds   = CombinedPhysicsDataset(n_frames=n_frames, split='val')

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader