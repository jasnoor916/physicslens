import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


class OscillatorDataset(Dataset):
    def __init__(self, data_dir, n_frames=20, split='train', train_frac=0.8):
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


def get_dataloaders(data_dir, n_frames=20, batch_size=8, num_workers=2):
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