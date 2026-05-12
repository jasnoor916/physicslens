import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FrameEncoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 256), nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim * 2)
        )

    def forward(self, x):
        h = self.conv(x)
        h = self.fc(h)
        mu     = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]
        logvar = torch.clamp(logvar, -10, 2)
        return mu, logvar

    def sample(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu


class FrameDecoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256 * 4 * 4), nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 256, 4, 4)
        return self.deconv(h)


class LatentDynamics(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.proj_in = nn.Linear(latent_dim, hidden_dim)
        self.pos_emb = nn.Embedding(200, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj_out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_seq):
        B, T, D = z_seq.shape
        positions = torch.arange(T, device=z_seq.device)
        h = self.proj_in(z_seq) + self.pos_emb(positions).unsqueeze(0)
        mask = torch.triu(torch.ones(T, T, device=z_seq.device), diagonal=1).bool()
        h = self.transformer(h, mask=mask)
        return self.proj_out(h)


class SceneEncoder(nn.Module):
    """
    Aggregates per-frame latents (z_1...z_T) into a scene-level latent z_scene.
    z_scene encodes properties of the *entire trajectory* — perfect for k, b.
    Bidirectional LSTM + attention pooling.
    """
    def __init__(self, frame_latent_dim=16, scene_latent_dim=32, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=frame_latent_dim,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.attn = nn.Linear(hidden * 2, 1)
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, scene_latent_dim)
        )
        self.scene_latent_dim = scene_latent_dim

    def forward(self, z_seq):
        # z_seq: (B, T, D_frame)
        h, _ = self.lstm(z_seq)               # (B, T, 2*hidden)
        attn_w = torch.softmax(self.attn(h), dim=1)  # (B, T, 1)
        pooled = (h * attn_w).sum(dim=1)      # (B, 2*hidden)
        z_scene = self.proj(pooled)           # (B, scene_latent_dim)
        return z_scene


class PositionProbe(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 1)

    def forward(self, z):
        return self.linear(z).squeeze(-1)


def weighted_reconstruction_loss(recon, target, fg_weight=50.0):
    with torch.no_grad():
        fg_mask = (target.max(dim=2, keepdim=True).values > 0.2).float()
        weights = 1.0 + (fg_weight - 1.0) * fg_mask
    sq_err = (recon - target) ** 2
    return (sq_err * weights).mean()


class WorldModel(nn.Module):
    """
    v6 — adds SCENE ENCODER for trajectory-level properties (k, b).

    Per-frame latent z_t encodes what's in this frame (position).
    Scene latent z_scene encodes what's true of the WHOLE video (k, b).
    """
    def __init__(self, latent_dim=16, scene_latent_dim=32,
                 beta=0.001, pos_aux_weight=20.0, fg_weight=50.0):
        super().__init__()
        self.encoder       = FrameEncoder(latent_dim)
        self.decoder       = FrameDecoder(latent_dim)
        self.dynamics      = LatentDynamics(latent_dim)
        self.scene_encoder = SceneEncoder(latent_dim, scene_latent_dim)
        self.pos_probe     = PositionProbe(latent_dim)

        self.latent_dim = latent_dim
        self.scene_latent_dim = scene_latent_dim
        self.beta = beta
        self.pos_aux_weight = pos_aux_weight
        self.fg_weight = fg_weight

    def encode_video(self, video):
        B, T, C, H, W = video.shape
        frames = rearrange(video, 'b t c h w -> (b t) c h w')
        mu, logvar = self.encoder(frames)
        z = self.encoder.sample(mu, logvar)
        mu     = rearrange(mu,     '(b t) d -> b t d', b=B)
        logvar = rearrange(logvar, '(b t) d -> b t d', b=B)
        z      = rearrange(z,      '(b t) d -> b t d', b=B)
        return mu, logvar, z

    def encode_scene(self, video):
        """Returns (z_per_frame_mean, z_scene)"""
        mu, _, _ = self.encode_video(video)
        z_scene = self.scene_encoder(mu)
        return mu, z_scene

    def forward(self, video, states=None):
        B, T, C, H, W = video.shape
        mu, logvar, z = self.encode_video(video)

        z_flat = rearrange(z, 'b t d -> (b t) d')
        recon  = self.decoder(z_flat)
        recon  = rearrange(recon, '(b t) c h w -> b t c h w', b=B)
        recon_loss = weighted_reconstruction_loss(recon, video, fg_weight=self.fg_weight)

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        mu_pred  = self.dynamics(mu[:, :-1, :])
        mu_target = mu[:, 1:, :].detach()
        dyn_loss = F.mse_loss(mu_pred, mu_target)

        if states is not None:
            x_gt = states[..., 0]
            mu_flat_for_probe = rearrange(mu, 'b t d -> (b t) d')
            x_pred = self.pos_probe(mu_flat_for_probe)
            x_pred = rearrange(x_pred, '(b t) -> b t', b=B)
            pos_loss = F.mse_loss(x_pred, x_gt) / 1.3
        else:
            pos_loss = torch.tensor(0.0, device=video.device)

        # Scene encoder runs but no aux loss in this version
        # (probe will be trained separately on top of scene latent)
        _ = self.scene_encoder(mu)

        total = (recon_loss
                 + self.beta * kl_loss
                 + dyn_loss
                 + self.pos_aux_weight * pos_loss)

        with torch.no_grad():
            mu_var_per_dim = mu.reshape(-1, self.latent_dim).var(dim=0)
            active_dims = (mu_var_per_dim > 0.01).sum().item()

        losses = {
            'total': total.item(),
            'recon': recon_loss.item(),
            'kl':    kl_loss.item(),
            'dyn':   dyn_loss.item(),
            'pos':   pos_loss.item(),
            'active_dims': active_dims
        }
        return total, losses

    @torch.no_grad()
    def reconstruct(self, video):
        self.eval()
        mu, _, _ = self.encode_video(video)
        B, T = video.shape[:2]
        mu_flat = rearrange(mu, 'b t d -> (b t) d')
        recon = self.decoder(mu_flat)
        return rearrange(recon, '(b t) c h w -> b t c h w', b=B)

    @torch.no_grad()
    def predict_position(self, video):
        self.eval()
        mu, _, _ = self.encode_video(video)
        B, T = mu.shape[:2]
        mu_flat = rearrange(mu, 'b t d -> (b t) d')
        x_pred = self.pos_probe(mu_flat)
        return rearrange(x_pred, '(b t) -> b t', b=B)

    @torch.no_grad()
    def get_scene_latent(self, video):
        """Returns scene-level latent (B, scene_latent_dim) for probing k, b."""
        self.eval()
        mu, z_scene = self.encode_scene(video)
        return z_scene

    @torch.no_grad()
    def rollout(self, z0, n_steps):
        self.eval()
        z_seq = [z0]
        for _ in range(n_steps - 1):
            seq = torch.stack(z_seq, dim=1)
            z_next = self.dynamics(seq)[0, -1:, :]
            z_seq.append(z_next)
        return torch.cat(z_seq, dim=0)