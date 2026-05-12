"""
data_gen_pendulum.py — Damped pendulum video dataset

True equation: θ̈ = -(g/L)·sin(θ) - b·θ̇
Small-angle limit: θ̈ ≈ -(g/L)·θ - b·θ̇

Parameters:
    L  ∈ [0.3, 2.0]   pendulum length (m)
    b  ∈ [0.05, 0.8]  damping coefficient
    θ0 ∈ [-π/3, π/3]  initial angle (rad)
    g  = 9.81 (fixed)
"""

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import numpy as np
import pygame
import cv2
from pathlib import Path

pygame.init()

G        = 9.81
IMG_SIZE = 64
FPS      = 30
N_FRAMES = 90
PX_PER_M = 25.0


def simulate_pendulum_rk4(L, b, theta0, omega0=0.0,
                          n_frames=N_FRAMES, fps=FPS):
    """RK4 integration of damped pendulum dynamics."""
    dt = 1.0 / fps
    theta, omega = theta0, omega0
    thetas, omegas = [theta], [omega]

    def deriv(th, om):
        return om, -(G / L) * np.sin(th) - b * om

    for _ in range(n_frames - 1):
        k1_th, k1_om = deriv(theta, omega)
        k2_th, k2_om = deriv(theta + 0.5*dt*k1_th, omega + 0.5*dt*k1_om)
        k3_th, k3_om = deriv(theta + 0.5*dt*k2_th, omega + 0.5*dt*k2_om)
        k4_th, k4_om = deriv(theta + dt*k3_th,     omega + dt*k3_om)

        theta += (dt / 6) * (k1_th + 2*k2_th + 2*k3_th + k4_th)
        omega += (dt / 6) * (k1_om + 2*k2_om + 2*k3_om + k4_om)

        thetas.append(theta)
        omegas.append(omega)

    return np.array(thetas, dtype=np.float32), np.array(omegas, dtype=np.float32)


def render_pendulum(theta_traj, L, img_size=IMG_SIZE):
    """Render pendulum frames with pygame."""
    W_render, H_render = 320, 240
    surface = pygame.Surface((W_render, H_render))

    pivot_x = W_render // 2
    pivot_y = H_render // 4
    rod_px  = int(L * PX_PER_M * 2)
    bob_r   = max(8, int(W_render * 0.04))

    frames = []
    for theta in theta_traj:
        surface.fill((20, 20, 30))

        bob_x = pivot_x + int(rod_px * np.sin(theta))
        bob_y = pivot_y + int(rod_px * np.cos(theta))

        # Rod
        pygame.draw.line(surface, (180, 180, 200),
                         (pivot_x, pivot_y), (bob_x, bob_y), 3)
        # Pivot
        pygame.draw.circle(surface, (120, 120, 140),
                           (pivot_x, pivot_y), 5)
        # Bob (red)
        pygame.draw.circle(surface, (220, 80, 80),
                           (bob_x, bob_y), bob_r)
        pygame.draw.circle(surface, (255, 130, 130),
                           (bob_x - 3, bob_y - 3), bob_r // 4)

        frame = pygame.surfarray.array3d(surface).swapaxes(0, 1)
        frame_resized = cv2.resize(frame, (img_size, img_size))
        frames.append(frame_resized)

    return np.stack(frames, axis=0).astype(np.uint8)


def generate_pendulum_video(L, b, theta0, omega0=0.0, save_path=None):
    theta_traj, omega_traj = simulate_pendulum_rk4(L, b, theta0, omega0)
    video = render_pendulum(theta_traj, L)
    states = np.column_stack([theta_traj, omega_traj])

    params = {
        'L':      np.float32(L),
        'b':      np.float32(b),
        'g':      np.float32(G),
        'theta0': np.float32(theta0),
        'omega0': np.float32(omega0),
    }

    if save_path is not None:
        np.savez(str(save_path), video=video, states=states, **params)

    return video, states, params


def generate_dataset(n_samples=1500, out_dir="data/pendulum", seed=42):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print(f"Generating {n_samples} pendulum videos → {out}")
    print(f"  L ∈ [0.3, 2.0]  b ∈ [0.05, 0.8]  θ0 ∈ [-π/3, π/3]")

    for i in range(n_samples):
        L      = rng.uniform(0.3, 2.0)
        b      = rng.uniform(0.05, 0.8)
        theta0 = rng.uniform(-np.pi / 3, np.pi / 3)

        save_path = out / f"sample_{i:04d}.npz"
        generate_pendulum_video(L, b, theta0, save_path=save_path)

        if (i + 1) % 100 == 0:
            T_period = 2 * np.pi * np.sqrt(L / G)
            print(f"  [{i+1:4d}/{n_samples}] L={L:.2f}m b={b:.3f} "
                  f"θ0={np.degrees(theta0):+.0f}° T={T_period:.2f}s")

    print(f"\nDone. {n_samples} samples saved.")


if __name__ == "__main__":
    generate_dataset(n_samples=1500)

    # Sanity check
    d = np.load("data/pendulum/sample_0000.npz")
    print(f"\nSanity check:")
    print(f"  video shape:  {d['video'].shape}")
    print(f"  states shape: {d['states'].shape}")
    print(f"  L={float(d['L']):.3f}  b={float(d['b']):.3f}")
    print(f"  θ range: [{d['states'][:,0].min():.3f}, {d['states'][:,0].max():.3f}] rad")