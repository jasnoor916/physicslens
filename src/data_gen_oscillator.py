import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import pymunk
import pygame
import numpy as np
import cv2
from pathlib import Path

pygame.init()

def generate_oscillator_video(k, b, m, x0, v0,
                               duration=3.0, fps=30,
                               save_path=None,
                               img_size=(64, 64)):
    W_render, H_render = 320, 240
    center_x, center_y = W_render // 2, H_render // 2
    scale = 50

    space = pymunk.Space()
    space.gravity = (0, 0)

    body = pymunk.Body(m, pymunk.moment_for_circle(m, 0, 10))
    body.position = (center_x + x0 * scale, center_y)
    body.velocity = (v0 * scale, 0)

    shape = pymunk.Circle(body, 10)
    shape.elasticity = 0.0
    shape.friction = 0.0
    space.add(body, shape)

    frames = []
    states = []
    dt = 1.0 / fps
    n_frames = int(duration * fps)

    for i in range(n_frames):
        x_m = (body.position.x - center_x) / scale
        v_ms = body.velocity.x / scale

        F_m = -k * x_m - b * v_ms
        F_px = F_m * scale

        body.apply_force_at_local_point((F_px, 0), (0, 0))
        space.step(dt)

        x_current = (body.position.x - center_x) / scale
        v_current = body.velocity.x / scale
        states.append([x_current, v_current])

        surface = pygame.Surface((W_render, H_render))
        surface.fill((20, 20, 30))

        pygame.draw.line(surface, (80, 80, 100),
                        (center_x, center_y),
                        (int(body.position.x), center_y), 2)

        pygame.draw.rect(surface, (120, 120, 140),
                        (center_x - 5, center_y - 20, 5, 40))

        bx = int(body.position.x)
        by = int(body.position.y)
        bx = max(15, min(W_render - 15, bx))
        pygame.draw.circle(surface, (220, 80, 80), (bx, by), 12)
        pygame.draw.circle(surface, (255, 120, 120), (bx - 3, by - 3), 4)

        frame = pygame.surfarray.array3d(surface)
        frame = frame.swapaxes(0, 1)
        frame_resized = cv2.resize(frame, (img_size[1], img_size[0]))
        frames.append(frame_resized)

    video = np.stack(frames, axis=0).astype(np.uint8)
    states = np.array(states, dtype=np.float32)
    params = {
        'k': np.float32(k), 'b': np.float32(b), 'm': np.float32(m),
        'x0': np.float32(x0), 'v0': np.float32(v0)
    }

    if save_path is not None:
        np.savez(str(save_path), video=video, states=states, **params)

    return video, states, params


def generate_dataset(n_samples=500, out_dir="data/oscillator", seed=42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    print(f"Generating {n_samples} oscillator videos...")
    for i in range(n_samples):
        k  = np.random.uniform(2.0, 20.0)
        b  = np.random.uniform(0.1, 2.0)
        m  = 1.0
        x0 = np.random.uniform(-2.0, 2.0)
        v0 = np.random.uniform(-2.0, 2.0)

        save_path = out_dir / f"sample_{i:04d}.npz"
        generate_oscillator_video(k, b, m, x0, v0, save_path=save_path)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n_samples}]")

    print(f"Done.")

if __name__ == "__main__":
    generate_dataset(n_samples=500)

    d = np.load("data/oscillator/sample_0000.npz")
    print(f"\nSanity check:")
    print(f"  video shape:  {d['video'].shape}")
    print(f"  states shape: {d['states'].shape}")
    print(f"  k={float(d['k']):.3f}, b={float(d['b']):.3f}")
    print(f"  x range: [{d['states'][:,0].min():.3f}, {d['states'][:,0].max():.3f}]")