# 🔬 PhysicsLens

> An AI that watches physical systems, learns their dynamics, simulates counterfactuals, and rediscovers the equations of motion — all from raw video.

**Built for [Build with MeDo](https://medo.com) Hackathon** · 7 days · Team: [Jasnoor Kaur](https://github.com/jasnoor916) + Aniket

---

## ⚡ TL;DR

Show PhysicsLens a 30-second video of a damped spring oscillator. With **zero supervision on physical parameters**, it:

| Capability | Result |
|---|---|
| Recovers ball position from video | **R² = 0.9995** |
| Recovers spring constant `k` | **R² = 0.997** |
| Recovers damping coefficient `b` | **R² = 0.989** |
| Rediscovers equation `ẍ = -kx - bẋ` via SINDy | **R² = 0.999** for `k` |

The model never sees `k` or `b` during world-model training. Yet its learned representations encode them with near-perfect linear separability.

---

## 🧠 The Pipeline

```
Raw Video (90 frames, 64×64)
    ↓
[1] FrameEncoder (CNN → 16-dim latent z_t)
    ↓
[2] SceneEncoder (BiLSTM + attention → 32-dim z_scene)
    ↓
    ├── PositionProbe   (linear: z_t → x)         R²=0.9995
    ├── PhysicsProbes   (z_scene → k, b)          R²=0.997, 0.989
    ├── SINDy           (recovered x(t) → ODE)    R²=0.999
    ├── Counterfactual Engine (modify k, b → new video)
    └── LLM Explainer   (GPT-4o, grounded in R²)
```

---

## 📊 Results

### Day 1: Damped Harmonic Oscillator

Trained on 1,500 procedurally-generated videos of damped springs with `k ∈ [2, 20]`, `b ∈ [0.1, 2.0]`.

**Key finding:** The frame-level latent `z_t` perfectly encodes position (R²=0.9995) but cannot encode trajectory-level properties like `k` and `b`. A separate scene encoder operating on the full 90-frame latent sequence recovers them at R²=0.997 / 0.989.

### Why 90 frames matters

For weakly damped systems (`b ≈ 0.1`), damping is invisible in fewer than ~1.5 oscillation periods (~3 seconds at 30 fps). Using the full 90-frame sequence vs. a 20-frame snippet jumped R²(b) from **0.06 → 0.989**.

---

## 🗂 Repository Structure

```
physicslens/
├── src/
│   ├── world_model.py            # FrameEncoder, FrameDecoder, LatentDynamics, SceneEncoder
│   ├── dataset.py                # OscillatorDataset, get_dataloaders
│   ├── train.py                  # World model training (R²=0.9995)
│   ├── train_scene_encoder.py    # Scene encoder training (R²=0.997)
│   ├── data_gen_oscillator.py    # pymunk-based video generator
│   ├── data_gen_pendulum.py      # [Day 2] Pendulum dynamics
│   ├── probe_and_sindy.py        # v1 (ORANGE: R²=0.49)
│   └── probe_and_sindy_v2.py     # v2 (GREEN: R²=0.997)
├── notebooks/
│   └── PhysicsLens.ipynb         # Colab training notebook
├── configs/
│   ├── oscillator.yaml
│   └── pendulum.yaml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/jasnoor916/physicslens.git
cd physicslens
pip install -r requirements.txt
```

### 2. Generate data (offline, no GPU needed)

```bash
python src/data_gen_oscillator.py
```

### 3. Train world model (needs GPU — use Colab)

```bash
python src/train.py
```

### 4. Train scene encoder + run SINDy

```bash
python src/probe_and_sindy_v2.py
```

---

## 🛠 Architecture Decisions That Mattered

| Iteration | Problem | Fix |
|---|---|---|
| v1 (β=4.0) | Posterior collapse, KL=0 | Reduced β to 0.001 |
| v2 (free_bits) | Decoder output black frames | Removed free_bits, weighted recon loss |
| v3 (state aux) | Velocity unobservable per-frame | Switched to position-only probe |
| v4 (k, b probes) | R² ≈ 0 from per-frame latents | Added scene-level encoder |
| v5 (BiLSTM) | R²(k)=0.49, R²(b)=0.06 | Used all 90 frames, separate heads |
| **v6 (current)** | — | **R²(k)=0.997, R²(b)=0.989** |

---

## 📈 What's Next

- [x] Day 1: Damped oscillator pipeline (✅ GREEN)
- [ ] Day 2: Pendulum + counterfactual rollout engine
- [ ] Day 3: Multi-scenario world model
- [ ] Day 4: LLM explainer agent (GPT-4o grounded in R² scores)
- [ ] Day 5: MeDo UI integration
- [ ] Day 6: 3-min demo video
- [ ] Day 7: Submit + social blitz

---

## 🧬 Tech Stack

PyTorch · pymunk · pygame · PySINDy · einops · scikit-learn · scipy · GPT-4o

---

## 📜 License

MIT