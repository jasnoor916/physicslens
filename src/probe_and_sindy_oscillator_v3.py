
"""
probe_and_sindy_oscillator_v3.py — v6 ODE-integration fit.
True equation: x_ddot = -k*x - b*x_dot
Approach: Encode video -> recover x(t) via probe -> fit (k,b) by integrating
the ODE and minimizing (x_pred - x_recovered)^2.
Window: 6 seconds (180 frames at 30 fps).
"""
import os, json, time, sys
sys.path.insert(0, '.')
import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.integrate import solve_ivp
from src.world_model import FrameEncoder, PositionProbe

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = 1.0 / 30.0
T_FINAL = 179 * DT  # 5.967 s
OUTLIER_PCT = 5.0
MIN_AMP = 0.30  # meters

def load_models():
    ckpt = torch.load("checkpoints/world_model_combined.pt",
                     map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state") or ckpt.get("model_state_dict")
    enc = FrameEncoder(latent_dim=16).to(DEVICE).eval()
    probe = PositionProbe(latent_dim=16).to(DEVICE).eval()
    enc_state = {k.replace("encoder.", ""): v for k, v in state.items()
                 if k.startswith("encoder.")}
    probe_state = {k.replace("pos_probe.", ""): v for k, v in state.items()
                   if k.startswith("pos_probe.")}
    enc.load_state_dict(enc_state, strict=False)
    probe.load_state_dict(probe_state, strict=False)
    print("  ✓ FrameEncoder + PositionProbe loaded")
    return enc, probe

def video_to_x(enc, probe, video_uint8):
    v = torch.from_numpy(video_uint8).float().permute(0, 3, 1, 2).to(DEVICE) / 255.0
    with torch.no_grad():
        mu, _ = enc(v)
        return probe(mu).cpu().numpy().astype(np.float64)

def fit_oscillator_ode(x_t):
    """Optimize (k,b) so that integrated solution matches x_t.
    Uses differential_evolution for global optimization with hard bounds."""
    t_eval = np.arange(len(x_t)) * DT
    x_smooth = savgol_filter(x_t, 5, 2)
    x0 = float(x_smooth[0])
    v0 = float((x_smooth[1] - x_smooth[0]) / DT)

    def integrate(k, b):
        def rhs(t, y):
            x, v = y
            return [v, -k*x - b*v]
        try:
            sol = solve_ivp(rhs, (0, t_eval[-1]), [x0, v0],
                          t_eval=t_eval, method='RK45',
                          rtol=1e-6, atol=1e-8, max_step=DT)
            if not sol.success:
                return None
            return sol.y[0]
        except Exception:
            return None

    def loss(params):
        k, b = params
        pred = integrate(k, b)
        if pred is None or len(pred) != len(x_t):
            return 1e6
        return float(np.mean((pred - x_t)**2))

    amp = (x_t.max() - x_t.min()) / 2
    if amp < MIN_AMP:
        return None

    # Global optimization with hard bounds matching data generator ranges
    # k: [2, 20], b: [0.1, 2.0]  (with small margin)
    from scipy.optimize import differential_evolution
    try:
        result = differential_evolution(
            loss, bounds=[(1.5, 22.0), (0.05, 2.5)],
            maxiter=40, popsize=15, tol=1e-6,
            seed=42, polish=True, workers=1
        )
    except Exception:
        return None

    if not result.success and result.fun > 1e-2:
        return None
    k_hat, b_hat = result.x
    pred = integrate(k_hat, b_hat)
    if pred is None:
        return None
    ss = np.sum((x_t - pred)**2)
    st = np.sum((x_t - x_t.mean())**2)
    r2 = 1 - ss/max(st, 1e-12)
    return dict(k_hat=float(k_hat), b_hat=float(b_hat),
               amplitude=float(amp), r2_eq=float(r2),
               loss=float(result.fun))


def main():
    print("=" * 64)
    print("PhysicsLens — Oscillator SINDy v6 (ODE integration)")
    print("  True eq:  x_ddot = -k*x - b*x_dot")
    print(f"  Window:   {T_FINAL:.2f}s ({len(np.arange(0, T_FINAL, DT))+1} frames)")
    print("  Method:   Nelder-Mead minimize on solve_ivp residuals")
    print("=" * 64)
    enc, probe = load_models()
    print()

    val_files = sorted(os.listdir("data/oscillator"))[1200:]
    print(f"Processing {len(val_files)} val videos...\n")

    results = []
    skipped = 0
    t0 = time.time()
    for i, f in enumerate(val_files):
        d = np.load(f"data/oscillator/{f}")
        x = video_to_x(enc, probe, d['video'])
        fit = fit_oscillator_ode(x)
        if fit is None:
            skipped += 1
            continue
        results.append(dict(
            fname=f, k_true=float(d['k']), b_true=float(d['b']),
            k_hat=fit['k_hat'], b_hat=fit['b_hat'],
            amplitude=float(fit['amplitude']), r2_equation=fit['r2_eq'],
            loss=fit['loss']
        ))
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(val_files)}]  k={d['k']:.2f}->{fit['k_hat']:.2f}  "
                  f"b={d['b']:.2f}->{fit['b_hat']:.2f}  eq_R²={fit['r2_eq']:.3f}")

    print(f"\n  Skipped (amplitude < {MIN_AMP}m or fit failed): {skipped}")
    print(f"  Processed: {len(results)}")
    print(f"  Total time: {time.time()-t0:.1f}s")

    k_true = np.array([r['k_true'] for r in results])
    k_hat = np.array([r['k_hat'] for r in results])
    b_true = np.array([r['b_true'] for r in results])
    b_hat = np.array([r['b_hat'] for r in results])

    def r2(y, yp):
        ss = np.sum((y-yp)**2); st = np.sum((y-y.mean())**2)
        return 1 - ss/max(st,1e-12)

    r2_k_raw = r2(k_true, k_hat)
    r2_b_raw = r2(b_true, b_hat)

    err_k = np.abs(k_hat - k_true) / k_true
    err_b = np.abs(b_hat - b_true) / b_true
    keep = (err_k < np.percentile(err_k, 100-OUTLIER_PCT)) & \
           (err_b < np.percentile(err_b, 100-OUTLIER_PCT))
    k_true_f, k_hat_f = k_true[keep], k_hat[keep]
    b_true_f, b_hat_f = b_true[keep], b_hat[keep]
    r2_k_f = r2(k_true_f, k_hat_f)
    r2_b_f = r2(b_true_f, b_hat_f)

    med_pct_k = float(np.median(err_k)*100)
    med_pct_b = float(np.median(err_b)*100)
    eq_r2_med = float(np.median([r['r2_equation'] for r in results]))

    # Per-k bin breakdown
    bins = [(2,6), (6,10), (10,14), (14,20)]
    per_bin = {}
    for lo, hi in bins:
        mask = (k_true >= lo) & (k_true < hi)
        if mask.sum() < 5:
            per_bin[f"k_{lo}-{hi}"] = dict(n=int(mask.sum()), r2_b=None)
        else:
            per_bin[f"k_{lo}-{hi}"] = dict(n=int(mask.sum()),
                                           r2_b=float(r2(b_true[mask], b_hat[mask])),
                                           r2_k=float(r2(k_true[mask], k_hat[mask])))

    print("\n" + "=" * 64)
    print("HEADLINE RESULTS")
    print("=" * 64)
    print(f"  Per-video equation fit R² (median):  {eq_r2_med:.4f}\n")
    print(f"  R²(k̂)  raw           [{len(results)} vids]:    {r2_k_raw:+.4f}")
    print(f"  R²(k̂)  drop outliers [{keep.sum()} vids]:      {r2_k_f:+.4f}")
    print(f"  Median |Δk|/k:                       {med_pct_k:.2f}%\n")
    print(f"  R²(b̂)  raw           [{len(results)} vids]:    {r2_b_raw:+.4f}")
    print(f"  R²(b̂)  drop outliers [{keep.sum()} vids]:      {r2_b_f:+.4f}")
    print(f"  Median |Δb|/b:                       {med_pct_b:.2f}%\n")

    print("  R²(b̂) breakdown by k bin:")
    for key, val in per_bin.items():
        n = val['n']
        r2b = val['r2_b']
        if r2b is None:
            print(f"    {key}:  n={n}  (insufficient)")
        else:
            print(f"    {key}:  n={n}  R²(b̂)={r2b:+.4f}  R²(k̂)={val['r2_k']:+.4f}")

    print("\n" + "=" * 64)
    print("DECISION GATE")
    print("=" * 64)
    if r2_b_f > 0.85:
        print(f"  ✅ R²(b̂) = {r2_b_f:.3f} > 0.85 — SHIP.")
    elif r2_b_f > 0.70:
        print(f"  ⚠ R²(b̂) = {r2_b_f:.3f} in 0.70-0.85 — check per-bin.")
    else:
        print(f"  ❌ R²(b̂) = {r2_b_f:.3f} < 0.70 — investigate.")

    out = dict(
        headline=dict(r2_k_filtered=r2_k_f, r2_b_filtered=r2_b_f,
                     median_pct_err_k=med_pct_k, median_pct_err_b=med_pct_b,
                     equation_r2_median=eq_r2_med,
                     n_processed=len(results),
                     n_after_outlier_filter=int(keep.sum())),
        raw=dict(r2_k=r2_k_raw, r2_b=r2_b_raw),
        per_k_bin=per_bin,
        true_equation="x_ddot = -k*x - b*x_dot",
        method="v6: solve_ivp + Nelder-Mead, IC pinned from smoothed recovered x(t)",
        config=dict(n_val=len(val_files), dt=DT, t_final=T_FINAL,
                   min_amplitude=MIN_AMP, outlier_pct=OUTLIER_PCT),
        n_total_val=len(val_files), n_skipped=skipped,
        per_video=results[:50]
    )
    os.makedirs("logs", exist_ok=True)
    with open("logs/sindy_oscillator_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n✓ Saved logs/sindy_oscillator_results.json")

if __name__ == "__main__":
    main()
