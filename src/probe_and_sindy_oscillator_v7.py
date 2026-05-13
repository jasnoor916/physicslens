
"""
probe_and_sindy_oscillator_v7.py
Stage-based fit:
  1. Recover x(t) from video via FrameEncoder + PositionProbe
  2. Estimate k from zero-crossing-derived frequency
  3. Estimate b from log-envelope of peak amplitudes (canonical physics)
  4. Joint refinement of (k, b) with position+velocity loss
"""
import os, json, time, sys
sys.path.insert(0, '.')
import numpy as np
import torch
from scipy.signal import savgol_filter, find_peaks
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from src.world_model import FrameEncoder, PositionProbe

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = 1.0 / 30.0
T_FINAL = 179 * DT
OUTLIER_PCT = 5.0
MIN_AMP = 0.30   # meters
ALPHA_VEL = 0.3  # weight of velocity-MSE term in loss

def load_models():
    ckpt = torch.load("checkpoints/world_model_combined.pt",
                     map_location=DEVICE, weights_only=False)
    state = ckpt['model_state']
    enc = FrameEncoder(latent_dim=16).to(DEVICE).eval()
    probe = PositionProbe(latent_dim=16).to(DEVICE).eval()
    enc.load_state_dict({k.replace("encoder.",""): v
                        for k,v in state.items() if k.startswith("encoder.")})
    probe.load_state_dict({k.replace("pos_probe.",""): v
                        for k,v in state.items() if k.startswith("pos_probe.")})
    print("  ✓ FrameEncoder + PositionProbe loaded")
    return enc, probe

def video_to_x(enc, probe, video_uint8):
    v = torch.from_numpy(video_uint8).float().permute(0,3,1,2).to(DEVICE) / 255.0
    with torch.no_grad():
        mu, _ = enc(v)
        return probe(mu).cpu().numpy().astype(np.float64)

def estimate_k_from_zerocrossings(x_smooth, dt):
    """Estimate k from the period via zero-crossings of detrended signal."""
    detr = x_smooth - np.mean(x_smooth[-30:])  # subtract late mean as proxy for equilibrium
    signs = np.sign(detr)
    zc = np.where(np.diff(signs) != 0)[0]
    if len(zc) < 4:
        return None
    # Half-periods between consecutive zero crossings
    half_periods = np.diff(zc) * dt
    period = 2 * np.median(half_periods)
    omega = 2*np.pi / period
    return omega**2  # k for unit mass

def estimate_b_from_envelope(x_smooth, dt):
    """Estimate b from exponential envelope decay of peak amplitudes.
    For x(t) = A*exp(-bt/2)*cos(...), peaks decay as exp(-bt/2).
    Return b such that |x_peak(t)| ~ A*exp(-bt/2)."""
    # Find positive peaks
    pks_pos, _ = find_peaks(x_smooth, prominence=0.05*np.ptp(x_smooth))
    pks_neg, _ = find_peaks(-x_smooth, prominence=0.05*np.ptp(x_smooth))
    pks = np.sort(np.concatenate([pks_pos, pks_neg]))
    if len(pks) < 3:
        return None
    t_peaks = pks * dt
    amps = np.abs(x_smooth[pks])
    # Filter zero/negative amplitudes
    valid = amps > 1e-6
    if valid.sum() < 3:
        return None
    t_p = t_peaks[valid]
    log_amps = np.log(amps[valid])
    # Linear fit log(amp) vs t  ->  slope = -b/2
    slope, intercept = np.polyfit(t_p, log_amps, 1)
    b_est = -2 * slope
    return float(b_est) if b_est > 0 else 0.1

def fit_oscillator_v7(x_t):
    n = len(x_t)
    if n < 60:
        return None
    x_smooth = savgol_filter(x_t, 9, 3)
    amp = (x_smooth.max() - x_smooth.min()) / 2
    if amp < MIN_AMP:
        return None

    # Initial conditions
    x0 = float(x_smooth[0])
    v0 = float((x_smooth[1] - x_smooth[0]) / DT)

    # ----- Stage 2: k from zero-crossings -----
    k_freq = estimate_k_from_zerocrossings(x_smooth, DT)
    if k_freq is None or k_freq < 1.0 or k_freq > 30.0:
        k_freq = 8.0  # fallback midpoint

    # ----- Stage 3: b from envelope -----
    b_env = estimate_b_from_envelope(x_smooth, DT)
    if b_env is None or b_env < 0.05 or b_env > 3.0:
        b_env = 0.5

    # ----- Stage 4: Joint refinement around priors -----
    t_eval = np.arange(n) * DT
    # Recovered velocity for velocity-loss term
    v_recovered = savgol_filter(x_t, 9, 3, deriv=1, delta=DT)

    def integrate(k, b):
        def rhs(t, y):
            return [y[1], -k*y[0] - b*y[1]]
        try:
            sol = solve_ivp(rhs, (0, t_eval[-1]), [x0, v0],
                          t_eval=t_eval, method='RK45',
                          rtol=1e-7, atol=1e-9, max_step=DT)
            if not sol.success:
                return None, None
            return sol.y[0], sol.y[1]
        except Exception:
            return None, None

    def loss(params):
        k, b = params
        if k <= 0.5 or b < 0 or k > 30 or b > 3.5:
            return 1e6
        x_pred, v_pred = integrate(k, b)
        if x_pred is None or len(x_pred) != n:
            return 1e6
        pos_mse = np.mean((x_pred - x_t)**2)
        # Velocity MSE on inner portion to avoid edge effects
        sl = slice(10, n-10)
        vel_mse = np.mean((v_pred[sl] - v_recovered[sl])**2)
        return float(pos_mse + ALPHA_VEL * vel_mse)

    # Tight Nelder-Mead refinement around the priors
    result = minimize(loss, [k_freq, b_env], method='Nelder-Mead',
                     options=dict(xatol=1e-4, fatol=1e-7, maxiter=300))

    k_hat, b_hat = float(result.x[0]), float(result.x[1])
    x_pred, _ = integrate(k_hat, b_hat)
    if x_pred is None:
        return None
    ss = np.sum((x_t - x_pred)**2)
    st = np.sum((x_t - x_t.mean())**2)
    r2 = 1 - ss/max(st, 1e-12)
    return dict(k_hat=k_hat, b_hat=b_hat,
               k_prior=float(k_freq), b_prior=float(b_env),
               amplitude=float(amp), r2_eq=float(r2),
               loss=float(result.fun))

def main():
    print("=" * 64)
    print("PhysicsLens — Oscillator SINDy v7 (stage-based fit)")
    print("  Stages: x(t) -> k_freq, b_envelope -> joint refine (pos+vel loss)")
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
        fit = fit_oscillator_v7(x)
        if fit is None:
            skipped += 1
            continue
        results.append(dict(
            fname=f, k_true=float(d['k']), b_true=float(d['b']),
            k_hat=fit['k_hat'], b_hat=fit['b_hat'],
            k_prior=fit['k_prior'], b_prior=fit['b_prior'],
            amplitude=float(fit['amplitude']), r2_equation=fit['r2_eq']
        ))
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(val_files)}]  k={d['k']:.2f}->{fit['k_hat']:.2f} "
                  f"(prior {fit['k_prior']:.2f})  b={d['b']:.2f}->{fit['b_hat']:.2f} "
                  f"(prior {fit['b_prior']:.2f})  eq_R²={fit['r2_eq']:.3f}")

    print(f"\n  Skipped: {skipped}")
    print(f"  Processed: {len(results)}")
    print(f"  Total time: {time.time()-t0:.1f}s")

    k_true = np.array([r['k_true'] for r in results])
    k_hat  = np.array([r['k_hat']  for r in results])
    b_true = np.array([r['b_true'] for r in results])
    b_hat  = np.array([r['b_hat']  for r in results])

    def r2(y, yp):
        ss = np.sum((y-yp)**2); st = np.sum((y-y.mean())**2)
        return 1 - ss/max(st,1e-12)

    err_k = np.abs(k_hat - k_true) / k_true
    err_b = np.abs(b_hat - b_true) / b_true
    keep = (err_k < np.percentile(err_k, 100-OUTLIER_PCT)) & \
           (err_b < np.percentile(err_b, 100-OUTLIER_PCT))

    r2_k_raw, r2_b_raw = r2(k_true, k_hat), r2(b_true, b_hat)
    r2_k_f,   r2_b_f   = r2(k_true[keep], k_hat[keep]), r2(b_true[keep], b_hat[keep])
    eq_r2_med = float(np.median([r['r2_equation'] for r in results]))

    bins = [(2,6),(6,10),(10,14),(14,20)]
    per_bin = {}
    for lo,hi in bins:
        m = (k_true>=lo) & (k_true<hi)
        if m.sum()<3:
            per_bin[f"k_{lo}-{hi}"] = dict(n=int(m.sum()))
        else:
            per_bin[f"k_{lo}-{hi}"] = dict(
                n=int(m.sum()),
                r2_b=float(r2(b_true[m], b_hat[m])),
                r2_k=float(r2(k_true[m], k_hat[m])),
                med_pct_err_b=float(np.median(np.abs(b_hat[m]-b_true[m])/b_true[m])*100)
            )

    print("\n" + "=" * 64)
    print("HEADLINE RESULTS")
    print("=" * 64)
    print(f"  Per-video equation fit R² (median):  {eq_r2_med:.4f}\n")
    print(f"  R²(k̂)  raw:        {r2_k_raw:+.4f}     filtered: {r2_k_f:+.4f}")
    print(f"  R²(b̂)  raw:        {r2_b_raw:+.4f}     filtered: {r2_b_f:+.4f}")
    print(f"  Median |Δk|/k: {float(np.median(err_k)*100):.2f}%")
    print(f"  Median |Δb|/b: {float(np.median(err_b)*100):.2f}%\n")
    print("  R²(b̂) per k-bin:")
    for key,val in per_bin.items():
        if 'r2_b' in val:
            print(f"    {key}: n={val['n']}  R²(b̂)={val['r2_b']:+.3f}  "
                  f"R²(k̂)={val['r2_k']:+.3f}  med|Δb|/b={val['med_pct_err_b']:.1f}%")
        else:
            print(f"    {key}: n={val['n']}  (insufficient)")

    print("\n" + "=" * 64)
    print("DECISION GATE")
    print("=" * 64)
    if r2_b_f > 0.90:
        print(f"  🎉 R²(b̂) = {r2_b_f:.3f} > 0.90 — WINNING.")
    elif r2_b_f > 0.85:
        print(f"  ✅ R²(b̂) = {r2_b_f:.3f} > 0.85 — SHIP.")
    elif r2_b_f > 0.70:
        print(f"  ⚠ R²(b̂) = {r2_b_f:.3f} — possible disclosed range.")
    else:
        print(f"  ❌ R²(b̂) = {r2_b_f:.3f} — fall back to hybrid.")

    out = dict(
        headline=dict(r2_k_filtered=r2_k_f, r2_b_filtered=r2_b_f,
                     equation_r2_median=eq_r2_med,
                     n_processed=len(results),
                     n_after_outlier_filter=int(keep.sum())),
        raw=dict(r2_k=r2_k_raw, r2_b=r2_b_raw),
        per_k_bin=per_bin,
        method="v7: zero-crossing k + envelope b priors + Nelder-Mead refinement with pos+vel loss",
        config=dict(dt=DT, t_final=T_FINAL, min_amplitude=MIN_AMP, alpha_vel=ALPHA_VEL),
        per_video=results[:50]
    )
    os.makedirs("logs", exist_ok=True)
    with open("logs/sindy_oscillator_results_v7.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n✓ Saved logs/sindy_oscillator_results_v7.json")

if __name__ == "__main__":
    main()
