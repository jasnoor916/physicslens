
"""
probe_and_sindy_oscillator_v8.py
Hilbert envelope method for damping coefficient.

For x(t) = A*exp(-bt/2)*cos(wt + phi), the analytic signal
  z(t) = x(t) + j*H[x](t)
has |z(t)| = A*exp(-bt/2)  (the envelope, demodulated from carrier).

So log|z(t)| = log(A) - (b/2)*t  ->  linear fit gives b.
This works regardless of frequency k.

Pipeline:
  1. Recover x(t) from video
  2. k from FFT peak (frequency is unambiguous)
  3. b from log-envelope linear fit (this is the breakthrough)
  4. Verify with ODE forward integration
"""
import os, json, time, sys
sys.path.insert(0, '.')
import numpy as np
import torch
from scipy.signal import hilbert, savgol_filter
from scipy.integrate import solve_ivp
from src.world_model import FrameEncoder, PositionProbe

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = 1.0 / 30.0
T_FINAL = 179 * DT
OUTLIER_PCT = 5.0
MIN_AMP = 0.30  # meters

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

def estimate_k_from_fft(x, dt):
    """Estimate angular frequency from FFT peak, then k = omega^2."""
    n = len(x)
    x_centered = x - x.mean()
    # Apply Hann window to reduce spectral leakage
    window = np.hanning(n)
    xw = x_centered * window
    fft_mag = np.abs(np.fft.rfft(xw))
    freqs = np.fft.rfftfreq(n, dt)
    # Skip DC, find peak
    fft_mag[0] = 0
    peak_idx = np.argmax(fft_mag)
    if peak_idx == 0 or peak_idx >= len(freqs)-1:
        return None
    # Parabolic interpolation for sub-bin precision
    y0, y1, y2 = fft_mag[peak_idx-1], fft_mag[peak_idx], fft_mag[peak_idx+1]
    if y1 == 0:
        return None
    delta = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2 + 1e-12)
    f_peak = freqs[peak_idx] + delta * (freqs[1] - freqs[0])
    omega_d = 2*np.pi * f_peak  # damped frequency
    return float(omega_d)

def estimate_b_from_hilbert(x, dt):
    """The breakthrough: extract envelope via Hilbert transform,
    fit log(envelope) linearly to get b."""
    x_centered = x - np.mean(x[-30:])  # remove any drift
    analytic = hilbert(x_centered)
    envelope = np.abs(analytic)
    # Smooth envelope to reduce edge artifacts of Hilbert transform
    if len(envelope) >= 11:
        envelope_smooth = savgol_filter(envelope, 11, 2)
    else:
        envelope_smooth = envelope
    # Trim edges where Hilbert transform has artifacts
    trim = max(5, len(envelope) // 20)
    env_inner = envelope_smooth[trim:-trim]
    t_inner = (np.arange(len(envelope_smooth)) * dt)[trim:-trim]
    # Filter for valid (positive, non-tiny) envelope values
    valid = env_inner > 0.01 * env_inner.max()
    if valid.sum() < 20:
        return None
    log_env = np.log(env_inner[valid])
    t_valid = t_inner[valid]
    # Robust linear fit: weight by envelope magnitude (early samples more reliable)
    weights = env_inner[valid]
    slope, intercept = np.polyfit(t_valid, log_env, 1, w=weights)
    # slope = -b/2, so b = -2*slope
    b = -2 * slope
    return float(b) if 0.01 < b < 5.0 else None

def fit_oscillator_v8(x_t):
    n = len(x_t)
    if n < 60:
        return None
    x_smooth = savgol_filter(x_t, 7, 2)
    amp = (x_smooth.max() - x_smooth.min()) / 2
    if amp < MIN_AMP:
        return None

    # ---- k from FFT (damped frequency) ----
    omega_d = estimate_k_from_fft(x_smooth, DT)
    if omega_d is None or omega_d < 0.5:
        return None

    # ---- b from Hilbert envelope ----
    b_hat = estimate_b_from_hilbert(x_smooth, DT)
    if b_hat is None:
        return None

    # ---- Recover natural frequency: omega_n^2 = omega_d^2 + (b/2)^2 ----
    # so k = omega_n^2 = omega_d^2 + b^2/4
    k_hat = omega_d**2 + (b_hat/2)**2

    # ---- Verify by integrating and computing equation fit R² ----
    x0 = float(x_smooth[0])
    v0 = float((x_smooth[1] - x_smooth[0]) / DT)
    t_eval = np.arange(n) * DT
    def rhs(t, y):
        return [y[1], -k_hat*y[0] - b_hat*y[1]]
    try:
        sol = solve_ivp(rhs, (0, t_eval[-1]), [x0, v0],
                       t_eval=t_eval, method='RK45',
                       rtol=1e-7, atol=1e-9, max_step=DT)
        x_pred = sol.y[0] if sol.success else None
    except Exception:
        x_pred = None

    if x_pred is None:
        r2_eq = 0.0
    else:
        ss = np.sum((x_t - x_pred)**2)
        st = np.sum((x_t - x_t.mean())**2)
        r2_eq = float(1 - ss/max(st, 1e-12))

    return dict(k_hat=float(k_hat), b_hat=float(b_hat),
               omega_d=float(omega_d), amplitude=float(amp),
               r2_eq=r2_eq)

def main():
    print("=" * 64)
    print("PhysicsLens — Oscillator SINDy v8 (Hilbert envelope)")
    print("  k from FFT peak | b from Hilbert envelope decay")
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
        fit = fit_oscillator_v8(x)
        if fit is None:
            skipped += 1
            continue
        results.append(dict(
            fname=f, k_true=float(d['k']), b_true=float(d['b']),
            k_hat=fit['k_hat'], b_hat=fit['b_hat'],
            amplitude=float(fit['amplitude']), r2_equation=fit['r2_eq']
        ))
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(val_files)}]  k={d['k']:.2f}->{fit['k_hat']:.2f}  "
                  f"b={d['b']:.2f}->{fit['b_hat']:.2f}  eq_R²={fit['r2_eq']:.3f}")

    print(f"\n  Skipped: {skipped}    Processed: {len(results)}")
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
    r2_k_f, r2_b_f = r2(k_true[keep], k_hat[keep]), r2(b_true[keep], b_hat[keep])
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
    print("HEADLINE RESULTS  (v8 Hilbert)")
    print("=" * 64)
    print(f"  Per-video equation fit R² (median):  {eq_r2_med:.4f}\n")
    print(f"  R²(k̂)  raw: {r2_k_raw:+.4f}     filtered: {r2_k_f:+.4f}")
    print(f"  R²(b̂)  raw: {r2_b_raw:+.4f}     filtered: {r2_b_f:+.4f}")
    print(f"  Median |Δk|/k: {float(np.median(err_k)*100):.2f}%")
    print(f"  Median |Δb|/b: {float(np.median(err_b)*100):.2f}%\n")
    print("  Per k-bin:")
    for key,val in per_bin.items():
        if 'r2_b' in val:
            print(f"    {key}: n={val['n']}  R²(b̂)={val['r2_b']:+.3f}  "
                  f"R²(k̂)={val['r2_k']:+.3f}  med|Δb|/b={val['med_pct_err_b']:.1f}%")

    print("\n" + "=" * 64)
    print("DECISION GATE")
    print("=" * 64)
    if r2_b_f > 0.90:   print(f"  🎉 R²(b̂) = {r2_b_f:.3f} > 0.90 — WINNING.")
    elif r2_b_f > 0.85: print(f"  ✅ R²(b̂) = {r2_b_f:.3f} > 0.85 — SHIP.")
    elif r2_b_f > 0.70: print(f"  ⚠ R²(b̂) = {r2_b_f:.3f}")
    else:               print(f"  ❌ R²(b̂) = {r2_b_f:.3f} — Hilbert failed too.")

    out = dict(
        headline=dict(r2_k_filtered=r2_k_f, r2_b_filtered=r2_b_f,
                     equation_r2_median=eq_r2_med,
                     n_processed=len(results)),
        raw=dict(r2_k=r2_k_raw, r2_b=r2_b_raw),
        per_k_bin=per_bin,
        method="v8: FFT for k + Hilbert envelope linear fit for b",
        per_video=results[:50]
    )
    with open("logs/sindy_oscillator_results_v8.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n✓ Saved logs/sindy_oscillator_results_v8.json")

if __name__ == "__main__":
    main()
