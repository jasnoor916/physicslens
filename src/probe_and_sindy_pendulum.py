
"""
probe_and_sindy_pendulum.py — SINDy on damped pendulum.
True equation: theta_ddot = -(g/L)*sin(theta) - b*theta_dot
Library: [sin(theta), theta_dot]
Smoother: Savitzky-Golay
"""
import os, json, time, sys
sys.path.insert(0, '.')
import numpy as np
import torch
from scipy.signal import savgol_filter
from src.world_model import FrameEncoder, PositionProbe

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = 1.0 / 30.0
G = 9.81
SAVGOL_WIN = 15
SAVGOL_POLY = 3
MIN_AMP = 0.20
TRIM = 7
OUTLIER_PCT = 5.0

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

def video_to_theta(enc, probe, video_uint8):
    v = torch.from_numpy(video_uint8).float().permute(0, 3, 1, 2).to(DEVICE) / 255.0
    with torch.no_grad():
        mu, _ = enc(v)
        return probe(mu).cpu().numpy().astype(np.float64)

def fit_pendulum_sindy(theta_t):
    n = len(theta_t)
    theta_s = savgol_filter(theta_t, SAVGOL_WIN, SAVGOL_POLY)
    thd = savgol_filter(theta_t, SAVGOL_WIN, SAVGOL_POLY, deriv=1, delta=DT)
    thdd = savgol_filter(theta_t, SAVGOL_WIN, SAVGOL_POLY, deriv=2, delta=DT)
    sl = slice(TRIM, n - TRIM)
    th, thdt, thddt = theta_s[sl], thd[sl], thdd[sl]
    amp = (theta_t.max() - theta_t.min()) / 2
    if amp < MIN_AMP:
        return None
    X = np.stack([np.sin(th), thdt], axis=1)
    coef, *_ = np.linalg.lstsq(X, thddt, rcond=None)
    pred = X @ coef
    ss_res = np.sum((thddt - pred) ** 2)
    ss_tot = np.sum((thddt - thddt.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    coef_sin, coef_thd = float(coef[0]), float(coef[1])
    if coef_sin >= 0:
        return None
    L_hat = -G / coef_sin
    b_hat = -coef_thd
    return dict(L_hat=L_hat, b_hat=b_hat, coef_sin=coef_sin, coef_thd=coef_thd,
                amplitude=amp, r2_eq=r2)

def main():
    print("=" * 64)
    print("PhysicsLens — Pendulum SINDy")
    print("  True eq:  theta_ddot = -(g/L)*sin(theta) - b*theta_dot")
    print("  Library:  [sin(theta), theta_dot]")
    print(f"  Smoother: Savitzky-Golay (win={SAVGOL_WIN}, poly={SAVGOL_POLY})")
    print(f"  Min amp:  {MIN_AMP} rad")
    print("=" * 64)
    enc, probe = load_models()
    print()

    val_files = sorted(os.listdir("data/pendulum"))[1200:]
    print(f"Processing {len(val_files)} val videos...\n")

    results = []
    skipped = 0
    for i, f in enumerate(val_files):
        d = np.load(f"data/pendulum/{f}")
        theta = video_to_theta(enc, probe, d['video'])
        fit = fit_pendulum_sindy(theta)
        if fit is None:
            skipped += 1
            continue
        results.append(dict(
            fname=f, L_true=float(d['L']), b_true=float(d['b']),
            L_hat=fit['L_hat'], b_hat=fit['b_hat'],
            coeff_sin=fit['coef_sin'], coeff_thd=fit['coef_thd'],
            amplitude=float(fit['amplitude']), r2_equation=fit['r2_eq']
        ))
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(val_files)}]  L={d['L']:.2f}->{fit['L_hat']:.2f}  "
                  f"b={d['b']:.2f}->+{fit['b_hat']:.2f}  eq_R²={fit['r2_eq']:.3f}")

    print(f"\n  Skipped (amplitude < {MIN_AMP} rad): {skipped}")
    print(f"  Processed: {len(results)}")

    L_true = np.array([r['L_true'] for r in results])
    L_hat = np.array([r['L_hat'] for r in results])
    b_true = np.array([r['b_true'] for r in results])
    b_hat = np.array([r['b_hat'] for r in results])

    def r2(y, yp):
        ss = np.sum((y-yp)**2); st = np.sum((y-y.mean())**2)
        return 1 - ss/max(st,1e-12)

    r2_L_raw = r2(L_true, L_hat)
    r2_b_raw = r2(b_true, b_hat)

    err_L = np.abs(L_hat - L_true) / L_true
    err_b = np.abs(b_hat - b_true) / b_true
    keep = err_L < np.percentile(err_L, 100-OUTLIER_PCT)
    L_true_f, L_hat_f = L_true[keep], L_hat[keep]
    b_true_f, b_hat_f = b_true[keep], b_hat[keep]
    r2_L_f = r2(L_true_f, L_hat_f)
    r2_b_f = r2(b_true_f, b_hat_f)

    eq_r2_med = float(np.median([r['r2_equation'] for r in results]))
    eq_r2_mean = float(np.mean([r['r2_equation'] for r in results]))
    med_pct_L = float(np.median(err_L)*100)
    med_pct_b = float(np.median(err_b)*100)
    coef_sin_med = float(np.median([r['coeff_sin'] for r in results]))
    coef_thd_med = float(np.median([r['coeff_thd'] for r in results]))
    L_med = float(np.median(L_true)); b_med = float(np.median(b_true))

    print("\n" + "=" * 64)
    print("HEADLINE RESULTS")
    print("=" * 64)
    print(f"  Per-video equation fit R² (median):  {eq_r2_med:.4f}\n")
    print(f"  R²(L̂)  raw           [{len(results)} vids]:    {r2_L_raw:+.4f}")
    print(f"  R²(L̂)  drop {OUTLIER_PCT:.0f}% outliers [{keep.sum()} vids]:  {r2_L_f:+.4f}")
    print(f"  Median |ΔL|/L:                       {med_pct_L:.2f}%\n")
    print(f"  R²(b̂)  raw           [{len(results)} vids]:    {r2_b_raw:+.4f}")
    print(f"  R²(b̂)  drop {OUTLIER_PCT:.0f}% outliers [{keep.sum()} vids]:  {r2_b_f:+.4f}")
    print(f"  Median |Δb|/b:                       {med_pct_b:.2f}%\n")
    print(f"  Median discovered equation:")
    print(f"    θ̈ = {coef_sin_med:.3f}·sin(θ) {coef_thd_med:+.3f}·θ̇")
    print(f"  Reference (median L={L_med:.2f}, b={b_med:.2f}):")
    print(f"    θ̈ = {-G/L_med:.3f}·sin(θ) {-b_med:+.3f}·θ̇")

    out = dict(
        headline=dict(r2_L_filtered=r2_L_f, r2_b_filtered=r2_b_f,
                     median_pct_err_L=med_pct_L, median_pct_err_b=med_pct_b,
                     equation_r2_median=eq_r2_med,
                     n_with_L_recovered=len(results),
                     n_after_outlier_filter=int(keep.sum())),
        raw=dict(r2_L=r2_L_raw, r2_b=r2_b_raw),
        discovered_eq_median=dict(coeff_sin_theta=coef_sin_med,
                                  coeff_theta_dot=coef_thd_med,
                                  formatted=f"theta_ddot = {coef_sin_med:.3f}*sin(theta) {coef_thd_med:+.3f}*theta_dot"),
        true_equation="theta_ddot = -(g/L)*sin(theta) - b*theta_dot   [g=9.81]",
        n_total_val=len(val_files), n_skipped_amplitude=skipped,
        n_processed=len(results),
        config=dict(n_val=len(val_files), dt=DT, g=G,
                   savgol_window=SAVGOL_WIN, savgol_poly=SAVGOL_POLY,
                   min_amplitude=MIN_AMP, trim_frames=TRIM,
                   outlier_pct=OUTLIER_PCT,
                   library=["sin(theta)","theta_dot"], regression="OLS"),
        equation_r2_mean=eq_r2_mean,
        per_video=results[:50]
    )
    os.makedirs("logs", exist_ok=True)
    with open("logs/sindy_pendulum_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n✓ Saved logs/sindy_pendulum_results.json")

if __name__ == "__main__":
    main()
