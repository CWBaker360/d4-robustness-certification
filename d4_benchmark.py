#!/usr/bin/env python3
"""
Synthetic validation benchmark for the public D4 construction
    D4(theta) = 2*arcsin((4/3)*sin(theta/8))
(source: CWBaker360/Constructible-Cubic-Trisection, paper/constructible_cubic_trisection.tex, Sec. 3)

Model:  Y_ij = D4(theta) + b + U_i + V_ij,  i=1..m constructions, j=1..k readings.
Truth is known in simulation, so every procedure can be scored for coverage.

Procedures scored:
  A. One-sided 95% lower confidence bound (LCB) on the mean angular error mu = e4(theta)+b
     A1 exact t, m-1 df, SE^2 = MS_between/(m k)   (balanced design: Satterthwaite reduces to this)
     A2 z with the same SE                          (ignores small-m df)
     A3 naive iid z, SE^2 = s^2_all/(m k)           (pseudo-replication: treats readings as constructions)
  B. 95%-confidence / 99.9%-coverage LOWER tolerance bound L for a future construction error W = b + U
     Normal one-sided factor (noncentral t) applied to the m setup means rbar_i.
     rbar_i has variance sigma_C^2 + sigma_R^2/k >= sigma_C^2, so L is conservative for W.
  C. Propagation: overshoot certified for future constructions where e4(theta) + L > 0.
     theta_min solves e4(theta) = -L. Scored: true P(e4(theta_min)+W > 0) >= 0.999 ?
     (Angular-level margin e4 is used directly; no angular tolerance is fed into the chord bound.)

Scenarios:
  control  : Gaussian, b = 0                        (assumptions exactly met)
  bias     : Gaussian, b = -5 arcsec                (systematic undershoot error)
  heavy    : Student-t(4) errors scaled to same SD  (assumption violated: tails)
  stage    : construction errors generated at physical stages (chord, scale, calibration, arc)
             through the exact nonlinear D4 map; sigma_C then depends on theta.

Run:  python3 d4_benchmark.py   (seed fixed; writes d4_benchmark_results.csv and d4_benchmark_log.txt)
"""
import numpy as np, csv, sys, time, platform
from scipy import stats, optimize

SEED = 20261104
NREP = 10_000
ARC = np.pi / (180 * 3600)                 # 1 arcsec in rad
SIG_C, SIG_R = 10 * ARC, 4 * ARC           # illustrative construction / reading SDs
THETAS_DEG = [10, 30, 60, 90, 120]
MS, KS = [4, 8, 16, 32], [2, 4, 8]
P_COV, CONF = 0.999, 0.95
TMAX = np.radians(120)
# physical stage SDs (unit radius) for 'stage' scenario
S_CH, S_SC, S_REL, S_ARC = 3e-5, 1.5e-5, 2e-5, 1 * ARC

def D4(t):  return 2 * np.arcsin(4 / 3 * np.sin(t / 8))
def e4(t):  return D4(t) - t / 3

def D4_stage(t, dc, ds, eta, da):
    c = 2 * np.sin(t / 8)
    return 2 * np.arcsin(2 / 3 * (1 + eta) * (c + dc) + ds) + da

def theta_min(L):
    """root of e4(theta) = -L on (0, 120 deg]; e4 is increasing there (E4' > 0 argument)."""
    out = np.full(L.shape, np.nan)
    need = -L
    ok = (need > 0) & (need < e4(TMAX))
    out[need <= 0] = 0.0
    for idx in np.flatnonzero(ok):
        out[idx] = optimize.brentq(lambda t: e4(t) - need[idx], 1e-9, TMAX, xtol=1e-14)
    return out   # nan => no certified angle in the domain

def draw(rng, kind, size, sd):
    if kind == "heavy":
        nu = 4.0
        return rng.standard_t(nu, size) * sd / np.sqrt(nu / (nu - 2))
    return rng.normal(0, sd, size)

def true_W_quantile(scen, b, theta, rng_big):
    """true 0.1% quantile of a future construction error W at angle theta"""
    if scen in ("control", "bias"):
        return b + stats.norm.ppf(1 - P_COV) * SIG_C
    if scen == "heavy":
        nu = 4.0
        return b + stats.t.ppf(1 - P_COV, nu) * SIG_C / np.sqrt(nu / (nu - 2))
    n = 2_000_000
    w = D4_stage(theta, rng_big.normal(0, S_CH, n), rng_big.normal(0, S_SC, n),
                 rng_big.normal(0, S_REL, n), rng_big.normal(0, S_ARC, n)) - D4(theta)
    return np.quantile(w, 1 - P_COV)

def run():
    rng = np.random.default_rng(SEED)
    rng_big = np.random.default_rng(SEED + 1)
    rows = []
    z = stats.norm.ppf(CONF)
    scenarios = {"control": 0.0, "bias": -5 * ARC, "heavy": 0.0, "stage": 0.0}
    for scen, b in scenarios.items():
        # stage-scenario true construction-error quantile depends on theta: cache on a grid
        qgrid = None
        if scen == "stage":
            grid = np.radians(np.linspace(0.5, 120, 60))
            qgrid = (grid, np.array([true_W_quantile(scen, b, g, rng_big) for g in grid]))
        for th_deg in THETAS_DEG:
            th = np.radians(th_deg)
            for m in MS:
                for k in KS:
                    # ---- simulate calibration experiments ----
                    if scen == "stage":
                        sh = (NREP, m)
                        U = D4_stage(th, rng.normal(0, S_CH, sh), rng.normal(0, S_SC, sh),
                                     rng.normal(0, S_REL, sh), rng.normal(0, S_ARC, sh)) - D4(th)
                        V = rng.normal(0, SIG_R, (NREP, m, k))
                        sigC_true = np.std(U)        # empirical, for scoring estimators
                        Wmean = U.mean()             # nonlinear mean shift (tiny)
                    else:
                        U = draw(rng, scen, (NREP, m), SIG_C)
                        V = draw(rng, scen, (NREP, m, k), SIG_R)
                        sigC_true, Wmean = SIG_C, 0.0
                    Y = D4(th) + b + U[:, :, None] + V
                    r = Y - th / 3                   # observed minus exact trisector
                    rbar_i = r.mean(2)
                    rbar = rbar_i.mean(1)
                    MSB = k * rbar_i.var(1, ddof=1)
                    MSW = ((r - rbar_i[:, :, None]) ** 2).sum((1, 2)) / (m * (k - 1))
                    sC2_raw = (MSB - MSW) / k
                    sC2 = np.maximum(0, sC2_raw)
                    mu_true = e4(th) + b + Wmean
                    # ---- A: LCB on mean error ----
                    se = np.sqrt(MSB / (m * k))
                    tq = stats.t.ppf(CONF, m - 1)
                    cov_t = np.mean(rbar - tq * se <= mu_true)
                    cov_z = np.mean(rbar - z * se <= mu_true)
                    se_naive = np.sqrt(r.reshape(NREP, -1).var(1, ddof=1) / (m * k))
                    cov_naive = np.mean(rbar - z * se_naive <= mu_true)
                    cert_rate = np.mean(rbar - tq * se > 0)   # mean-overshoot declared
                    # ---- B: tolerance bound for future construction error W ----
                    Kt = stats.nct.ppf(CONF, m - 1, stats.norm.ppf(P_COV) * np.sqrt(m)) / np.sqrt(m)
                    w_i = rbar_i - e4(th)            # requires known theta: calibration reference
                    Lw = w_i.mean(1) - Kt * w_i.std(1, ddof=1)
                    qW = true_W_quantile(scen, b, th, rng_big) if scen != "stage" \
                        else np.interp(th, *qgrid)
                    tol_ok = np.mean(Lw <= qW)
                    # ---- C: propagate to theta_min and score validity at theta_min ----
                    tmin = theta_min(Lw)
                    have = np.isfinite(tmin) & (tmin > 0)
                    if scen == "stage":
                        q_at = np.interp(tmin[have], *qgrid)
                    else:
                        q_at = np.full(have.sum(), qW)
                    valid = np.mean(e4(tmin[have]) + q_at >= 0) if have.any() else np.nan
                    mc = lambda p: np.sqrt(p * (1 - p) / NREP)
                    rows.append(dict(
                        scenario=scen, theta_deg=th_deg, m=m, k=k, obs=m * k,
                        cov_lcb_t=cov_t, cov_lcb_z=cov_z, cov_lcb_naive=cov_naive,
                        mc_se_cov=mc(CONF), mean_overshoot_cert_rate=cert_rate,
                        sigC_hat_bias_arcsec=(np.sqrt(sC2).mean() - sigC_true) / ARC,
                        sigC_hat_rmse_arcsec=np.sqrt(np.mean((np.sqrt(sC2) - sigC_true) ** 2)) / ARC,
                        sigR_hat_bias_arcsec=(np.sqrt(MSW).mean() - SIG_R) / ARC,
                        neg_sigC2_frac=np.mean(sC2_raw < 0),
                        tol_conf_achieved=tol_ok, tol_L_median_arcsec=np.median(Lw) / ARC,
                        theta_min_median_deg=np.degrees(np.nanmedian(tmin)) if have.any() else np.nan,
                        no_cert_frac=np.mean(~np.isfinite(tmin)),
                        theta_min_valid_frac=valid))
            print(f"  {scen:8s} theta={th_deg:3d} done", flush=True)
    return rows

if __name__ == "__main__":
    t0 = time.time()
    rows = run()
    with open("d4_benchmark_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with open("d4_benchmark_log.txt", "w") as f:
        f.write(f"seed={SEED} NREP={NREP} python={sys.version.split()[0]} numpy={np.__version__} "
                f"scipy={__import__('scipy').__version__} platform={platform.platform()}\n")
        f.write(f"sigma_C={SIG_C/ARC} arcsec sigma_R={SIG_R/ARC} arcsec; stage SDs chord={S_CH} "
                f"scale={S_SC} rel={S_REL} arc={S_ARC/ARC} arcsec; runtime={time.time()-t0:.1f}s\n")
    print(f"done in {time.time()-t0:.1f}s")
