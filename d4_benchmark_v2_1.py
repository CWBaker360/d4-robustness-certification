#!/usr/bin/env python3
"""
Baker Geometric Robustness Benchmark v2.1 (Bonferroni default) -- Angle-Dependent and Distribution-Robust Certification

Public construction only:  D4(theta) = 2*arcsin((4/3)*sin(theta/8))
(CWBaker360/Constructible-Cubic-Trisection, paper/constructible_cubic_trisection.tex, Sec. 3)
v1 and v2 are frozen; v2.1 writes with prefix 'v21_<rule>_'.
Changes from v2: --rule bonferroni (DEFAULT, per-node conf 1-0.05/5 = 0.99, valid WITHOUT independence)
  or --rule product (per-node 0.95^(1/5), exact only for independent node experiments; comparison only);
  Clopper-Pearson one-sided 95% upper bounds on failure rates; per-node coverage pass/fail flags.

Model (calibration):  Y_ij = D4(theta_g) + b + U_i + V_ij,  m setups x k readings at node angle theta_g.
Target quantity:      W = b + U, error of a FUTURE construction (no readout), at any theta in [10,120] deg.
Goal:                 lower bound L(theta) with  P(W(theta) >= L(theta)) >= 0.999  (coverage),
                      simultaneously over the target domain with confidence >= 95%.

Certification design
  * Calibration at G = 5 node angles {10,30,60,90,120} deg, independent experiments.
  * Bonferroni: per-node confidence 0.99 => simultaneous >= 0.95 with no independence assumption.
  * Envelope between nodes: L_env(theta) = min(L_a, L_b) on [theta_a, theta_b].
    Valid between nodes ONLY under the stated assumption that the 0.1% quantile of W is monotone in
    theta between adjacent nodes. Below 10 deg: no certificate is issued (outside calibrated domain).
  * Certified set: theta on a 0.25-deg grid with e4(theta) + L_env(theta) > 0; theta_min = start of the
    final certified interval ending at 120 deg.
  * Methods:  P  = parametric normal one-sided tolerance factor (noncentral t) on setup means
              NP = distribution-free order statistic on setup means:  L = X_(r),
                   coverage confidence = P(Binomial(m, 1-p) >= r)
    Both are applied to setup means rbar_i (variance includes readout/k); validity is scored against W.

Outcomes per run (reported separately, each with Monte Carlo SE):
  issued      : a certified interval exists
  valid|issued: every certified grid angle truly satisfies e4(theta) + q_W(theta) >= 0
  valid_all   : issued AND valid   (no-certificate is NOT counted as an incorrect certificate;
                                    wrong_cert = issued AND not valid is reported explicitly)
Run:  python3 d4_benchmark_v2_1.py            (Bonferroni default)
      python3 d4_benchmark_v2_1.py --rule product   (independent-node comparison)
"""
import numpy as np, csv, sys, time, platform, json
from scipy import stats, optimize

SEED = 20261104 + 21
ARC = np.pi / (180 * 3600)
SIG_C, SIG_R, K_READ = 10 * ARC, 4 * ARC, 2
P_COV, SIM_CONF = 0.999, 0.95
NODES_DEG = [10, 30, 60, 90, 120]
RULE = sys.argv[sys.argv.index('--rule') + 1] if '--rule' in sys.argv else 'bonferroni'
assert RULE in ('bonferroni', 'product')
NODE_CONF = 1 - (1 - SIM_CONF) / len(NODES_DEG) if RULE == 'bonferroni' else SIM_CONF ** (1 / len(NODES_DEG))
TMAX = np.radians(120)
GRID = np.radians(np.arange(10, 120.0001, 0.25))
S_CH, S_SC, S_REL, S_ARC = 3e-5, 1.5e-5, 2e-5, 1 * ARC
SCEN = {"control": 0.0, "bias": -5 * ARC, "heavy": 0.0, "stage": 0.0}
NREP_P, NREP_NP = 10_000, 2_000
M_P = [4, 8, 16, 32]

def D4(t): return 2 * np.arcsin(4 / 3 * np.sin(t / 8))
def e4(t): return D4(t) - t / 3
def D4_stage(t, rng, size):
    c = 2 * np.sin(t / 8)
    return (2 * np.arcsin(2 / 3 * (1 + rng.normal(0, S_REL, size)) * (c + rng.normal(0, S_CH, size))
                          + rng.normal(0, S_SC, size)) + rng.normal(0, S_ARC, size))

def K_param(m, p, g): return stats.nct.ppf(g, m - 1, stats.norm.ppf(p) * np.sqrt(m)) / np.sqrt(m)
def np_order(m, p, g):
    """largest r with P(Bin(m,1-p) >= r) >= g ; 0 => no valid order statistic"""
    r = int(stats.binom.ppf(1 - g, m, 1 - p))        # P(Bin <= r-1) <= 1-g
    while r > 0 and stats.binom.sf(r - 1, m, 1 - p) < g: r -= 1
    while stats.binom.sf(r, m, 1 - p) >= g: r += 1
    return r
def n_required(p, g):
    """smallest m for which the minimum (r=1) gives confidence g: 1 - p^m >= g"""
    return int(np.ceil(np.log(1 - g) / np.log(p)))

def sample_setups(rng, scen, b, th, reps, m):
    """setup means rbar_i - e4(theta) = b + U_i + Vbar_i  (calibration needs known theta)"""
    if scen == "stage":
        U = D4_stage(th, rng, (reps, m)) - D4(th)
    elif scen == "heavy":
        U = rng.standard_t(4, (reps, m)) * SIG_C / np.sqrt(2)
    else:
        U = rng.normal(0, SIG_C, (reps, m))
    if scen == "heavy":
        V = rng.standard_t(4, (reps, m, K_READ)) * SIG_R / np.sqrt(2)
    else:
        V = rng.normal(0, SIG_R, (reps, m, K_READ))
    return b + U + V.mean(2)

def true_qW(scen, b, rng_big):
    """true 0.1% quantile of future-construction error W on GRID (and nodes)"""
    if scen in ("control", "bias"): return lambda t: np.full(np.shape(t), b + stats.norm.ppf(1 - P_COV) * SIG_C)
    if scen == "heavy": return lambda t: np.full(np.shape(t), b + stats.t.ppf(1 - P_COV, 4) * SIG_C / np.sqrt(2))
    g = np.radians(np.arange(5, 121, 2.5))
    q = np.array([np.quantile(D4_stage(t, rng_big, 2_000_000) - D4(t), 1 - P_COV) for t in g])
    return lambda t: np.interp(t, g, q)

def envelope(Lnodes):
    """Lnodes: (reps, G) -> L_env on GRID: min of the two bracketing nodes"""
    nodes = np.radians(NODES_DEG)
    idx = np.clip(np.searchsorted(nodes, GRID, side="right") - 1, 0, len(nodes) - 2)
    at_node = np.isclose(GRID[:, None], nodes[None, :]).any(1)
    La, Lb = Lnodes[:, idx], Lnodes[:, idx + 1]
    Lenv = np.minimum(La, Lb)
    # exactly at a node use that node's bound
    for j, t in enumerate(nodes):
        Lenv[:, np.isclose(GRID, t)] = Lnodes[:, [j]]
    return Lenv

def score(Lnodes, qW):
    Lenv = envelope(Lnodes)
    M = e4(GRID)[None, :] + Lenv > 0
    # final certified interval ending at 120 deg
    fail_from_end = np.flip(~M, 1)
    has_fail = fail_from_end.any(1)
    n_ok_tail = np.where(has_fail, fail_from_end.argmax(1), M.shape[1])
    issued = n_ok_tail > 0
    start = M.shape[1] - n_ok_tail
    truth_ok = e4(GRID) + qW(GRID) >= 0                     # truly >= 99.9% overshoot at theta
    # valid iff every certified grid angle in the final interval is truly ok
    bad_cum = np.flip(np.cumsum(np.flip(~truth_ok)))        # #bad at index >= i
    valid = issued & (bad_cum[np.minimum(start, M.shape[1] - 1)] == 0)
    tmin = np.where(issued, np.degrees(GRID[np.minimum(start, M.shape[1] - 1)]), np.nan)
    node_cov = (Lnodes <= qW(np.radians(NODES_DEG))[None, :])
    return issued, valid, tmin, node_cov

def mcse(p, n): return float(np.sqrt(max(p * (1 - p), 1e-12) / n))
def cp_upper(x, n, g=0.95):
    """Clopper-Pearson one-sided upper bound on a failure probability (x failures in n)"""
    return 1.0 if x >= n else float(stats.beta.ppf(g, x + 1, n - x))

def verify_factors(rng):
    out = {}
    out["K(10,.99,.95) vs table 3.981"] = K_param(10, .99, .95)
    out["K(20,.95,.95) vs table 2.396"] = K_param(20, .95, .95)
    for m in (4, 32):                                        # Monte Carlo check of the factor itself
        K = K_param(m, P_COV, NODE_CONF)
        x = rng.normal(size=(200_000, m))
        out[f"MC coverage of K_param(m={m}) target {NODE_CONF:.5f}"] = float(
            np.mean(x.mean(1) - K * x.std(1, ddof=1) <= stats.norm.ppf(1 - P_COV)))
    m = n_required(P_COV, NODE_CONF); r = np_order(m, P_COV, NODE_CONF)
    u = rng.uniform(size=(20_000, m))
    out[f"MC coverage of NP order stat (m={m}, r={r}) target {NODE_CONF:.5f}"] = float(
        np.mean(np.partition(u, r - 1, 1)[:, r - 1] <= 1 - P_COV))
    return out

def run():
    rng, rng_big, rng_chk = (np.random.default_rng(SEED + i) for i in range(3))
    log = {"seed": SEED, "rule": RULE, "node_conf": NODE_CONF,
           "simultaneous_conf_guarantee": (1 - len(NODES_DEG) * (1 - NODE_CONF)) if RULE == "bonferroni" else None,
           "simultaneous_conf_if_independent": NODE_CONF ** len(NODES_DEG)}
    nreq = {f"p={P_COV},conf={g}": n_required(P_COV, g) for g in (0.95, NODE_CONF, 0.99)}
    log["NP minimum sample sizes (r=1)"] = nreq
    print("NP sample sizes required:", nreq, flush=True)
    log["factor verification"] = verify_factors(rng_chk)
    print(json.dumps(log["factor verification"], indent=1), flush=True)
    M_NP = sorted({n_required(P_COV, NODE_CONF), 8000})
    rows, node_rows = [], []
    for scen, b in SCEN.items():
        qW = true_qW(scen, b, rng_big)
        log[f"truth {scen}: certifiable domain start (deg)"] = float(
            np.degrees(GRID[np.argmax(e4(GRID) + qW(GRID) >= 0)]))
        for meth, ms, nrep in (("P", M_P, NREP_P), ("NP", M_NP, NREP_NP)):
            for m in ms:
                Lnodes = np.empty((nrep, len(NODES_DEG)))
                if meth == "P": K = K_param(m, P_COV, NODE_CONF)
                else: r = np_order(m, P_COV, NODE_CONF)
                for j, td in enumerate(NODES_DEG):
                    th = np.radians(td)
                    for s in range(0, nrep, 500):
                        n = min(500, nrep - s)
                        x = sample_setups(rng, scen, b, th, n, m)
                        Lnodes[s:s + n, j] = (x.mean(1) - K * x.std(1, ddof=1)) if meth == "P" \
                            else np.partition(x, r - 1, 1)[:, r - 1]
                issued, valid, tmin, node_cov = score(Lnodes, qW)
                pi, pv = issued.mean(), valid.mean()
                pvi = valid[issued].mean() if issued.any() else np.nan
                simul = node_cov.all(1).mean()
                rows.append(dict(scenario=scen, method=meth, m=m, k=K_READ, reps=nrep,
                                 issued=pi, issued_se=mcse(pi, nrep),
                                 valid_given_issued=pvi,
                                 valid_given_issued_se=mcse(pvi, issued.sum()) if issued.any() else np.nan,
                                 valid_all=pv, valid_all_se=mcse(pv, nrep),
                                 wrong_cert=(issued & ~valid).mean(),
                                 wrong_cert_se=mcse((issued & ~valid).mean(), nrep),
                                 wrong_cert_count=int((issued & ~valid).sum()),
                                 wrong_cert_upper95=cp_upper(int((issued & ~valid).sum()), nrep),
                                 simul_fail_upper95=cp_upper(int((~node_cov.all(1)).sum()), nrep),
                                 simul_node_coverage=simul, simul_node_coverage_se=mcse(simul, nrep),
                                 theta_min_median_deg=np.nanmedian(tmin) if issued.any() else np.nan,
                                 theta_min_p90_deg=np.nanpercentile(tmin, 90) if issued.any() else np.nan))
                for j, td in enumerate(NODES_DEG):
                    c = node_cov[:, j].mean()
                    node_rows.append(dict(scenario=scen, method=meth, m=m, node_deg=td,
                                          node_coverage=c, node_coverage_se=mcse(c, nrep),
                                          target=NODE_CONF,
                                          node_fail_upper95=cp_upper(int((~node_cov[:, j]).sum()), nrep),
                                          coverage_pass=bool(c >= NODE_CONF - 2 * mcse(NODE_CONF, nrep))))
                print(f"  {scen:8s} {meth:2s} m={m:5d} issued={pi:.3f} valid|iss={pvi:.4f} "
                      f"simul={simul:.4f} tmin={rows[-1]['theta_min_median_deg']:.2f}", flush=True)
    return rows, node_rows, log

if __name__ == "__main__":
    t0 = time.time()
    rows, node_rows, log = run()
    for name, data in ((f"v21_{RULE}_results.csv", rows), (f"v21_{RULE}_node_coverage.csv", node_rows)):
        with open(name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)
    log.update(python=sys.version.split()[0], numpy=np.__version__,
               scipy=__import__("scipy").__version__, platform=platform.platform(),
               runtime_s=round(time.time() - t0, 1))
    with open(f"v21_{RULE}_log.json", "w") as f: json.dump(log, f, indent=1, default=float)
    print(f"done in {time.time()-t0:.1f}s")
