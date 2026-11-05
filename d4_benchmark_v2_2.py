#!/usr/bin/env python3
"""
Baker Geometric Robustness Benchmark v2.2 -- readout transfer, Lipschitz envelope, exact threshold.
Public construction only: D4(theta) = 2*arcsin((4/3)*sin(theta/8))
(CWBaker360/Constructible-Cubic-Trisection, paper/constructible_cubic_trisection.tex, Sec. 3)
v1, v2, v2.1 are frozen. v2.2 writes with prefix 'v22_'.

Target:  for a FUTURE construction, error W(theta) = b + U(theta) (no readout).  Certify the set of theta
in [10,120] deg on which  P(e4(theta) + W(theta) <= 0) <= ALPHA = 1e-3,  with >= 95% confidence.

(1) Readout transfer (proof).  Calibration sees X = W + Vbar (setup mean of k readings).
    If P(X <= L_X) <= ALPHA - BETA and P(Vbar > delta) <= BETA, then with L_W = L_X - delta:
      {W <= L_W} = {X - Vbar <= L_X - delta} subset of {X <= L_X} union {Vbar > delta},
    so P(W <= L_W) <= ALPHA.
    delta is NOT estimated from within-setup residuals (those are V_ij - Vbar_i, dependent, and not V).
    It comes from an INDEPENDENT reference experiment: n_ref blocks of k readings of a known reference
    angle, giving direct iid draws of Vbar.  delta = max of the n_ref draws (one-sided order statistic).
    Assumption (stated, not proved): the readout distribution is the same in the reference experiment
    and in the calibration experiments, and does not depend on theta.
(2) Confidence budget (Bonferroni, no independence needed): 5 node tolerance bounds + 1 readout bound
    = 6 events, each at confidence 1 - 0.05/6.
(3) Envelope (proof).  Hypothesis H_Lip: |W(theta) - W(phi)| <= Lam*|theta - phi| for every realisation.
    Then the lower ALPHA-quantile q(theta) is Lam-Lipschitz, and on the event that all node bounds hold,
      L(theta) = max_j ( L_j - Lam*|theta - theta_j| )  <=  q(theta)   for all theta in [10,120].
    Angular scenarios: W does not depend on theta (modelling hypothesis) => Lam = 0.
    Stage scenario: errors TRUNCATED at 4 sigma (Gaussian stage errors are unbounded, so no pathwise Lam
    exists without truncation); Lam from the analytic bound in lam_stage_bound().
(4) Exact threshold (proof).  e4'' > 0 on (0,120deg]: with f(u)=arcsin(k sin u), k=4/3,
      f''(u) = k (k^2-1) sin u / (1 - k^2 sin^2 u)^(3/2) > 0.
    L is piecewise linear (breakpoints: nodes and pairwise crossings), so on each piece
    M = e4 + L is strictly convex with slope e4' + s, s in {-Lam, 0, +Lam}; its minimum is at the left
    endpoint (s >= 0) or at theta_c solving e4'(theta_c) = Lam (s < 0).  The final certified interval
    [theta_min, 120] is found exactly by checking piece minima from the right and one brentq root.
    No grid is used for certification.  If M(120) <= 0: NO CERTIFICATE (not a geometric failure).
Methods compared:
    P          parametric normal tolerance on X, 5 nodes @0.99, no readout transfer   (conditional: Gaussian)
    NP_naive   v2.1 order statistic on X at ALPHA, 5 nodes @0.99, no readout transfer (shown to fail)
    NP_xfer    v2.2: order statistic on X at ALPHA-BETA, minus delta, 6 events @1-0.05/6
Scenarios: control, bias, heavy (t4), stage4 (truncated physical stage errors, angle-dependent),
           atom (regression test: the counterexample where naive X->W transfer fails).
Truth scoring (simulation only, not certification) uses the true CDF of W on a 0.05-deg grid.
Run: python3 d4_benchmark_v2_2.py
"""
import numpy as np, csv, sys, time, platform, json
from scipy import stats, optimize

SEED = 20261104 + 22
ARC = np.pi / (180 * 3600)
ALPHA, BETA, SIM_CONF = 1e-3, 1e-4, 0.95
NODES = np.radians([10, 30, 60, 90, 120]); TLO, TMAX = NODES[0], NODES[-1]
SIG_C, SIG_R, K_READ = 10 * ARC, 4 * ARC, 2
S_CH, S_SC, S_REL, S_ARC, TRUNC = 3e-5, 1.5e-5, 2e-5, 1 * ARC, 4.0
ATOM_A, ATOM_SIG, ATOM_D = 60 * ARC, 2 * ARC, 20 * ARC      # counterexample parameters
NREP_P, NREP_NP, CHUNK = 10_000, 2_000, 100
M_P = [8, 32]
SCORE_GRID = np.radians(np.arange(10, 120.0001, 0.05))

def D4(t): return 2 * np.arcsin(4 / 3 * np.sin(t / 8))
def e4(t): return D4(t) - t / 3
def e4p(t): return (4 / 3) * np.cos(t / 8) / 4 / np.sqrt(1 - (16 / 9) * np.sin(t / 8) ** 2) * 1 - 1 / 3
def e4pp(t):
    k, u = 4 / 3, t / 8
    return k * (k * k - 1) * np.sin(u) / (1 - k * k * np.sin(u) ** 2) ** 1.5 / 32

def n_req(level, conf):  # smallest n with 1-(1-level)^n >= conf  (extreme order statistic)
    return int(np.ceil(np.log(1 - conf) / np.log(1 - level)))
def np_order(m, p_tail, g):
    r = 1
    while stats.binom.sf(r, m, p_tail) >= g: r += 1
    assert stats.binom.sf(r - 1, m, p_tail) >= g
    return r
def K_param(m, p_tail, g): return stats.nct.ppf(g, m - 1, stats.norm.ppf(1 - p_tail) * np.sqrt(m)) / np.sqrt(m)
def cp_upper(x, n, g=0.95): return 1.0 if x >= n else float(stats.beta.ppf(g, x + 1, n - x))
def mcse(p, n): return float(np.sqrt(max(p * (1 - p), 1e-12) / max(n, 1)))

def tnorm(rng, sd, size):
    z = rng.standard_normal(size)
    bad = np.abs(z) > TRUNC
    while bad.any():
        z[bad] = rng.standard_normal(bad.sum()); bad = np.abs(z) > TRUNC
    return z * sd

def stage_W(t, rng, size):
    c = 2 * np.sin(t / 8)
    return (2 * np.arcsin(2 / 3 * (1 + tnorm(rng, S_REL, size)) * (c + tnorm(rng, S_CH, size))
                          + tnorm(rng, S_SC, size)) + tnorm(rng, S_ARC, size)) - D4(t)

def lam_stage_bound():
    """|dW/dtheta| <= Lam for all truncated stage errors (proof in header; bound derived as follows).
       W' = 2 x' [eta s(a) + s(a) - s(x)],  x = (2/3)c, x' = (2/3)c' = cos(theta/8)/6 <= 1/6
       s(z) = (1-z^2)^-1/2, s' = z s^3; a = (2/3)(1+eta)(c+dc)+ds.  |a-x| <= (2/3)(|eta|(c+|dc|)+|dc|)+|ds|."""
    eta, dc, ds = TRUNC * S_REL, TRUNC * S_CH, TRUNC * S_SC
    cmax = 2 * np.sin(TMAX / 8)
    amax = 2 / 3 * (1 + eta) * (cmax + dc) + ds
    smax, spmax = 1 / np.sqrt(1 - amax ** 2), amax / (1 - amax ** 2) ** 1.5
    da = 2 / 3 * (eta * (cmax + dc) + dc) + ds
    xp_max = 1 / 6                                   # x' = (2/3)*(1/4)cos(theta/8) <= 1/6
    return 2 * xp_max * (eta * smax + spmax * da)

SCEN = {
    "control": dict(b=0.0, U="norm", V="norm", lam=0.0),
    "bias":    dict(b=-5 * ARC, U="norm", V="norm", lam=0.0),
    "heavy":   dict(b=0.0, U="t4", V="t4", lam=0.0),
    "stage4":  dict(b=0.0, U="stage", V="norm", lam=lam_stage_bound()),
    "atom":    dict(b=0.0, U="atom", V="unif", lam=0.0),
}

def draw_U(rng, kind, t, size):
    if kind == "norm": return rng.normal(0, SIG_C, size)
    if kind == "t4": return rng.standard_t(4, size) * SIG_C / np.sqrt(2)
    if kind == "stage": return stage_W(t, rng, size)
    if kind == "atom":
        u = rng.normal(0, ATOM_SIG, size); u[rng.random(size) < ALPHA] = -ATOM_A; return u
def draw_Vbar(rng, kind, size):
    if kind == "norm": v = rng.normal(0, SIG_R, size + (K_READ,))
    elif kind == "t4": v = rng.standard_t(4, size + (K_READ,)) * SIG_R / np.sqrt(2)
    else: v = rng.uniform(-ATOM_D, ATOM_D, size + (K_READ,))
    return v.mean(-1)

class Truth:
    """true lower ALPHA-quantile of W(theta) and pointwise validity check"""
    def __init__(self, sc, rng):
        self.sc = sc; b = sc["b"]
        if sc["U"] == "norm": self.q = lambda t: np.full(np.shape(t), b + stats.norm.ppf(ALPHA) * SIG_C)
        elif sc["U"] == "t4": self.q = lambda t: np.full(np.shape(t), b + stats.t.ppf(ALPHA, 4) * SIG_C / np.sqrt(2))
        elif sc["U"] == "atom": self.q = lambda t: np.full(np.shape(t), b - ATOM_A)
        else:
            g = np.radians(np.arange(8, 121, 2.0))
            qs = np.array([np.quantile(stage_W(t, rng, 2_000_000), ALPHA) for t in g])
            self.q = lambda t: np.interp(t, g, qs) + b
    def ok(self, t):
        """True iff P(e4+W <= 0) <= ALPHA at theta.  Atom: F jumps at -A, so need e4 > A strictly."""
        if self.sc["U"] == "atom": return e4(t) > ATOM_A - self.sc["b"]
        return e4(t) + self.q(t) >= 0

THETA_C_CACHE = {}
def theta_c(lam):
    if lam not in THETA_C_CACHE:
        THETA_C_CACHE[lam] = optimize.brentq(lambda t: e4p(t) - lam, 1e-9, TMAX) if e4p(TMAX) > lam else np.inf
    return THETA_C_CACHE[lam]

def certify(Lj, lam):
    """exact final certified interval [theta_min, 120] for L(theta)=max_j(L_j-lam|theta-theta_j|); nan=none"""
    Lf = lambda t: np.max(Lj - lam * np.abs(t - NODES))
    M = lambda t: e4(t) + Lf(t)
    if M(TMAX) <= 0: return np.nan
    bps = set(NODES.tolist())
    if lam > 0:
        for i in range(len(NODES)):
            for j in range(i + 1, len(NODES)):
                for sgn_i in (-1, 1):
                    for sgn_j in (-1, 1):           # L_i - lam*sgn_i*(t-ti) = L_j - lam*sgn_j*(t-tj)
                        den = lam * (sgn_j - sgn_i)
                        if den != 0:
                            t = (Lj[j] - Lj[i] + lam * (sgn_j * NODES[j] - sgn_i * NODES[i])) / den
                            if TLO < t < TMAX: bps.add(float(t))
    bps = np.array(sorted(bps))
    tc = theta_c(lam) if lam > 0 else np.inf
    for u, v in zip(bps[-2::-1], bps[:0:-1]):       # pieces from the right
        mid = 0.5 * (u + v)
        slope = (Lf(mid + 1e-12 * (v - u)) - Lf(mid - 1e-12 * (v - u))) / (2e-12 * (v - u))
        tmin_pt = u if slope >= -1e-15 else min(max(tc, u), v)
        if M(tmin_pt) > 0 and M(u) > 0: continue        # whole piece certified (convex: min checked)
        # final interval starts inside (tmin_pt, v]: M increasing there
        if M(v) <= 0: return float(v)
        return float(optimize.brentq(M, tmin_pt, v, xtol=1e-15))
    return float(TLO)

def run():
    rng, rng_truth, rng_chk = (np.random.default_rng(SEED + i) for i in range(3))
    ev5, ev6 = 1 - (1 - SIM_CONF) / 5, 1 - (1 - SIM_CONF) / 6
    m_naive = n_req(ALPHA, ev5); m_x = n_req(ALPHA - BETA, ev6); n_ref = n_req(BETA, ev6)
    log = dict(seed=SEED, alpha=ALPHA, beta=BETA, conf_5events=ev5, conf_6events=ev6,
               m_NP_naive=m_naive, m_NP_xfer=m_x, n_ref_blocks=n_ref, k=K_READ,
               lam_stage4_rad_per_rad=SCEN["stage4"]["lam"],
               min_e4pp_on_domain=float(e4pp(np.linspace(1e-6, TMAX, 100001)).min()))
    # numerical sanity of Lambda bound (not a proof): finite-difference |dW/dtheta| over random draws
    t0, h = np.radians(np.linspace(10, 119, 30)), 1e-4
    r2 = np.random.default_rng(SEED + 99); mx = 0.0
    for t in t0:
        st = np.random.default_rng(r2.integers(1 << 62)).bit_generator.state
        a = np.random.default_rng(); a.bit_generator.state = st; w1 = stage_W(t, a, 200_000)
        a.bit_generator.state = st; w2 = stage_W(t + h, a, 200_000)
        mx = max(mx, np.max(np.abs(w2 - w1)) / h)
    log["lam_stage4_finite_difference_max"] = float(mx)
    print(json.dumps(log, indent=1), flush=True)
    rows, node_rows = [], []
    for name, sc in SCEN.items():
        truth = Truth(sc, rng_truth)
        qn = truth.q(NODES)
        ok_grid = truth.ok(SCORE_GRID)
        bad_from = np.flip(np.cumsum(np.flip(~ok_grid)))     # #bad grid points at index >= i
        def valid(tmin):
            i = np.searchsorted(SCORE_GRID, tmin)
            return True if i >= len(SCORE_GRID) else bad_from[i] == 0
        methods = [("P", m, NREP_P) for m in M_P] + [("NP_naive", m_naive, NREP_NP), ("NP_xfer", m_x, NREP_NP)]
        for meth, m, nrep in methods:
            Lnodes = np.empty((nrep, len(NODES))); delta = np.zeros(nrep)
            if meth == "P": Kf = K_param(m, ALPHA, ev5)
            elif meth == "NP_naive": r = np_order(m, ALPHA, ev5)
            else: r = np_order(m, ALPHA - BETA, ev6)
            for s in range(0, nrep, CHUNK):
                n = min(CHUNK, nrep - s)
                for j, t in enumerate(NODES):
                    x = sc["b"] + draw_U(rng, sc["U"], t, (n, m)) + draw_Vbar(rng, sc["V"], (n, m))
                    Lnodes[s:s + n, j] = (x.mean(1) - Kf * x.std(1, ddof=1)) if meth == "P" \
                        else np.partition(x, r - 1, 1)[:, r - 1]
                if meth == "NP_xfer":   # independent reference experiment, known reference angle
                    for q in range(n):
                        delta[s + q] = draw_Vbar(rng, sc["V"], (n_ref,)).max()
            Lw = Lnodes - delta[:, None]
            node_cov = Lw <= qn[None, :]
            tmins = np.array([certify(Lw[i], sc["lam"]) for i in range(nrep)])
            issued = np.isfinite(tmins)
            val = np.array([issued[i] and valid(tmins[i]) for i in range(nrep)])
            wrong = int((issued & ~val).sum()); simul = node_cov.all(1)
            pi = issued.mean()
            rows.append(dict(scenario=name, method=meth, m=m, reps=nrep, lam=sc["lam"],
                issued=pi, issued_se=mcse(pi, nrep),
                valid_given_issued=val[issued].mean() if issued.any() else np.nan,
                valid_all=val.mean(), wrong_cert_count=wrong,
                wrong_cert_rate=wrong / nrep, wrong_cert_upper95=cp_upper(wrong, nrep),
                simul_node_coverage=simul.mean(), simul_fail_upper95=cp_upper(int((~simul).sum()), nrep),
                delta_median_arcsec=np.median(delta) / ARC,
                theta_min_median_deg=np.degrees(np.nanmedian(tmins)) if issued.any() else np.nan,
                theta_min_p90_deg=np.degrees(np.nanpercentile(tmins, 90)) if issued.any() else np.nan,
                truth_theta_start_deg=float(np.degrees(SCORE_GRID[np.argmax(bad_from == 0)]))))
            tgt = ev6 if meth == "NP_xfer" else ev5
            for j, t in enumerate(NODES):
                c = node_cov[:, j].mean()
                node_rows.append(dict(scenario=name, method=meth, m=m, node_deg=round(np.degrees(t)),
                    node_coverage=c, node_coverage_se=mcse(c, nrep), target=tgt,
                    node_fail_upper95=cp_upper(int((~node_cov[:, j]).sum()), nrep),
                    coverage_pass=bool(c >= tgt - 2 * mcse(tgt, nrep))))
            R = rows[-1]
            print(f"  {name:7s} {meth:8s} m={m:5d} issued={R['issued']:.3f} valid|iss={R['valid_given_issued']:.4f} "
                  f"wrong={wrong} simul={R['simul_node_coverage']:.4f} tmin={R['theta_min_median_deg']:.2f} "
                  f"truth={R['truth_theta_start_deg']:.2f}", flush=True)
    # regression test: naive transfer must FAIL on the atom counterexample, v2.2 transfer must PASS
    nr = [r for r in node_rows if r["scenario"] == "atom"]
    naive_fail = not all(r["coverage_pass"] for r in nr if r["method"] == "NP_naive")
    xfer_pass = all(r["coverage_pass"] for r in nr if r["method"] == "NP_xfer")
    log["regression_atom"] = dict(naive_fails=naive_fail, xfer_passes=xfer_pass,
                                  PASS=bool(naive_fail and xfer_pass))
    print("REGRESSION (atom counterexample):", log["regression_atom"], flush=True)
    return rows, node_rows, log

if __name__ == "__main__":
    t0 = time.time()
    rows, node_rows, log = run()
    for fn, data in (("v22_results.csv", rows), ("v22_node_coverage.csv", node_rows)):
        with open(fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)
    log.update(python=sys.version.split()[0], numpy=np.__version__, scipy=__import__("scipy").__version__,
               platform=platform.platform(), runtime_s=round(time.time() - t0, 1))
    with open("v22_log.json", "w") as f: json.dump(log, f, indent=1, default=float)
    print(f"done in {time.time()-t0:.1f}s")
