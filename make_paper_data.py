#!/usr/bin/env python3
"""Generate every numerical table in the manuscript from frozen benchmark outputs and exact formulas.
Run from paper/:  python3 make_paper_data.py   -> tables/*.tex, paper_data.json
Inputs (frozen): ../d4_benchmark_results.csv (v1), ../v21_bonferroni_results.csv, ../v22_results.csv,
                 ../v22_node_coverage.csv, and the benchmark sources for hashing."""
import hashlib, json, os, math
import pandas as pd
from mpmath import iv, mp
from scipy import optimize
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(os.path.dirname(__file__), "tables"); os.makedirs(OUT, exist_ok=True)
iv.dps = mp.dps = 40
data = {}

def sci(x):  # "5e-5" -> "5\\times10^{-5}", "1e-8" -> "10^{-8}"
    m, e = x.split("e"); return (f"10^{{{int(e)}}}" if m == "1" else f"{m}\\times10^{{{int(e)}}}")

def w(name, body):
    open(os.path.join(OUT, name + ".tex"), "w").write(body)

# ---------- interval enclosures (outward-rounded, mpmath.iv) ----------
PI = iv.pi
def rad(deg_str): return iv.mpf(deg_str) * PI / 180
def E4_iv(t): return 2 * iv.sin(t / 8) - iv.mpf(3) / 2 * iv.sin(t / 6)
def F_iv(t, ec, es, ea, r):
    return (1 - r) * (2 * iv.sin(t / 8) - ec) - iv.mpf(3) / 2 * es - iv.mpf(3) / 2 * iv.sin(t / 6 + ea / 2)

def dec_bracket(x, d):
    """decimal strings lo < x < hi with d decimals, one ulp-of-display outside the float root"""
    q = 10 ** d
    return f"{math.floor(x * q - 1) / q:.{d}f}", f"{math.ceil(x * q + 1) / q:.{d}f}"

def enclose(f_float, f_iv, lo, hi, d=6, unit="deg"):
    """float root via brentq, then CERTIFY the displayed decimal endpoints themselves:
       f_iv(endpoint) strictly < 0 at the lower and > 0 at the upper, in outward-rounded interval arithmetic.
       The endpoints are exact decimals in `unit` (deg -> converted to rad as an interval).
       Uniqueness of the root is supplied by the monotonicity / convexity theorems in the paper."""
    x = optimize.brentq(f_float, lo, hi, xtol=1e-15)
    xd = math.degrees(x) if unit == "deg" else x
    lo_s, hi_s = dec_bracket(xd, d)
    to_arg = (lambda z: rad(z)) if unit == "deg" else (lambda z: iv.mpf(z))
    fa, fb = f_iv(to_arg(lo_s)), f_iv(to_arg(hi_s))
    assert fa.b < 0 and fb.a > 0, (lo_s, hi_s, fa, fb)
    return dict(lo=lo_s, hi=hi_s)

def outward(z, d):
    """interval z -> decimal strings rounded outward (floor of lower, ceil of upper) at d decimals"""
    q = mp.mpf(10) ** d
    return mp.nstr(mp.floor(z.a * q) / q, 30, strip_zeros=False)[:], mp.nstr(mp.ceil(z.b * q) / q, 30)

E4f = lambda t: 2 * math.sin(t / 8) - 1.5 * math.sin(t / 6)
Ff = lambda t, ec, es, ea, r: (1 - r) * (2 * math.sin(t / 8) - ec) - 1.5 * es - 1.5 * math.sin(t / 6 + ea / 2)
T120 = math.radians(120)

rows = []
for e in ["1e-8", "1e-6", "1e-4"]:
    ef = float(e); ei = iv.mpf(e)
    enc = enclose(lambda t: E4f(t) - ef, lambda t: E4_iv(t) - ei, 1e-6, T120, d=9)
    cubic = math.degrees(24 / 7 ** (1 / 3) * ef ** (1 / 3))
    rows.append((e, enc, cubic))
data["chord_thresholds"] = [(e, enc, c) for e, enc, c in rows]
body = "\\begin{tabular}{lccc}\n\\toprule\n$\\varepsilon_c$ & certified enclosure of $\\theta_{\\min}$ (deg) & cubic estimate (deg) \\\\\n\\midrule\n"
for e, enc, c in rows:
    body += f"$10^{{{int(e.split('e')[1])}}}$ & $[{enc['lo']},\\ {enc['hi']}]$ & {c:.6f} \\\\\n"
w("t_chord", body + "\\bottomrule\n\\end{tabular}\n")

cap = E4_iv(rad("120")); c120 = 2 * iv.sin(rad("120") / 8)
data["eps_c_cap"] = [str(cap.a), str(cap.b)]
data["r_cap"] = [str((cap / c120).a), str((cap / c120).b)]

models = {  # (eps_c, eps_s, eps_a [rad, string expression], r)  unit radius; drafting models at R = 100 mm
    "Fine numerical/CAD": ("1e-8", "1e-9", ("rad", "1e-8"), "1e-7"),
    "Controlled drafting (R = 100 mm)": ("1e-4", "5e-5", ("deg", "0.01"), "1e-4"),
    "Coarse drafting (R = 100 mm)": ("5e-4", "2e-4", ("deg", "0.05"), "1e-3"),
}
body = "\\begin{tabular}{lcccc c}\n\\toprule\nModel & $\\varepsilon_c$ & $\\varepsilon_s$ & $\\varepsilon_a$ & $r$ & enclosure of $\\theta_{\\min}$ (deg) \\\\\n\\midrule\n"
data["combined"] = {}
for name, (ec, es, (unit, ea), r) in models.items():
    eci, esi, ri = iv.mpf(ec), iv.mpf(es), iv.mpf(r)
    eai = iv.mpf(ea) if unit == "rad" else rad(ea)
    eaf = float(ea) if unit == "rad" else math.radians(float(ea))
    enc = enclose(lambda t: Ff(t, float(ec), float(es), eaf, float(r)),
                  lambda t: F_iv(t, eci, esi, eai, ri), 1e-6, T120, d=7)
    data["combined"][name] = enc
    eas = f"${sci(ea)}$ rad" if unit == "rad" else f"${ea}^\\circ$"
    body += f"{name} & ${sci(ec)}$ & ${sci(es)}$ & {eas} & ${sci(r)}$ & $[{enc['lo']},\\ {enc['hi']}]$ \\\\\n"
w("t_combined", body + "\\bottomrule\n\\end{tabular}\n")

# critical radius for the pencil budget: 0.30 mm, 0.15 mm, 0.1%, 0.10 deg
def Fr_f(R): return Ff(T120, 0.3 / R, 0.15 / R, math.radians(0.1), 1e-3)
def Fr_i(R): return F_iv(rad("120"), iv.mpf("0.3") / R, iv.mpf("0.15") / R, rad("0.1"), iv.mpf("1e-3"))
rc = enclose(Fr_f, Fr_i, 100, 1000, d=7, unit="mm")
data["R_crit_mm"] = [rc["lo"], rc["hi"]]

def endp(z, which):  # endpoint of an mpmath interval as an exact mpf
    t = str(z.a if which == "a" else z.b).strip("[]").split(",")
    return mp.mpf(t[0] if which == "a" else t[-1])
def out_dec(z, d):    # outward-rounded decimal display of interval z
    mp.dps = 50; q = mp.mpf(10) ** d
    lo, hi = mp.floor(endp(z, "a") * q) / q, mp.ceil(endp(z, "b") * q) / q
    return (mp.nstr(lo, d + 3, strip_zeros=False).rstrip("0"), mp.nstr(hi, d + 3, strip_zeros=False).rstrip("0"))
ecap = out_dec(cap, 13); rcap = out_dec(cap / c120, 9)
macros = (f"\\newcommand{{\\EcapLo}}{{{ecap[0]}}}\n\\newcommand{{\\EcapHi}}{{{ecap[1]}}}\n"
          f"\\newcommand{{\\RcapLo}}{{{rcap[0]}}}\n\\newcommand{{\\RcapHi}}{{{rcap[1]}}}\n"
          f"\\newcommand{{\\RcritLo}}{{{rc['lo']}}}\n\\newcommand{{\\RcritHi}}{{{rc['hi']}}}\n"
          f"\\newcommand{{\\ArcGate}}{{DATA_GATE}}\n")
# angular gate: eps_a < 2 asin((4/3)(1-r) sin 15deg) - 40deg, r = 1e-3 (asin evaluated in high precision mp)
mp.dps = 40
gate = 2 * mp.asin(mp.mpf(4) / 3 * (1 - mp.mpf("1e-3")) * mp.sin(mp.pi / 12)) - 40 * mp.pi / 180
data["eps_a_gate_deg_r1e-3"] = float(gate * 180 / mp.pi)
data_gate = data["eps_a_gate_deg_r1e-3"]
w("macros", macros.replace("DATA_GATE", f"{data_gate:.4f}"))

# ---------- v1 ----------
v1 = pd.read_csv(os.path.join(ROOT, "d4_benchmark_results.csv"))
g = v1.groupby(["scenario", "m"])[["cov_lcb_t", "cov_lcb_z", "cov_lcb_naive", "tol_conf_achieved"]].mean()
body = "\\begin{tabular}{llcccc}\n\\toprule\nScenario & $m$ & exact $t$ & $z$ & naive iid $z$ & Gaussian tolerance \\\\\n\\midrule\n"
for (s, m), r in g.iterrows():
    if m in (4, 32):
        body += f"{s} & {m} & {100*r.cov_lcb_t:.2f}\\% & {100*r.cov_lcb_z:.2f}\\% & {100*r.cov_lcb_naive:.2f}\\% & {100*r.tol_conf_achieved:.2f}\\% \\\\\n"
w("t_v1", body + "\\bottomrule\n\\end{tabular}\n")
a60 = v1[(v1.scenario == "control") & (v1.theta_deg == 60) & (v1.obs == 32)]
body = "\\begin{tabular}{ccc}\n\\toprule\nconstructions $m$ & readings $k$ & median $\\theta_{\\min}$ (deg) \\\\\n\\midrule\n"
for _, r in a60.sort_values("m", ascending=False).iterrows():
    body += f"{r.m} & {r.k} & {r.theta_min_median_deg:.2f} \\\\\n"
w("t_alloc", body + "\\bottomrule\n\\end{tabular}\n")

# ---------- v2.1 ----------
v21 = pd.read_csv(os.path.join(ROOT, "v21_bonferroni_results.csv"))
body = "\\begin{tabular}{llrccc}\n\\toprule\nScenario & method & $m$ & simultaneous node coverage & wrong certificates & median $\\theta_{\\min}$ \\\\\n\\midrule\n"
for _, r in v21.iterrows():
    if r.method == "NP" or r.m in (4, 32):
        body += (f"{r.scenario} & {r.method} & {r.m} & {100*r.simul_node_coverage:.2f}\\% & "
                 f"{int(round(r.wrong_cert*r.reps))}/{r.reps} & {r.theta_min_median_deg:.2f}$^\\circ$ \\\\\n")
w("t_v21", body + "\\bottomrule\n\\end{tabular}\n")

# ---------- v2.2 ----------
v22 = pd.read_csv(os.path.join(ROOT, "v22_results.csv"))
lab = {"P": "Gaussian", "NP_naive": "NP naive", "NP_xfer": "NP transfer"}
body = ("\\begin{tabular}{llrccccc}\n\\toprule\nScenario & method & $m$ & simult.\\ cov. & wrong / runs & 95\\% upper & "
        "$\\theta_{\\min}$ & truth \\\\\n\\midrule\n")
for _, r in v22.iterrows():
    body += (f"{r.scenario} & {lab[r.method]} & {r.m} & {100*r.simul_node_coverage:.2f}\\% & {r.wrong_cert_count}/{r.reps} & "
             f"{100*r.wrong_cert_upper95:.2f}\\% & {r.theta_min_median_deg:.2f}$^\\circ$ & {r.truth_theta_start_deg:.2f}$^\\circ$ \\\\\n")
w("t_v22", body + "\\bottomrule\n\\end{tabular}\n")
data["v22_delta_arcsec"] = v22[v22.method == "NP_xfer"].set_index("scenario").delta_median_arcsec.round(2).to_dict()
log22 = json.load(open(os.path.join(ROOT, "v22_log.json")))
data["v22_log"] = {k: log22[k] for k in ("m_NP_xfer", "n_ref_blocks", "lam_stage4_rad_per_rad",
                                         "lam_stage4_finite_difference_max", "regression_atom", "seed")}

# ---------- exact calibration cost ----------
def n_req_exact(level, conf):
    mp.dps = 50
    n = int(math.ceil(math.log(1 - conf) / math.log(1 - level)))
    ok = lambda n: 1 - (1 - mp.mpf(level)) ** n >= mp.mpf(conf)
    while not ok(n): n += 1
    while ok(n - 1): n -= 1
    return n
ev6 = 1 - 0.05 / 6
body = ("\\begin{tabular}{cccccc}\n\\toprule\nPopulation coverage $1-\\alpha$ & $\\beta=\\alpha/10$ & setups per node & "
        "total setups (5 nodes) & reference blocks & conf.\\ per event \\\\\n\\midrule\n")
data["cost"] = []
for alpha in ("0.01", "0.005", "0.001"):
    a_ = float(alpha); b_ = a_ / 10
    mx = n_req_exact(a_ - b_, ev6); nr = n_req_exact(b_, ev6)
    data["cost"].append(dict(alpha=a_, beta=b_, m_x=mx, total=5 * mx, n_ref=nr))
    body += f"{100*(1-a_):.1f}\\% & {b_:g} & {mx:,} & {5*mx:,} & {nr:,} & $1-0.05/6$ \\\\\n"
w("t_cost", body + "\\bottomrule\n\\end{tabular}\n")

# ---------- hashes ----------
files = ["d4_benchmark.py", "d4_benchmark_results.csv", "d4_benchmark_v2.py", "v2_results.csv",
         "d4_benchmark_v2_1.py", "v21_bonferroni_results.csv", "v21_product_results.csv",
         "d4_benchmark_v2_2.py", "v22_results.csv", "v22_node_coverage.csv", "v22_log.json"]
hashes = {f: hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest() for f in files}
data["sha256"] = hashes
body = "\\begin{tabular}{ll}\n\\toprule\nFile & SHA-256 \\\\\n\\midrule\n"
for f, h in hashes.items():
    body += f"\\texttt{{{f.replace('_', chr(92)+'_')}}} & {{\\tiny\\texttt{{{h}}}}} \\\\\n"
w("t_hash", body + "\\bottomrule\n\\end{tabular}\n")
json.dump(data, open(os.path.join(os.path.dirname(__file__), "paper_data.json"), "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in data.items() if k != "sha256"}, indent=1, default=str))
