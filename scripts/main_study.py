"""Quantum kernel learning, v6.1: v6 with the reviewer's remaining corrections applied (design and primary outcome unchanged).

Changes relative to v6
  A. timing: the ignored control pilot is no longer counted inside `alloc_s` (it is reported only as `pilot_s`).
  B. timing: `alloc_s` is the time to COMPUTE an allocation once and no longer depends on how many replicates share a cache;
     reusable allocations (coef, ctrl_*) are stored as ONE vector per trained model and budget and reported as a one-time cost,
     query-specific allocations (pilot_*) as microseconds per query.
  C. diagnostics: `rho_within` = mean WITHIN-model variance-weighted ratio; `rho_pooled` = pooled ratio of mean continuous
     variances across all fitted models (same convention as the finite-budget V_ratio table, which stays the headline statistic).
  D. allocator: the 1e-14 perturbation is removed.  Zero coefficients are handled explicitly (they receive exactly one shot while any
     positive coefficient exists; an all-zero row is split evenly).  The allocator is exact for the stated objective up to
     floating-point arithmetic; this is verified (not proved) at start-up against a heap-greedy reference and brute force.
  E. data audit: source path, SHA-256, dimensions, class counts, dropped rows, column order, zero/NaN counts, library versions and
     settings are printed and saved to v6_1_dataset_audit.json; mismatches against the documented dataset sizes are WARNINGS.
     Preprocessing is unchanged (no imputation; Pima zeros are kept as recorded).
  F. sensitivity: a second measurement scheme gives every policy its OWN independent random streams (coupled streams stay the
     main analysis).  Under it pilot_sp1 - ctrl_sp1 is a pure-noise null check (identical allocations).  The size of the coupling
     effect is printed side by side.  NumPy's binomial sampler is one inverse-CDF draw per element for small n*p, which is why shared
     streams couple policies; the independent scheme removes that dependence.
Everything else (policies, budgets, primary outcome, splits, training) is as in v6; with scheme='coupled' v6.1 reproduces v6.

Policies (all exactly B shots per query, asserted): uniform | coef (a=beta^2) | oracle_wf (a=beta^2 p(1-p), exact p;
NON-DEPLOYABLE, not a performance bound) | pilot_spK (pilot of K shots/entry discarded, smoothed weights) |
ctrl_spK (pilot simulated and ignored, coef weights) | oracle_pilot_spK (oracle weights, budget B - K*m).
PRIMARY (unchanged): average_precision, coef - uniform, B_q=1280, train=128 shots/pair, on every workload.
Exploratory: everything else.  Simulation only; exact conditions are oracle references, not free.
"""
import os, sys, time, glob, json, zlib, hashlib, platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

# ------------------------------- settings -------------------------------
SHOTS = [8, 32, 128]          # shots per kernel entry
NSPLITS, NREPS = 20, 3        # data splits x measurement replicates per split
N_QUBITS, FM_REPS, BW = 4, 1, 0.5
N_TRAIN, N_TEST = 96, 2000
S0 = 2                        # initial shots per pair for random / adaptive
ROUNDS = 8

PATH = "/kaggle/input/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020/ai4i2020.csv"
LEAK = ["UDI", "UID", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]


# ------------------------------- data -------------------------------
def standin(n=10000, seed=0):
    """Used ONLY when the real CSV is missing (code testing, never a result)."""
    r = np.random.default_rng(seed)
    t = r.choice(["L", "M", "H"], n, p=[.5, .3, .2])
    air = 300 + 2 * r.standard_normal(n); proc = air + 10 + r.standard_normal(n)
    rpm = 1500 + 180 * r.standard_normal(n); tq = np.clip(40 + 10 * r.standard_normal(n), 3, None)
    wear = r.integers(0, 250, n) + pd.Series(t).map({"L": 2, "M": 3, "H": 5}).values
    pw = tq * rpm * 2 * np.pi / 60
    lim = pd.Series(t).map({"L": 11000, "M": 12000, "H": 13000}).values
    fail = ((proc - air < 8.6) & (rpm < 1380)) | (pw < 3500) | (pw > 9000) | (wear * tq > lim) | (r.random(n) < .001)
    return pd.DataFrame({"Type": t, "Air temperature [K]": air, "Process temperature [K]": proc,
                         "Rotational speed [rpm]": rpm, "Torque [Nm]": tq, "Tool wear [min]": wear,
                         "Machine failure": fail.astype(int)})


def find_ai4i():
    if os.path.exists(PATH):
        return PATH
    hits = glob.glob("/kaggle/input/**/ai4i*.csv", recursive=True) + \
           glob.glob("/kaggle/input/**/*maintenance*.csv", recursive=True)
    return hits[0] if hits else None


def load_ai4i():
    p = find_ai4i()
    df = pd.read_csv(p) if p else standin()
    y = df["Machine failure"].values
    X = df.drop(columns=["Machine failure"] + [c for c in LEAK if c in df.columns])
    X = pd.get_dummies(X, columns=["Type"], dtype=float).values
    return X, y, p is not None


# ------------------------- exact kernel simulation -------------------------
def feature_state(x, reps):
    n = len(x); dim = 2 ** n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    z = 1 - 2 * ((np.arange(dim)[:, None] >> np.arange(n)) & 1)
    phase = z @ x
    for i in range(n):
        for j in range(i + 1, n):
            phase = phase + (np.pi - x[i]) * (np.pi - x[j]) * z[:, i] * z[:, j]
    H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for r in range(reps):
        if r > 0:
            psi = psi.reshape([2] * n)
            for q in range(n):
                psi = np.moveaxis(np.tensordot(H1, psi, axes=([1], [q])), 0, q)
            psi = psi.reshape(-1)
        psi = psi * np.exp(1j * phase)
    return psi


def states(X, reps):
    return np.array([feature_state(x, reps) for x in X])


def exact_kernel(SA, SB):
    return np.abs(SA.conj() @ SB.T) ** 2


# -------------- allocation policies (exact budget, unique pairs) --------------
def measure(p, k, rng):
    return rng.binomial(k, np.clip(p, 0, 1))


def alloc_int(weights, total, rng=None):
    """Integer allocation of `total` proportional to weights (largest-remainder; sums exactly)."""
    w = weights / weights.sum()
    raw = w * total
    base = np.floor(raw).astype(np.int64)
    rem = int(total - base.sum())
    if rem > 0:
        base[np.argsort(-(raw - base), kind="stable")[:rem]] += 1
    return base


def check_budget(P, total, s0):
    if total < max(s0, 1) * P:
        raise ValueError(f"budget {total} below initialisation cost {s0 * P}")


def pol_uniform(p, total, rng):
    P = len(p)
    n = np.full(P, total // P, dtype=np.int64)
    r = total - int(n.sum())
    if r:
        n[rng.choice(P, r, replace=False)] += 1
    return measure(p, n, rng) / n, n


def pol_random(p, total, rng):
    P = len(p); check_budget(P, total, S0)
    n = np.full(P, S0, dtype=np.int64); succ = measure(p, n, rng)
    add = rng.multinomial(total - S0 * P, np.full(P, 1.0 / P))
    succ = succ + measure(p, add, rng)
    n = n + add
    return succ / n, n


def pol_adaptive(p, total, rng):
    """Round-based variance-proportional allocation."""
    P = len(p); check_budget(P, total, S0)
    n = np.full(P, S0, dtype=np.int64); succ = measure(p, n, rng)
    extra = total - S0 * P
    for r in range(ROUNDS):
        a = extra // ROUNDS + (1 if r < extra % ROUNDS else 0)
        if a == 0:
            continue
        ph = succ / n
        var = (ph * (1 - ph) + 1.0 / (n + 2)) / n
        add = alloc_int(var, a)
        succ = succ + measure(p, add, rng)
        n = n + add
    return succ / n, n


POLICIES = {"uniform": pol_uniform, "random": pol_random, "adaptive": pol_adaptive}


# ------------------------------- helpers -------------------------------
def build_K(phat, iu, n):
    K = np.eye(n)
    K[iu] = phat
    K[(iu[1], iu[0])] = phat
    return K


def psd_repair(K):
    w, V = np.linalg.eigh((K + K.T) / 2)
    return (V * np.clip(w, 0, None)) @ V.T


def score_row(yte, s, pred):
    return dict(auc=roc_auc_score(yte, s), average_precision=average_precision_score(yte, s),
                bal_acc=balanced_accuracy_score(yte, pred))


def boot_ci(d, B=10000, seed=0):
    r = np.random.default_rng(seed)
    m = r.choice(d, size=(B, len(d)), replace=True).mean(axis=1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)




# ------------------------------- v6 settings -------------------------------
import heapq, itertools
TRAIN_SHOTS_PER_PAIR = 128
TRAIN_DRAWS = 2
BUDGETS = [640, 1280, 2560]
PILOT_SP = [1, 2, 4]
NSPLITS, NREPS = 30, 3
NREPS_IND = 4                       # measurement replicates in the independent-stream sensitivity scheme
RUN_SENSITIVITY = True
N_TEST_BY = {"ai4i": 2000, "breast": 400, "pima": 300}
PRIMARY = dict(metric="average_precision", policy="coef", budget=1280)
KAGGLE_BC = "/kaggle/input/datasets/uciml/breast-cancer-wisconsin-data/data.csv"


def load_breast():
    if os.path.exists(KAGGLE_BC):
        df = pd.read_csv(KAGGLE_BC)
        y = (df["diagnosis"] == "M").astype(int).values
        X = df.drop(columns=["id", "diagnosis"], errors="ignore")
        return X.loc[:, ~X.columns.str.startswith("Unnamed")].values, y, "kaggle"
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    return X, 1 - y, "sklearn copy"


def find_pima():
    pats = ["/kaggle/input/**/*pima*.csv", "/kaggle/input/**/diabetes.csv", "/kaggle/input/**/*diabetes*.csv"]
    hits = [h for p in pats for h in glob.glob(p, recursive=True)]
    return hits[0] if hits else None


PIMA_COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]


def _read_pima_raw(p):
    """Returns (X_df, y_series, ok_mask, layout, raw_rows) or None.  Handles a file with an 'Outcome' header and the classic
    UCI layout (no header, 8 features in the documented order, label last)."""
    df = pd.read_csv(p)
    if "Outcome" in df.columns:
        layout = "header present (Outcome column)"
        X, y = df.drop(columns=["Outcome"]), df["Outcome"]; raw_rows = len(df)
    else:
        df = pd.read_csv(p, header=None)
        if df.shape[1] != 9:
            return None
        layout = "no header; classic UCI column order ASSUMED"
        X, y = df.iloc[:, :8], df.iloc[:, 8]; raw_rows = len(df)
    X = X.apply(pd.to_numeric, errors="coerce"); y = pd.to_numeric(y, errors="coerce")
    ok = X.notna().all(axis=1) & y.isin([0, 1])
    return X, y, ok, layout, raw_rows


def load_pima():
    p = find_pima()
    if p is None:
        return None
    r = _read_pima_raw(p)
    if r is None:
        return None
    X, y, ok, _, _ = r
    if ok.sum() < 500:
        return None
    return X[ok].values.astype(float), y[ok].values.astype(int), p


def load_workload(name):
    if name == "ai4i":
        X, y, real = load_ai4i(); return X, y, ("kaggle" if real else "STAND-IN")
    if name == "breast":
        return load_breast()
    return load_pima()


# ------------------------------- exact allocator -------------------------------
def _n_at_threshold(A, tau):
    """Shots per entry when taking every marginal shot with gain a/(k(k+1)) >= tau (n>=1)."""
    x = A / tau[:, None]
    n = np.floor((1.0 + np.sqrt(1.0 + 4.0 * x)) / 2.0)
    n = np.maximum(n, 1.0)
    n = np.where(n * (n - 1.0) > x, n - 1.0, n)               # float guards at integer boundaries
    n = np.where((n + 1.0) * n <= x, n + 1.0, n)
    return np.maximum(n, 1.0)


def alloc_optimal(A, B):
    """EXACT (up to floating-point arithmetic) row-wise integer minimiser of sum_j A_j/n_j  s.t. sum_j n_j = B, n_j >= 1.
    The objective is separable and convex, so the greedy order by marginal gain A_j/(k(k+1)) is optimal.  A threshold bisection
    yields a valid prefix of that order, then single shots are added by largest marginal gain.  No perturbation is added:
    zero coefficients receive exactly one shot while any positive coefficient exists; a row of all zeros is split evenly."""
    A = np.asarray(A, dtype=float)
    Q, m = A.shape
    assert B >= m and (A >= 0).all() and np.isfinite(A).all()
    amax = A.max(1)
    allzero = amax <= 0
    S = np.where(allzero[:, None], 1.0, A / np.where(allzero, 1.0, amax)[:, None])   # argmin is scale invariant
    lo = np.full(Q, 0.5 / (B * (B + 1.0)))                     # S(lo) > B
    hi = np.full(Q, 0.5 * (1 + 1e-9))                          # S(hi) = m <= B
    for _ in range(30):                                        # bisection in log space on the marginal-gain threshold; correctness does not depend on the count
        mid = np.sqrt(lo * hi)
        over = _n_at_threshold(S, mid).sum(1) > B
        lo = np.where(over, mid, lo); hi = np.where(over, hi, mid)
    n = _n_at_threshold(S, hi).astype(np.int64)                # all shots with gain >= hi: a valid greedy prefix, sum <= B
    rem = (B - n.sum(1)).astype(np.int64)
    assert (rem >= 0).all()
    while (rem > 0).any():                                     # finish exactly: one shot at a time, largest marginal gain
        gain = S / (n * (n + 1.0))
        j = gain.argmax(1)
        rows = np.flatnonzero(rem > 0)
        n[rows, j[rows]] += 1
        rem = (B - n.sum(1)).astype(np.int64)
    assert (n.sum(1) == B).all() and (n >= 1).all()
    return n


def alloc_exact_heap(a, B):
    a = np.asarray(a, dtype=float); m = len(a)
    n = np.ones(m, dtype=np.int64); h = [(-a[j] / 2.0, j) for j in range(m)]; heapq.heapify(h)
    for _ in range(B - m):
        _, j = heapq.heappop(h); n[j] += 1; heapq.heappush(h, (-a[j] / (n[j] * (n[j] + 1.0)), j))
    return n


def alloc_uniform(Q, m, B, rng):
    n = np.full((Q, m), B // m, dtype=np.int64)
    r = B - m * (B // m)
    if r:
        ranks = np.argsort(np.argsort(rng.random((Q, m)), axis=1), axis=1)
        n = n + (ranks < r)
    assert (n.sum(1) == B).all()
    return n


def cond_var(beta, P, n):
    return (beta[None, :] ** 2 * P * (1 - P) / n).sum(1)


# ------------------------------- inference policies -------------------------------
def streams(split, draw, B, rep, tag=None):
    """tag=None: COUPLED scheme (every policy gets the same alloc/pilot/final stream seeds, main analysis).
    tag=<policy>: INDEPENDENT scheme (each policy gets its own streams; sensitivity analysis)."""
    extra = [] if tag is None else [zlib.crc32(tag.encode())]
    return {k: np.random.default_rng([split, draw, B, rep, i] + extra) for i, k in enumerate(["alloc", "pilot", "final"], 1)}


def is_reusable(policy):
    """Allocation depends only on the fitted model and budget: one vector per trained model & budget, computed once."""
    return policy == "coef" or policy.startswith("ctrl_sp")


def infer(policy, P, beta, bias, B, rngs, cache=None):
    """Returns dict(scores, V, shots, n, f, pilot_s, alloc_s, final_s, agg_s, cache_hit, batch_q).
    alloc_s = time to COMPUTE the allocation once (pilot sampling excluded).  On a cache hit the stored computation time is
    reported, so alloc_s does not depend on how many replicates share the cache; cache_hit says whether it was recomputed."""
    Q, m = P.shape
    sp = int(policy.split("sp")[1]) if "_sp" in policy else 0
    f = sp * m / B if sp else 0.0
    B2 = B - sp * m
    if sp and B2 < m:
        raise ValueError("pilot infeasible at this budget")
    out = dict(pilot_s=0.0, cache_hit=0, batch_q=Q)

    def cached(key, fn):
        if cache is not None and key in cache:
            n_, t_ = cache[key]
            return n_, t_, 1
        t0_ = time.perf_counter(); n_ = fn(); t_ = time.perf_counter() - t0_
        if cache is not None:
            cache[key] = (n_, t_)
        return n_, t_, 0

    if policy == "uniform":
        t0 = time.perf_counter(); n = alloc_uniform(Q, m, B, rngs["alloc"]); t_alloc = time.perf_counter() - t0
    elif is_reusable(policy):
        n, t_alloc, out["cache_hit"] = cached((policy, B), lambda: np.broadcast_to(
            alloc_optimal((beta ** 2)[None, :], B2 if sp else B)[0], (Q, m)))
    elif policy == "oracle_wf" or policy.startswith("oracle_pilot_sp"):
        n, t_alloc, out["cache_hit"] = cached((policy, B), lambda: alloc_optimal(
            beta[None, :] ** 2 * P * (1 - P), B2 if sp else B))
    elif policy.startswith("pilot_sp"):
        tp = time.perf_counter(); k1 = rngs["pilot"].binomial(sp, P); out["pilot_s"] = time.perf_counter() - tp
        t0 = time.perf_counter()
        pt = (k1 + 1.0) / (sp + 2.0)
        n = alloc_optimal(beta[None, :] ** 2 * pt * (1 - pt), B2); t_alloc = time.perf_counter() - t0
    else:
        raise ValueError(policy)
    if policy.startswith("ctrl_sp"):                                    # simulate the ignored pilot for cost parity (NOT in alloc_s)
        tp = time.perf_counter(); _ = rngs["pilot"].binomial(sp, P); out["pilot_s"] = time.perf_counter() - tp
    out["alloc_s"] = t_alloc
    t0 = time.perf_counter(); est = rngs["final"].binomial(n, P) / n; out["final_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); scores = est @ beta + bias; out["agg_s"] = time.perf_counter() - t0
    shots = int(n[0].sum()) + sp * m
    assert (n.sum(1) + sp * m == B).all() and shots == B, (policy, shots, B)
    out.update(scores=scores, V=cond_var(beta, P, n), shots=shots, f=f, n=n)
    return out


POLICIES = (["uniform", "coef", "oracle_wf"] + [f"pilot_sp{k}" for k in PILOT_SP] + [f"ctrl_sp{k}" for k in PILOT_SP]
            + [f"oracle_pilot_sp{k}" for k in PILOT_SP])
IND_POLICIES = ["uniform", "coef", "oracle_wf"] + [f"pilot_sp{k}" for k in PILOT_SP] + [f"ctrl_sp{k}" for k in PILOT_SP]


def selftest():
    rng = np.random.default_rng(123)
    # 1) reviewer's counterexample and heavy-tailed instances vs heap greedy (objective compared; RAW coefficients, no perturbation)
    A0 = np.array([0.0047748079, 310.6400190063, 9.2526800315, 1.7596238068, 1.6077253721])
    n = alloc_optimal(A0[None, :], 29)[0]
    assert np.isclose((A0 / n).sum(), (A0 / alloc_exact_heap(A0, 29)).sum(), rtol=1e-12), ("counterexample", n)
    bad = 0
    for _ in range(4000):
        m = int(rng.integers(2, 45)); a = rng.lognormal(0, rng.uniform(0.2, 3.0), m); B = int(rng.integers(m, 25 * m))
        a[rng.random(m) < 0.1] = 0.0                                   # explicit zero coefficients
        if a.sum() == 0:
            continue
        n = alloc_optimal(a[None, :], B)[0]; g = alloc_exact_heap(a, B)
        ok = np.isclose((a / n).sum(), (a / g).sum(), rtol=1e-10, atol=1e-15) and (n[a == 0] == 1).all() and n.sum() == B and (n >= 1).all()
        bad += not ok
    assert bad == 0, f"allocator failed on {bad}/4000 random instances (incl. zero coefficients)"
    # 2) all-zero rows are split evenly; multi-row call mixes zero / normal rows correctly
    Az = np.zeros((3, 7)); Az[1] = rng.lognormal(0, 1, 7)
    nz = alloc_optimal(Az, 50)
    assert (nz.sum(1) == 50).all() and nz[0].max() - nz[0].min() <= 1 and nz[2].max() - nz[2].min() <= 1
    assert np.isclose((Az[1] / nz[1]).sum(), (Az[1] / alloc_exact_heap(Az[1], 50)).sum(), rtol=1e-12)
    # 3) brute-force enumeration on tiny problems (incl. a zero coefficient)
    for _ in range(300):
        m = int(rng.integers(2, 5)); B = int(rng.integers(m, m + 8)); a = rng.lognormal(0, 1.5, m)
        if rng.random() < 0.3:
            a[0] = 0.0
        best = min((np.sum(a / np.array(c)), c) for c in itertools.product(range(1, B + 1), repeat=m) if sum(c) == B)
        n = alloc_optimal(a[None, :], B)[0]
        assert np.isclose((a / n).sum(), best[0], rtol=1e-12, atol=1e-15), (a, B, n, best)
    # 4) pilot_sp1 == ctrl_sp1 exactly under COUPLED streams (a one-shot smoothed pilot cannot distinguish entries)
    Q, m = 60, 20; P = rng.uniform(0.02, 0.98, (Q, m)); beta = rng.normal(size=m) * rng.lognormal(0, 1, m)
    for B in [160, 400]:
        r1 = infer("pilot_sp1", P, beta, 0.1, B, streams(1, 0, B, 0)); r2 = infer("ctrl_sp1", P, beta, 0.1, B, streams(1, 0, B, 0))
        assert (r1["n"] == r2["n"]).all() and np.allclose(r1["scores"], r2["scores"]), "pilot_sp1 != ctrl_sp1"
    # 5) independent scheme: same allocations, DIFFERENT noise (so pilot_sp1 - ctrl_sp1 is a pure-noise null check)
    r1 = infer("pilot_sp1", P, beta, 0.1, 400, streams(1, 0, 400, 0, "pilot_sp1")); r2 = infer("ctrl_sp1", P, beta, 0.1, 400, streams(1, 0, 400, 0, "ctrl_sp1"))
    assert (r1["n"] == r2["n"]).all() and not np.allclose(r1["scores"], r2["scores"])
    # 6) budgets exact for every policy
    for pol in POLICIES:
        try:
            infer(pol, P, beta, 0.1, 400, streams(1, 0, 400, 0))
        except ValueError:
            pass
    # 7) timing accounting: cache hit reports the stored computation time; control alloc_s excludes pilot sampling
    cache = {}
    a1 = infer("coef", P, beta, 0.1, 400, streams(1, 0, 400, 0), cache); a2 = infer("coef", P, beta, 0.1, 400, streams(1, 0, 400, 1), cache)
    assert a1["cache_hit"] == 0 and a2["cache_hit"] == 1 and a1["alloc_s"] == a2["alloc_s"]
    c1 = infer("ctrl_sp2", P, beta, 0.1, 400, streams(1, 0, 400, 0), cache)
    assert c1["pilot_s"] > 0 and c1["alloc_s"] >= 0
    print("selftest passed: exact allocator without perturbation (4000 heavy-tailed incl. zero coefficients + 300 brute-force + "
          "counterexample + all-zero rows), pilot_sp1 == ctrl_sp1 (coupled) and != (independent), budgets exact, timing accounting")


# ------------------------------- dataset audit -------------------------------
EXPECTED = {"ai4i": dict(rows=10000, positives=339), "breast": dict(rows=569, positives=212, negatives=357),
            "pima": dict(rows=768, positives=268, negatives=500)}


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _check(name, a):
    exp = EXPECTED[name]; msgs = []
    for k, v in exp.items():
        if a.get(k) != v:
            msgs.append(f"{k}={a.get(k)} (documented {v})")
    a["matches_documented"] = not msgs
    a["warnings"] = msgs
    return a


def audit_datasets():
    out = dict(env=dict(python=sys.version.split()[0], platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__,
                        sklearn=sklearn.__version__),
               config=dict(N_QUBITS=N_QUBITS, FM_REPS=FM_REPS, BW=BW, N_TRAIN=N_TRAIN, N_TEST_BY=N_TEST_BY, BUDGETS=BUDGETS,
                           PILOT_SP=PILOT_SP, NSPLITS=NSPLITS, NREPS=NREPS, NREPS_IND=NREPS_IND, TRAIN_DRAWS=TRAIN_DRAWS,
                           TRAIN_SHOTS_PER_PAIR=TRAIN_SHOTS_PER_PAIR, PRIMARY=PRIMARY))
    # AI4I
    p = find_ai4i()
    if p:
        df = pd.read_csv(p); y = df["Machine failure"]
        feats = pd.get_dummies(df.drop(columns=["Machine failure"] + [c for c in LEAK if c in df.columns]), columns=["Type"], dtype=float).columns.tolist()
        out["ai4i"] = _check("ai4i", dict(source=p, sha256=_sha(p), rows=len(df), columns=list(df.columns), dropped_columns=[c for c in LEAK if c in df.columns],
                                          features_used=feats, positives=int((y == 1).sum()), negatives=int((y == 0).sum()), nan_cells=int(df.isna().sum().sum())))
    else:
        out["ai4i"] = dict(source="STAND-IN (generated; not the real dataset)")
    # Breast cancer
    if os.path.exists(KAGGLE_BC):
        df = pd.read_csv(KAGGLE_BC); y = (df["diagnosis"] == "M").astype(int)
        feats = [c for c in df.columns if c not in ("id", "diagnosis") and not str(c).startswith("Unnamed")]
        out["breast"] = _check("breast", dict(source=KAGGLE_BC, sha256=_sha(KAGGLE_BC), rows=len(df), features_used=feats, n_features=len(feats),
                                              positives=int(y.sum()), negatives=int((1 - y).sum()), nan_cells=int(df[feats].isna().sum().sum())))
    else:
        Xb, yb, _ = load_breast()
        out["breast"] = _check("breast", dict(source="scikit-learn bundled copy (same WDBC data)", rows=len(yb), n_features=Xb.shape[1],
                                              positives=int(yb.sum()), negatives=int((1 - yb).sum())))
    # Pima
    p = find_pima(); r = _read_pima_raw(p) if p else None
    if r is not None:
        X, y, ok, layout, raw = r
        Xo, yo = X[ok], y[ok]
        zeros = {c: int((Xo.iloc[:, i] == 0).sum()) for i, c in enumerate(PIMA_COLS)}
        decl = [str(c) for c in pd.read_csv(p, nrows=0).columns] if "header present" in layout else PIMA_COLS
        out["pima"] = _check("pima", dict(source=p, sha256=_sha(p), layout=layout, raw_rows=raw, rows=int(ok.sum()), dropped_rows=int(raw - ok.sum()),
                                          columns_declared_or_assumed=decl, positives=int((yo == 1).sum()), negatives=int((yo == 0).sum()),
                                          zero_counts_per_feature=zeros,
                                          zero_treatment="NONE: zeros kept exactly as recorded (no imputation, no row removal); in this dataset zeros in "
                                                         "Glucose/BloodPressure/SkinThickness/Insulin/BMI commonly encode missing measurements"))
    else:
        out["pima"] = dict(source="NOT FOUND")
    return out


def print_audit(a):
    print("\n=== RUN AUDIT ===")
    print("env:", a["env"])
    for name in ["ai4i", "breast", "pima"]:
        d = a[name]
        if "sha256" not in d and "rows" not in d:
            print(f"[{name}] {d['source']}"); continue
        print(f"[{name}] source={d['source']}")
        if "sha256" in d:
            print(f"        sha256={d['sha256']}")
        print(f"        rows={d.get('rows')} positives={d.get('positives')} negatives={d.get('negatives')}"
              + (f" dropped_rows={d['dropped_rows']}" if "dropped_rows" in d else "") + (f" layout={d['layout']}" if "layout" in d else ""))
        if "zero_counts_per_feature" in d:
            print("        zeros per feature (kept as recorded):", d["zero_counts_per_feature"])
        if "matches_documented" in d:
            print("        documented-size check:", "matches" if d["matches_documented"] else "WARNING: " + "; ".join(d["warnings"]))


# ------------------------------- one split -------------------------------
def run_split(workload, split_seed):
    X, y, _ = load_workload(workload)
    n_test = N_TEST_BY[workload]
    Xa, Xte, ya, yte = train_test_split(X, y, test_size=n_test, stratify=y, random_state=split_seed)
    rs = np.random.default_rng(split_seed)
    pos, neg = np.flatnonzero(ya == 1), np.flatnonzero(ya == 0)
    idx = np.r_[rs.choice(pos, N_TRAIN // 2, replace=False), rs.choice(neg, N_TRAIN // 2, replace=False)]
    Xtr, ytr = Xa[idx], ya[idx]
    sc = StandardScaler().fit(Xtr); pca = PCA(N_QUBITS, random_state=split_seed).fit(sc.transform(Xtr))
    mm = MinMaxScaler((0, BW)).fit(pca.transform(sc.transform(Xtr)))
    prep = lambda A: np.clip(mm.transform(pca.transform(sc.transform(A))), 0, BW)
    Ztr, Zte = prep(Xtr), prep(Xte)
    Str, Ste = states(Ztr, FM_REPS), states(Zte, FM_REPS)
    Kt, Ke = exact_kernel(Str, Str), exact_kernel(Ste, Str)
    n = len(Ztr); iu = np.triu_indices(n, 1); p_true = Kt[iu]; P_pairs = len(p_true)
    rows = []
    schemes = [("coupled", POLICIES, NREPS)] + ([("independent", IND_POLICIES, NREPS_IND)] if RUN_SENSITIVITY else [])
    for draw in range(TRAIN_DRAWS):
        prng = np.random.default_rng([split_seed, TRAIN_SHOTS_PER_PAIR, draw, 99])
        p_hat, n_sh = pol_uniform(p_true, TRAIN_SHOTS_PER_PAIR * P_pairs, prng)
        assert int(n_sh.sum()) == TRAIN_SHOTS_PER_PAIR * P_pairs
        Kp = psd_repair(build_K(p_hat, iu, n))
        clf = SVC(kernel="precomputed", class_weight="balanced").fit(Kp, ytr)
        sv = clf.support_; beta = clf.dual_coef_.ravel(); bias = float(clf.intercept_[0]); m = len(sv)
        P = np.clip(Ke[:, sv], 0, 1)
        f_exact = P @ beta + bias
        assert np.allclose(f_exact, clf.decision_function(Ke), atol=1e-8), "manual SV score != decision_function"
        pred_exact = (f_exact > 0).astype(int)
        A_true = beta[None, :] ** 2 * P * (1 - P)
        Vopt = np.sqrt(A_true).sum(1) ** 2                              # continuous optimum variance x B, per query
        Vuni = m * A_true.sum(1)                                        # continuous uniform variance x B, per query
        rho_q = Vopt / (Vuni + 1e-300)
        sq = np.sqrt(P * (1 - P))
        diag = dict(m_sv=m, rho_mean=float(rho_q.mean()), rho_within=float(Vopt.sum() / Vuni.sum()),
                    cont_opt_mean=float(Vopt.mean()), cont_uni_mean=float(Vuni.mean()),
                    cv_beta=float(np.abs(beta).std() / np.abs(beta).mean()),
                    cv_sqrtpq=float((sq.std(1) / (sq.mean(1) + 1e-12)).mean()))
        rows.append(dict(workload=workload, scheme="coupled", split=split_seed, draw=draw, rep=0, budget=0, policy="exact_inference",
                         label_agree=1.0, score_mse=0.0, cond_var=0.0, pilot_frac=0.0, frac_rho_lt=np.nan,
                         pilot_s=0.0, alloc_s=0.0, final_s=0.0, agg_s=0.0, cache_hit=np.nan, batch_q=float(P.shape[0]),
                         **diag, **score_row(yte, f_exact, pred_exact)))
        for B in BUDGETS:
            cache = {}                                                  # allocations depend only on (policy, budget) for a fixed model
            for scheme, pols, nreps in schemes:
                for rep in range(nreps):
                    for pol in pols:
                        try:
                            rngs = streams(split_seed, draw, B, rep, tag=None if scheme == "coupled" else pol)
                            r = infer(pol, P, beta, bias, B, rngs, cache)
                        except ValueError:
                            continue
                        pred = (r["scores"] > 0).astype(int)
                        sp = int(pol.split("sp")[1]) if "_sp" in pol else 0
                        rows.append(dict(workload=workload, scheme=scheme, split=split_seed, draw=draw, rep=rep, budget=B, policy=pol,
                                         label_agree=float((pred == pred_exact).mean()),
                                         score_mse=float(np.mean((r["scores"] - f_exact) ** 2)), cond_var=float(r["V"].mean()),
                                         pilot_frac=r["f"], frac_rho_lt=float((rho_q < 1 - r["f"]).mean()) if sp else np.nan,
                                         pilot_s=r["pilot_s"], alloc_s=r["alloc_s"], final_s=r["final_s"], agg_s=r["agg_s"],
                                         cache_hit=r["cache_hit"], batch_q=float(r["batch_q"]), **diag, **score_row(yte, r["scores"], pred)))
    return pd.DataFrame(rows)


# ------------------------------- reporting -------------------------------
def report(res, workload, full):
    r = res[(res.workload == workload) & (res.scheme == "coupled")]
    keys = ["split", "budget", "policy"]
    num = ["auc", "average_precision", "bal_acc", "label_agree", "score_mse", "cond_var", "pilot_frac", "frac_rho_lt", "m_sv",
           "rho_mean", "rho_within", "cont_opt_mean", "cont_uni_mean", "cv_beta", "cv_sqrtpq", "pilot_s", "alloc_s", "final_s",
           "agg_s", "batch_q"]
    sp = r.groupby(keys)[num].mean().reset_index()
    ex = sp[sp.policy == "exact_inference"]; q = sp[sp.policy != "exact_inference"]
    print("\n" + "=" * 100 + f"\nWORKLOAD: {workload}   (splits={r.split.nunique()}; main analysis = coupled streams)\n" + "=" * 100)

    d = ex[["m_sv", "rho_mean", "rho_within", "cv_beta", "cv_sqrtpq", "auc", "average_precision", "bal_acc"]].mean()
    pooled = ex.cont_opt_mean.mean() / ex.cont_uni_mean.mean()
    print("\n--- Trained-model diagnostics (mean over splits) ---"); print(d.round(4).to_string())
    print(f"Idealised (continuous) oracle/uniform variance ratio: mean query-level rho = {d.rho_mean:.4f}; mean WITHIN-model "
          f"variance-weighted ratio = {d.rho_within:.4f}; POOLED ratio of mean variances across all fitted models = {pooled:.4f}. "
          f"Finite-budget V_ratio below is the headline headroom statistic.")

    print("\n--- Variance headroom (ratio of mean conditional variances vs uniform; lower is better) ---")
    base = q[q.policy == "uniform"].groupby("budget").cond_var.mean()
    tab = []
    for (B, pol), g in q.groupby(["budget", "policy"]):
        tab.append(dict(budget=B, policy=pol, V_ratio=g.cond_var.mean() / base[B], pilot_frac=g.pilot_frac.mean(),
                        frac_queries_rho_lt_1mf=g.frac_rho_lt.mean()))
    print(pd.DataFrame(tab).round(4).to_string(index=False))

    print("\n--- Recovery: share of the learnable variance gain the real pilot achieves ---")
    print("   recovery = (V_ctrl - V_pilot) / (V_ctrl - V_oracle_pilot); 0 = learned nothing, 1 = as good as oracle weights at the same overhead")
    V = lambda p, B: q[(q.policy == p) & (q.budget == B)].cond_var.mean()
    for B in BUDGETS:
        for k in PILOT_SP:
            if q[(q.policy == f"pilot_sp{k}") & (q.budget == B)].empty:
                continue
            den = V(f"ctrl_sp{k}", B) - V(f"oracle_pilot_sp{k}", B)
            rec = (V(f"ctrl_sp{k}", B) - V(f"pilot_sp{k}", B)) / den if abs(den) > 1e-15 else np.nan
            print(f"B={B:>4} sp={k}: V_ctrl={V(f'ctrl_sp{k}', B):.5f}  V_pilot={V(f'pilot_sp{k}', B):.5f}  V_oracle_pilot={V(f'oracle_pilot_sp{k}', B):.5f}  recovery={rec:+.3f}")

    print("\n--- Predictive metrics (mean over splits); calibration: sqrt(mean realised MSE) vs sqrt(mean conditional V) ---")
    t = q.groupby(["budget", "policy"])[["auc", "average_precision", "label_agree"]].mean()
    t["emp_rmse"] = np.sqrt(q.groupby(["budget", "policy"]).score_mse.mean())
    t["theory_rmse"] = np.sqrt(q.groupby(["budget", "policy"]).cond_var.mean())
    print(t.round(4).to_string())

    def paired(a_pol, b_pol, B, m):
        a = q[(q.budget == B) & (q.policy == a_pol)].sort_values("split")[m].values
        b = q[(q.budget == B) & (q.policy == b_pol)].sort_values("split")[m].values
        return boot_ci(a - b)

    mu, lo, hi = paired(PRIMARY["policy"], "uniform", PRIMARY["budget"], PRIMARY["metric"])
    print(f"\n--- PRIMARY ({workload}): {PRIMARY['metric']}, {PRIMARY['policy']} - uniform, B_q={PRIMARY['budget']} ---")
    print(f"{mu:+.4f}  95% bootstrap CI over splits [{lo:+.4f}, {hi:+.4f}]")

    print("\n--- Exploratory paired differences (mean [95% CI over splits]; AP | AUC) ---")
    pairs = [(p, "uniform") for p in POLICIES if p != "uniform" and not p.startswith("oracle_pilot")] + \
            [(f"pilot_sp{k}", f"ctrl_sp{k}") for k in PILOT_SP] + [(f"ctrl_sp{k}", "coef") for k in PILOT_SP]
    for B in BUDGETS:
        for a, b in pairs:
            if q[(q.budget == B) & (q.policy == a)].empty or q[(q.budget == B) & (q.policy == b)].empty:
                continue
            x = paired(a, b, B, "average_precision"); z = paired(a, b, B, "auc")
            print(f"B={B:>4} {a:>10} - {b:<9} | AP {x[0]:+.4f} [{x[1]:+.4f},{x[2]:+.4f}] | AUC {z[0]:+.4f} [{z[1]:+.4f},{z[2]:+.4f}]")
    if full:
        print("\n--- Classical simulation cost per query batch (seconds; pilot sampling is NOT part of allocation) ---")
        print("   alloc_s = time to COMPUTE the allocation once (cache-independent). 'reusable' allocations are ONE vector per trained model and budget:")
        print("   a one-time cost amortised over all future queries.  Query-specific allocations cost alloc_us_per_query each time.  oracle_* are non-deployable.")
        g = q.groupby("policy")[["pilot_s", "alloc_s", "final_s", "agg_s", "batch_q"]].mean()
        g["reusable"] = [is_reusable(p) for p in g.index]
        g["alloc_us_per_query"] = np.where(g.reusable | g.index.str.startswith("oracle"), np.nan, g.alloc_s / g.batch_q * 1e6)
        g["alloc_one_time_s"] = np.where(g.reusable, g.alloc_s, np.nan)
        print(g.drop(columns=["batch_q"]).round(5).to_string())


def report_sensitivity(res, workload):
    r = res[(res.workload == workload) & (res.policy != "exact_inference")]
    if r.scheme.nunique() < 2:
        return
    sp = r.groupby(["scheme", "split", "budget", "policy"])[["average_precision", "auc", "cond_var"]].mean().reset_index()
    print("\n" + "-" * 100 + f"\nSENSITIVITY ({workload}): coupled final-stage streams (main analysis) vs INDEPENDENT streams per policy "
          f"({NREPS_IND} measurement replicates)\n" + "-" * 100)

    def pdiff(scheme, a, b, B, m):
        s = sp[(sp.scheme == scheme) & (sp.budget == B)]
        x = s[s.policy == a].sort_values("split")[m].values; y = s[s.policy == b].sort_values("split")[m].values
        return boot_ci(x - y)

    contrasts = [("coef", "uniform"), ("oracle_wf", "uniform"), ("pilot_sp2", "ctrl_sp2"), ("pilot_sp4", "ctrl_sp4"),
                 ("ctrl_sp2", "coef"), ("ctrl_sp4", "coef"), ("pilot_sp1", "ctrl_sp1")]
    print("(pilot_sp1 - ctrl_sp1 has IDENTICAL allocations: exactly 0 when coupled; under independent streams it is a pure-noise null check)")
    widths = {}
    for B in BUDGETS:
        for a, b in contrasts:
            have = all(not sp[(sp.scheme == s) & (sp.budget == B) & (sp.policy == p)].empty for s in ["coupled", "independent"] for p in (a, b))
            if not have:
                continue
            line = f"B={B:>4} {a:>9} - {b:<9}"
            for scheme in ["coupled", "independent"]:
                ap = pdiff(scheme, a, b, B, "average_precision"); au = pdiff(scheme, a, b, B, "auc")
                line += f" | {scheme[:5]} AP {ap[0]:+.4f} [{ap[1]:+.4f},{ap[2]:+.4f}] AUC {au[0]:+.4f} [{au[1]:+.4f},{au[2]:+.4f}]"
                widths[(scheme, B, a, b)] = ap[2] - ap[1]
            print(line)
    B0 = PRIMARY["budget"]; a0, b0 = PRIMARY["policy"], "uniform"
    wc, wi = widths.get(("coupled", B0, a0, b0)), widths.get(("independent", B0, a0, b0))
    if wc and wi:
        print(f"\nPRIMARY contrast CI width: coupled {wc:.4f} vs independent {wi:.4f} (ratio {wi / wc:.2f}x) -- coupling narrows intervals; "
              f"report both and judge conclusions on whether the mean effect and its sign are stable.")

    print("\n--- Variance headroom (ratio of mean conditional variances vs uniform) under both schemes ---")
    base = {s: sp[(sp.scheme == s) & (sp.policy == "uniform")].groupby("budget").cond_var.mean() for s in ["coupled", "independent"]}
    for B in BUDGETS:
        parts = []
        for pol in ["coef", "oracle_wf", "pilot_sp2", "pilot_sp4", "ctrl_sp2", "ctrl_sp4"]:
            v = [sp[(sp.scheme == s) & (sp.budget == B) & (sp.policy == pol)].cond_var.mean() / base[s][B] for s in ["coupled", "independent"]]
            if not np.isnan(v).any():
                parts.append(f"{pol} {v[0]:.4f}/{v[1]:.4f}")
        print(f"B={B:>4} (coupled/independent): " + "  ".join(parts))


def main(nsplits=None):
    nsplits = NSPLITS if nsplits is None else nsplits
    ok = load_ai4i()[2]
    print("AI4I DATA:", ("REAL from " + str(find_ai4i())) if ok else "STAND-IN (code test only, not a result)")
    if not ok and os.environ.get("ALLOW_STANDIN") != "1":
        raise SystemExit("AI4I CSV not found - research runs fail closed. Attach the dataset via 'Add Input'. "
                         "(Set ALLOW_STANDIN=1 only for code testing.)")
    audit = audit_datasets(); print_audit(audit)
    with open("v6_1_dataset_audit.json", "w") as fh:
        json.dump(audit, fh, indent=2, default=str)
    pima = load_pima()
    print("PIMA (validation workload not used during development):", ("found at " + pima[2]) if pima else
          "NOT attached -> skipped (add a Pima diabetes dataset for the untouched-workload check)")
    selftest()
    workloads = ["ai4i", "breast"] + (["pima"] if pima else [])
    t0 = time.time(); out = []
    for wl in workloads:
        for s_ in range(nsplits):
            out.append(run_split(wl, s_))
            if (s_ + 1) % 10 == 0:
                print(f"{wl}: {s_ + 1}/{nsplits} splits ({time.time() - t0:.0f}s)", flush=True)
    res = pd.concat(out, ignore_index=True)
    res.to_csv("v6_1_results.csv", index=False)
    for wl in workloads:
        report(res, wl, full=(wl == "ai4i"))
        report_sensitivity(res, wl)
    print("\nNotes: intervals describe uncertainty across resampled splits within ONE dataset per workload. 'oracle_wf' and "
          "'oracle_pilot' are non-deployable references, not performance bounds. Only the PRIMARY comparison was fixed in advance. "
          "Simulation under ideal probabilities with binomial sampling: no hardware noise, no compilation, no bias.")
    return res


if __name__ == "__main__":
    main()
