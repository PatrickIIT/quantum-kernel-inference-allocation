"""Figures for the manuscript; every number is transcribed from the v6.1 / robustness run outputs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
W = {"AI4I": "#0072B2", "Breast Cancer": "#D55E00", "Pima": "#009E73"}   # Okabe-Ito; markers differ as well
MK = {"AI4I": "o", "Breast Cancer": "s", "Pima": "^"}

# ---- Figure 1: conditional variance ratio vs uniform at B_q = 1280 (finite-budget means over 30 splits)
pol = ["coef", "oracle_wf", "pilot_sp1", "pilot_sp2", "pilot_sp4", "oracle_pilot_sp2"]
lab = ["coefficient\n(free)", "oracle weights\n(non-deployable)", "pilot, K=1", "pilot, K=2", "pilot, K=4", "oracle weights,\nK=2 budget"]
V = {"AI4I": [.9295, .8705, .9838, 1.0410, 1.1804, .9783],
     "Breast Cancer": [.8606, .8285, .8930, .9274, 1.0038, .8934],
     "Pima": [.9215, .8860, .9769, 1.0384, 1.1877, .9992]}
fig, ax = plt.subplots(figsize=(6.0, 2.9))
y = np.arange(len(pol))[::-1]
for k, (w, v) in enumerate(V.items()):
    ax.scatter(v, y + (k - 1) * 0.18, s=26, color=W[w], marker=MK[w], label=w, zorder=3, edgecolor="white", linewidth=0.4)
ax.axvline(1.0, color="0.35", lw=0.8, ls="--"); ax.text(1.002, y[0] + 0.45, "uniform = 1", fontsize=7.5, color="0.25", va="bottom")
ax.set_yticks(y); ax.set_yticklabels(lab); ax.set_xlabel("Mean conditional score variance relative to uniform (lower is better)")
ax.grid(axis="x", color="0.9", lw=0.5); ax.set_axisbelow(True); ax.legend(frameon=False, loc="lower right", fontsize=7.5)
fig.tight_layout(); fig.savefig("fig_variance.pdf"); plt.close(fig)

# ---- Figure 2: primary comparison, coef - uniform AP at B_q = 1280, coupled vs independent streams
P = {"AI4I": ((.0043, .0031, .0057), (.0033, -.0008, .0073)),
     "Breast Cancer": ((.0040, .0035, .0046), (.0035, .0019, .0050)),
     "Pima": ((.0043, .0032, .0054), (.0044, -.0006, .0095))}
fig, ax = plt.subplots(figsize=(6.0, 2.3))
rows = []
for w, (c, i) in P.items():
    rows += [(f"{w}, coupled", c, w, "filled"), (f"{w}, independent", i, w, "open")]
for r, (name, (m, lo, hi), w, st) in enumerate(rows[::-1]):
    ax.hlines(r, lo, hi, color=W[w], lw=1.4)
    ax.scatter([m], [r], s=30, marker=MK[w], color=W[w] if st == "filled" else "white", edgecolor=W[w], linewidth=1.1, zorder=3)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
ax.axvline(0, color="0.35", lw=0.8, ls="--"); ax.set_xlabel(r"$\Delta$AP (coefficient $-$ uniform), 95% split-bootstrap interval")
ax.grid(axis="x", color="0.9", lw=0.5); ax.set_axisbelow(True); fig.tight_layout(); fig.savefig("fig_primary.pdf"); plt.close(fig)

# ---- Figure 3: robustness grid, coef - uniform AP at B_q = 1280 (12 splits per configuration)
cfg = ["baseline", "bw=0.25", "bw=1.0", "reps=2", "ntrain=48", "ntrain=120", "C=0.1", "C=10", "shots/pair=32", "shots/pair=512", "kbias=0.1"]
R = {"AI4I": [(.0038, .0016, .0061), (.0022, .0001, .0046), (.0031, .0008, .0056), (.0025, .0008, .0043), (.0026, .0014, .0039), (.0054, .0037, .0070),
              (.0004, -.0007, .0017), (.0049, .0036, .0063), (.0045, .0017, .0070), (.0034, .0012, .0061), (.0033, .0010, .0055)],
     "Breast Cancer": [(.0042, .0032, .0053), (.0039, .0031, .0047), (.0098, .0082, .0115), (.0088, .0073, .0104), (.0028, .0021, .0035), (.0053, .0044, .0063),
              (.0001, -.0001, .0002), (.0483, .0407, .0560), (.0046, .0038, .0054), (.0036, .0029, .0042), (.0053, .0042, .0065)],
     "Pima": [(.0035, .0023, .0048), (.0023, .0004, .0038), (.0037, .0019, .0059), (.0029, .0007, .0051), (.0013, -.0004, .0030), (.0045, .0027, .0064),
              (.0003, -.0004, .0009), (.0047, .0013, .0083), (.0027, .0015, .0039), (.0028, .0005, .0050), (.0030, .0004, .0058)]}
fig, axs = plt.subplots(1, 3, figsize=(6.6, 3.1), sharey=True)
for ax, (w, rr) in zip(axs, R.items()):
    for r, (m, lo, hi) in enumerate(rr):
        yy = len(cfg) - 1 - r
        ax.hlines(yy, lo, hi, color=W[w], lw=1.4); ax.scatter([m], [yy], s=22, marker=MK[w], color=W[w], zorder=3)
    ax.axvline(0, color="0.35", lw=0.8, ls="--"); ax.set_title(w, fontsize=8.5); ax.grid(axis="x", color="0.9", lw=0.5); ax.set_axisbelow(True)
    if w == "Breast Cancer":
        ax.set_xlim(-0.001, 0.0125)
        yy = len(cfg) - 1 - cfg.index("C=10")
        ax.annotate("+0.0483 [0.0407, 0.0560]\n(off scale)", xy=(0.0125, yy), xytext=(0.0124, yy), ha="right", va="center", fontsize=6.5, color="0.2")
axs[0].set_yticks(range(len(cfg))); axs[0].set_yticklabels(cfg[::-1])
axs[1].set_xlabel(r"$\Delta$AP (coefficient $-$ uniform), 95% interval; Breast Cancer panel has its own x-range", fontsize=7.5)
fig.tight_layout(); fig.savefig("fig_robust.pdf"); plt.close(fig)
print("ok")
