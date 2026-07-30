"""
run_experiments.py
==================
Full sim-to-real study for TWI target detection.

Arms compared:
  (A) Baseline CNN, real-only          (reproduces the published baseline setting)
  (B) Idealized-synthetic only         (square+Gaussian; no physics)  -> should transfer poorly
  (C) Physics-synthetic only           (no adaptation)
  (D) Physics-synthetic + DANN-MUSIC   (PROPOSED: physics-guided + subspace anchor)
  (E) All-real upper bound (oracle)    (train on all real patches)

Plus ablations for (D): physics-guidance on/off, MUSIC-anchor on/off.

Outputs:
  outputs/metrics.json
  outputs/efficiency_curve.csv
  figures/*.png
"""

import os, sys, json, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_io import load_real_dataset, to_range_image, normalize01
from music import estimate_wall_params, wall_dist_from_estimate
from models import BaselineCNN, DANN_MUSIC
from datasets import (extract_real_patches, make_synth_patches,
                      make_idealized_patches, to_tensors)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"
ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Data two targets")
OUT = "/home/claude/twi_project/outputs"
FIG = "/home/claude/twi_project/figures"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
N_CLASSES = 2   # target-present vs background (binary detection)
PATCH_W = 48


# ---------------------------------------------------------------- metrics
def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


# ---------------------------------------------------------------- training
def train_plain(model, Xt, yt, Mt, epochs=35, lr=1e-3, use_music=False, bs=128):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    n = Xt.shape[0]
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            out = model(Xt[idx], Mt[idx]) if use_music else model(Xt[idx])
            loss = lossf(out, yt[idx])
            loss.backward(); opt.step()
    return model


def train_dann(model, Xs, ys, Ms, Xr, Mr, epochs=45, lr=1e-3,
               use_music=True, adapt=True, bs=128):
    """Domain-adversarial training: source (synthetic, labelled) + target (real, unlabelled)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    n_s = Xs.shape[0]; n_r = Xr.shape[0]
    model.train()
    for ep in range(epochs):
        p = ep / epochs
        lambd = (2.0 / (1.0 + np.exp(-10 * p)) - 1.0) if adapt else 0.0
        perm = torch.randperm(n_s)
        for i in range(0, n_s, bs):
            sidx = perm[i:i+bs]
            opt.zero_grad()
            # source: classification + domain=0
            ys_out, ds_out, _ = model(Xs[sidx], Ms[sidx], lambd)
            loss_cls = lossf(ys_out, ys[sidx])
            dom_src = torch.zeros(sidx.shape[0], dtype=torch.long, device=Xs.device)
            loss_dsrc = lossf(ds_out, dom_src)
            # target: domain=1 (no labels used)
            ridx = torch.randint(0, n_r, (sidx.shape[0],))
            _, dr_out, _ = model(Xr[ridx], Mr[ridx], lambd)
            dom_tgt = torch.ones(ridx.shape[0], dtype=torch.long, device=Xr.device)
            loss_dtgt = lossf(dr_out, dom_tgt)
            loss = loss_cls + (loss_dsrc + loss_dtgt if adapt else 0.0)
            loss.backward(); opt.step()
    return model


@torch.no_grad()
def evaluate(model, Xt, yt, Mt, dann=False):
    model.eval()
    if dann:
        out, _, _ = model(Xt, Mt, 0.0)
    else:
        try:
            out = model(Xt, Mt)
        except TypeError:
            out = model(Xt)
    pred = out.argmax(1).cpu().numpy()
    return metrics(yt.cpu().numpy(), pred), pred


# ---------------------------------------------------------------- main
def main():
    caps = load_real_dataset(ROOT)
    freq = caps[0]["raw"]["freq"]

    # --- real patches per capture (leave-one-capture-out) ---
    real_patches = []
    for c in caps:
        Xp, yp, Mp = extract_real_patches(c, patch_w=PATCH_W, stride=8)
        real_patches.append((Xp, yp, Mp, c["id"]))
        print(f"[real {c['id']}] {Xp.shape[0]} patches, label dist {np.bincount(yp, minlength=2)}")

    # --- wall estimate (physics-guided) from real data ---
    est = estimate_wall_params(caps[0]["raw"]["S21"], freq)
    wall = wall_dist_from_estimate(est)
    print("Physics-guided wall estimate:", est)

    # --- synthetic source sets ---
    N_SYN = 250
    Xphy, yphy, Mphy = make_synth_patches(freq, N_SYN, PATCH_W, seed=10, wall_dist=wall)
    Xphy_ng, yphy_ng, Mphy_ng = make_synth_patches(freq, N_SYN, PATCH_W, seed=11, wall_dist=None)  # no physics-guidance
    Xide, yide, Mide = make_idealized_patches(N_SYN, PATCH_W, seed=12)
    print(f"Synthetic(physics): {Xphy.shape}  label dist {np.bincount(yphy, minlength=2)}")
    print(f"Synthetic(idealized): {Xide.shape} label dist {np.bincount(yide, minlength=2)}")

    results = {}
    FRACS = [0.0, 0.5, 1.0]   # fraction of held-in real patches used for fine-tune

    # leave-one-capture-out: test on capture k, the OTHER capture supplies any
    # real fine-tuning patches.
    def loco(arm_fn):
        accs = {fr: [] for fr in FRACS}
        f1s = {fr: [] for fr in FRACS}
        for k in range(len(caps)):
            Xte, yte, Mte, _ = real_patches[k]
            # other capture's patches as the available real pool
            j = 1 - k
            Xpool, ypool, Mpool, _ = real_patches[j]
            for fr in FRACS:
                m, pred = arm_fn(Xpool, ypool, Mpool, Xte, yte, Mte, fr)
                accs[fr].append(m["accuracy"]); f1s[fr].append(m["f1"])
        return ({fr: float(np.mean(accs[fr])) for fr in FRACS},
                {fr: float(np.mean(f1s[fr])) for fr in FRACS})

    # ---------- Arm A: real-only baseline (train on pool real patches) -------
    def arm_real_only(Xpool, ypool, Mpool, Xte, yte, Mte, fr):
        Xt, yt, Mt = to_tensors(Xpool, ypool, Mpool, DEV)
        model = BaselineCNN(N_CLASSES).to(DEV)
        train_plain(model, Xt, yt, Mt, epochs=30)
        Xe, ye, Me = to_tensors(Xte, yte, Mte, DEV)
        return evaluate(model, Xe, ye, Me)

    # ---------- generic synthetic+finetune arm -------------------------------
    def make_synth_arm(Xs, ys, Ms, proposed=False, adapt=False,
                       use_music=False, physics=True):
        def arm(Xpool, ypool, Mpool, Xte, yte, Mte, fr):
            # ablation: mute the MUSIC anchor by zeroing it everywhere
            Ms_u = Ms if use_music else np.zeros_like(Ms)
            Mpool_u = Mpool if use_music else np.zeros_like(Mpool)
            Mte_u = Mte if use_music else np.zeros_like(Mte)
            Xs_t, ys_t, Ms_t = to_tensors(Xs, ys, Ms_u, DEV)
            Xe, ye, Me = to_tensors(Xte, yte, Mte_u, DEV)
            # optional real fine-tune subset
            n_use = int(round(fr * Xpool.shape[0]))
            if proposed:
                Xr_t, _, Mr_t = to_tensors(Xpool, ypool, Mpool_u, DEV)
                model = DANN_MUSIC(N_CLASSES, music_dim=64).to(DEV)
                train_dann(model, Xs_t, ys_t, Ms_t, Xr_t, Mr_t,
                           epochs=30, use_music=use_music, adapt=adapt)
                if n_use > 0:   # light supervised fine-tune on real
                    idx = np.random.choice(Xpool.shape[0], n_use, replace=False)
                    Xf, yf, Mf = to_tensors(Xpool[idx], ypool[idx], Mpool_u[idx], DEV)
                    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
                    lf = nn.CrossEntropyLoss(); model.train()
                    for _ in range(25):
                        opt.zero_grad(); o,_,_ = model(Xf, Mf, 0.0)
                        lf(o, yf).backward(); opt.step()
                return evaluate(model, Xe, ye, Me, dann=True)
            else:
                model = BaselineCNN(N_CLASSES).to(DEV)
                train_plain(model, Xs_t, ys_t, Ms_t, epochs=30)
                if n_use > 0:
                    idx = np.random.choice(Xpool.shape[0], n_use, replace=False)
                    Xf, yf, Mf = to_tensors(Xpool[idx], ypool[idx], Mpool_u[idx], DEV)
                    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
                    lf = nn.CrossEntropyLoss(); model.train()
                    for _ in range(25):
                        opt.zero_grad(); lf(model(Xf), yf).backward(); opt.step()
                return evaluate(model, Xe, ye, Me)
        return arm

    print("\n=== Running arms (leave-one-capture-out) ===")
    import time
    def run_and_save(name, arm_fn):
        t0 = time.time()
        r = dict(zip(["acc","f1"], loco(arm_fn)))
        results[name] = r
        # checkpoint after every arm
        with open(os.path.join(OUT, "metrics_partial.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"{name} done in {time.time()-t0:.0f}s -> f1@0%={r['f1'][0.0]:.3f}", flush=True)

    run_and_save("A_real_only", arm_real_only)
    run_and_save("B_idealized", make_synth_arm(Xide, yide, Mide, proposed=False))
    run_and_save("C_physics_noadapt", make_synth_arm(Xphy, yphy, Mphy, proposed=False))
    run_and_save("D_proposed", make_synth_arm(Xphy, yphy, Mphy, proposed=True, adapt=True, use_music=True))
    run_and_save("D_abl_no_music", make_synth_arm(Xphy, yphy, Mphy, proposed=True, adapt=True, use_music=False))
    run_and_save("D_abl_no_physics", make_synth_arm(Xphy_ng, yphy_ng, Mphy_ng, proposed=True, adapt=True, use_music=True))
    run_and_save("D_abl_no_adapt", make_synth_arm(Xphy, yphy, Mphy, proposed=True, adapt=False, use_music=True))

    results["meta"] = {"wall_estimate": est, "n_synth": N_SYN,
                       "fracs": FRACS, "patch_w": PATCH_W,
                       "real_patch_counts": [int(rp[0].shape[0]) for rp in real_patches]}

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved metrics.json")

    # ---------- figures ----------
    _fig_efficiency(results, FRACS)
    _fig_ablation(results)
    _fig_bars(results)
    print("Saved figures.")
    return results


def _fig_efficiency(results, FRACS):
    plt.figure(figsize=(6.2, 4.4))
    arms = [("A_real_only","Real-only baseline","o","-"),
            ("B_idealized","Idealized-synthetic","s","--"),
            ("C_physics_noadapt","Physics-synthetic (no adapt)","^",":"),
            ("D_proposed","Proposed (physics+MUSIC-DANN)","D","-")]
    for key,label,mk,ls in arms:
        ys = [results[key]["f1"][fr] for fr in FRACS]
        plt.plot([f*100 for f in FRACS], ys, marker=mk, ls=ls, label=label, lw=2)
    plt.xlabel("Real fine-tuning data used (%)")
    plt.ylabel("Macro F1 on held-out real capture")
    plt.title("Sim-to-real data-efficiency (leave-one-capture-out)")
    plt.grid(alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "efficiency_curve.png"), dpi=150)
    plt.close()
    # csv
    import csv
    with open(os.path.join(OUT, "efficiency_curve.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm"]+[f"f1@{int(fr*100)}%" for fr in FRACS])
        for key,label,_,_ in arms:
            w.writerow([label]+[round(results[key]["f1"][fr],4) for fr in FRACS])


def _fig_ablation(results):
    fr = 0.0  # zero-shot transfer (no real labels) — the purest adaptation test
    names = ["Proposed (full)","- MUSIC anchor","- Physics guidance","- Adaptation"]
    keys = ["D_proposed","D_abl_no_music","D_abl_no_physics","D_abl_no_adapt"]
    vals = [results[k]["f1"][fr] for k in keys]
    plt.figure(figsize=(6.0,4.0))
    bars = plt.bar(names, vals, color=["#2c7fb8","#7fcdbb","#fdae6b","#d95f0e"])
    for b,v in zip(bars,vals):
        plt.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}", ha="center", fontsize=9)
    plt.ylabel("Macro F1 (zero real labels)")
    plt.title("Ablation at 0% real data (component contribution)")
    plt.xticks(rotation=12, fontsize=8); plt.ylim(0, max(vals)*1.2+0.05)
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "ablation.png"), dpi=150); plt.close()


def _fig_bars(results):
    fr = 0.0
    names = ["Real-only","Idealized-syn","Physics-syn","Proposed"]
    keys = ["A_real_only","B_idealized","C_physics_noadapt","D_proposed"]
    accs = [results[k]["acc"][fr] for k in keys]
    f1s = [results[k]["f1"][fr] for k in keys]
    x = np.arange(len(names)); w = 0.38
    plt.figure(figsize=(6.2,4.0))
    plt.bar(x-w/2, accs, w, label="Accuracy", color="#3182bd")
    plt.bar(x+w/2, f1s, w, label="Macro F1", color="#e6550d")
    plt.xticks(x, names, fontsize=9); plt.ylabel("Score (0% real labels)")
    plt.title("Zero-shot transfer to real data")
    plt.legend(); plt.grid(axis="y", alpha=0.3); plt.ylim(0,1.05); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "zeroshot_bars.png"), dpi=150); plt.close()


if __name__ == "__main__":
    res = main()
    print(json.dumps({k:v for k,v in res.items() if k!="meta"}, indent=2)[:1500])
