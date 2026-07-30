"""
generate_synthetic_dataset.py
=============================
Generates and SAVES the complete synthetic Through-Wall Imaging (TWI) dataset
used in the sim-to-real study, with full provenance (labels, wall parameters,
per-scene metadata). Run this once to materialise all synthetic data as files
on disk under  synthetic/.

What it produces
----------------
synthetic/
  physics_guided/                 <- physics-faithful, conditioned on REAL wall
      images.npy                  (N, 28, 48)  float32 range-image patches
      labels.npy                  (N,)         0 = background/clutter, 1 = target
      music.npy                   (N, 64)      MUSIC pseudo-spectrum feature
      metadata.json               per-scene wall params, target positions, materials
  physics_unguided/               <- physics-faithful, RANDOM wall (ablation source)
      images.npy / labels.npy / music.npy / metadata.json
  idealized/                      <- baseline-paper style (square + Gaussian noise)
      images.npy / labels.npy / music.npy / metadata.json
  preview/                        <- PNGs so you can SEE the synthetic data
      physics_guided_grid.png
      idealized_grid.png
      class_balance.png
  DATASET_SUMMARY.json            counts, shapes, wall estimate, RNG seeds

Reproducibility: every set is generated from a fixed RNG seed (printed and saved),
so re-running reproduces identical data.
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_io import load_real_dataset, to_range_image, normalize01
from synth_twi import generate_scene, MAT_LIST
from music import estimate_wall_params, wall_dist_from_estimate, music_feature

# ----------------------------------------------------------------------------
REAL_ROOT = os.path.join(os.path.dirname(__file__), "data", "Data two targets")
OUT = os.path.join(os.path.dirname(__file__), "synthetic")
PATCH_W = 48
N_SCENES = 400          # -> 800 patches per physics set (1 target patch + 1 bg patch)
SEEDS = {"physics_guided": 10, "physics_unguided": 11, "idealized": 12}


def save_set(name, X, y, M, metas):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "images.npy"), X.astype(np.float32))
    np.save(os.path.join(d, "labels.npy"), y.astype(np.int64))
    np.save(os.path.join(d, "music.npy"), M.astype(np.float32))
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump(metas, f, indent=2)
    print(f"  saved {name}: images{X.shape} labels{y.shape} "
          f"music{M.shape} -> {d}")


def gen_physics(freq, n_scenes, seed, wall_dist):
    """Physics-faithful generator. One positive + one negative patch per scene."""
    rng = np.random.default_rng(seed)
    half = PATCH_W // 2
    X, y, M, metas = [], [], [], []
    for s in range(n_scenes):
        kw = {}
        if wall_dist is not None:
            eps_w, d_w, standoff = wall_dist(rng)
            kw.update(eps_w=eps_w, d_w=d_w, standoff=standoff)
        S21, meta = generate_scene(freq, n_pos=28,
                                   n_targets=int(rng.integers(1, 3)),
                                   rng=rng, **kw)
        img, rngax = to_range_image(S21, freq)
        mf = music_feature(S21)
        # positive patch centred on a target
        ty = float(np.mean([t["y"] for t in meta["targets"]]))
        ctr = int(np.clip(np.argmin(np.abs(rngax - ty)), half, img.shape[1]-half-1))
        X.append(normalize01(img[:, ctr-half:ctr+half])); y.append(1); M.append(mf)
        # negative patch from a target-free near-range/clutter region
        neg = int(np.clip(np.argmin(np.abs(rngax - 0.4)), half, img.shape[1]-half-1))
        if abs(neg - ctr) < PATCH_W:
            neg = min(img.shape[1]-half-1, ctr + PATCH_W)
        X.append(normalize01(img[:, neg-half:neg+half])); y.append(0); M.append(mf)
        metas.append({
            "scene": s,
            "eps_w": round(meta["eps_w"], 4),
            "d_w": round(meta["d_w"], 4),
            "standoff": round(meta["standoff"], 4),
            "n_targets": meta["n_targets"],
            "materials": meta["materials"],
            "target_xy": [[round(t["x"], 4), round(t["y"], 4)] for t in meta["targets"]],
            "positive_patch_range_m": round(float(rngax[ctr]), 3),
            "negative_patch_range_m": round(float(rngax[neg]), 3),
        })
    return np.stack(X), np.array(y), np.stack(M), metas


def gen_idealized(n_scenes, seed):
    """Baseline-paper style: bright square on Gaussian noise. No TWI physics."""
    rng = np.random.default_rng(seed)
    X, y, M, metas = [], [], [], []
    npos = 28
    for s in range(n_scenes * 2):   # match physics set size
        bg = rng.normal(0.5, 0.15, size=(npos, PATCH_W)).astype(np.float32)
        lab = int(rng.integers(0, 2))
        sq = None
        if lab == 1:
            sz = 6
            r0 = int(rng.integers(0, npos - sz)); c0 = int(rng.integers(0, PATCH_W - sz))
            bg[r0:r0+sz, c0:c0+sz] = 1.0
            sq = [r0, c0, sz]
        X.append(normalize01(bg)); y.append(lab)
        M.append(rng.standard_normal(64).astype(np.float32))  # uninformative MUSIC
        metas.append({"scene": s, "label": lab, "square_rc_size": sq,
                      "note": "idealized: no wall, no multipath, no material physics"})
    return np.stack(X), np.array(y), np.stack(M), metas


def preview_grid(X, y, title, path, n=12):
    idx = np.arange(min(n, len(X)))
    cols = 4; rows = int(np.ceil(len(idx)/cols))
    plt.figure(figsize=(cols*2.4, rows*1.7))
    for i, k in enumerate(idx):
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(X[k], aspect="auto", cmap="viridis")
        ax.set_title(f"label={int(y[k])}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(title, fontsize=11)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Loading REAL data to estimate wall parameters (physics-guided conditioning)...")
    caps = load_real_dataset(REAL_ROOT)
    freq = caps[0]["raw"]["freq"]
    est = estimate_wall_params(caps[0]["raw"]["S21"], freq)
    wall = wall_dist_from_estimate(est)
    print(f"  estimated wall: eps_w={est['eps_w']:.2f}, d_w={est['d_w']:.3f} m, "
          f"standoff={est['standoff']:.3f} m")

    print("\nGenerating PHYSICS-GUIDED set (conditioned on real wall)...")
    Xpg, ypg, Mpg, mpg = gen_physics(freq, N_SCENES, SEEDS["physics_guided"], wall)
    save_set("physics_guided", Xpg, ypg, Mpg, mpg)

    print("Generating PHYSICS-UNGUIDED set (random wall; ablation source)...")
    Xpu, ypu, Mpu, mpu = gen_physics(freq, N_SCENES, SEEDS["physics_unguided"], None)
    save_set("physics_unguided", Xpu, ypu, Mpu, mpu)

    print("Generating IDEALIZED set (square + Gaussian; prior approach)...")
    Xid, yid, Mid, mid = gen_idealized(N_SCENES, SEEDS["idealized"])
    save_set("idealized", Xid, yid, Mid, mid)

    # previews
    pv = os.path.join(OUT, "preview"); os.makedirs(pv, exist_ok=True)
    preview_grid(Xpg, ypg, "Physics-guided synthetic patches (conditioned on real wall)",
                 os.path.join(pv, "physics_guided_grid.png"))
    preview_grid(Xid, yid, "Idealized synthetic patches (square + Gaussian)",
                 os.path.join(pv, "idealized_grid.png"))
    # class balance
    plt.figure(figsize=(6, 3.4))
    sets = {"physics_guided": ypg, "physics_unguided": ypu, "idealized": yid}
    xpos = np.arange(len(sets)); w = 0.35
    bg = [np.sum(v == 0) for v in sets.values()]
    tg = [np.sum(v == 1) for v in sets.values()]
    plt.bar(xpos - w/2, bg, w, label="background (0)", color="#9ecae1")
    plt.bar(xpos + w/2, tg, w, label="target (1)", color="#fc9272")
    plt.xticks(xpos, list(sets.keys()), fontsize=8); plt.ylabel("count")
    plt.title("Synthetic class balance"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(pv, "class_balance.png"), dpi=130); plt.close()

    summary = {
        "patch_shape": [28, PATCH_W],
        "n_scenes_per_physics_set": N_SCENES,
        "patches_per_physics_set": int(Xpg.shape[0]),
        "rng_seeds": SEEDS,
        "real_wall_estimate": est,
        "materials": MAT_LIST,
        "label_meaning": {"0": "background / wall clutter", "1": "hidden target present"},
        "frequency_band_GHz": [float(freq[0]/1e9), float(freq[-1]/1e9)],
        "n_frequencies": int(len(freq)),
        "sets": {
            "physics_guided": {"images": list(Xpg.shape), "balance": [int(np.sum(ypg==0)), int(np.sum(ypg==1))]},
            "physics_unguided": {"images": list(Xpu.shape), "balance": [int(np.sum(ypu==0)), int(np.sum(ypu==1))]},
            "idealized": {"images": list(Xid.shape), "balance": [int(np.sum(yid==0)), int(np.sum(yid==1))]},
        },
    }
    with open(os.path.join(OUT, "DATASET_SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote synthetic/DATASET_SUMMARY.json")
    print("Done. Synthetic dataset materialised under:", OUT)


if __name__ == "__main__":
    main()
