"""
inspect_synthetic.py
====================
Quick look at the saved synthetic dataset: prints shapes/balance, shows a few
sample patches with their labels and the scene metadata that produced them.

Usage:
    python3 inspect_synthetic.py                 # summary of all sets
    python3 inspect_synthetic.py physics_guided  # detail one set
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(__file__)
SYN = os.path.join(HERE, "synthetic")
SETS = ["physics_guided", "physics_unguided", "idealized"]


def summarize():
    if not os.path.isdir(SYN):
        print("synthetic/ not found. Run: python3 generate_synthetic_dataset.py")
        return
    s = json.load(open(os.path.join(SYN, "DATASET_SUMMARY.json")))
    print("=== Synthetic dataset summary ===")
    print("patch shape:", s["patch_shape"])
    print("frequency band (GHz):", s["frequency_band_GHz"], "| n_freq:", s["n_frequencies"])
    print("real wall estimate:", s["real_wall_estimate"])
    print("label meaning:", s["label_meaning"])
    print("seeds:", s["rng_seeds"])
    for name, info in s["sets"].items():
        print(f"  {name:18s} images {info['images']}  balance(bg,target)={info['balance']}")


def detail(name):
    d = os.path.join(SYN, name)
    X = np.load(os.path.join(d, "images.npy"))
    y = np.load(os.path.join(d, "labels.npy"))
    M = np.load(os.path.join(d, "music.npy"))
    meta = json.load(open(os.path.join(d, "metadata.json")))
    print(f"=== {name} ===")
    print("images:", X.shape, "labels:", np.bincount(y), "music:", M.shape)
    print("\nfirst 3 scene metadata entries:")
    for m in meta[:3]:
        print(" ", json.dumps(m))
    # ascii peek at one target and one background patch
    ti = int(np.where(y == 1)[0][0]); bi = int(np.where(y == 0)[0][0])
    for idx, lab in [(ti, "TARGET"), (bi, "BACKGROUND")]:
        patch = X[idx]
        col_energy = patch.mean(axis=0)  # energy along range
        bar = "".join("#" if v > 0.6 else ("+" if v > 0.4 else ".") for v in col_energy)
        print(f"\n{lab} patch #{idx} range-energy profile:\n  {bar}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        detail(sys.argv[1])
    else:
        summarize()
