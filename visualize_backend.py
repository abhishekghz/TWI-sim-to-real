"""
visualize_backend.py
====================
Visualises the BACKEND of synthetic TWI data generation — the raw intermediate
stages BEFORE the final normalized preview patch. This shows *how the data looks
under the hood* at each step of the physics forward model:

  Stage 1: Raw complex S21 (real & imaginary parts) vs frequency  [what the VNA "sees"]
  Stage 2: |S21| magnitude across all 28 scan positions           [the frequency-domain B-scan]
  Stage 3: Component breakdown — wall-only vs target-only vs total [the physics decomposition]
  Stage 4: Per-position range profiles (after inverse-FFT)         [range compression]
  Stage 5: The assembled range image (dB)                         [before normalization]
  Stage 6: The final normalized patch                             [what the network sees]

Output: synthetic/backend/*.png
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "src"))
from data_io import load_real_dataset, to_range_image, normalize01, freq_axis_params, C
from synth_twi import (generate_scene, _wall_response, _target_response,
                       MATERIALS, MAT_LIST)
from music import estimate_wall_params

OUT = os.path.join(HERE, "synthetic", "backend")
os.makedirs(OUT, exist_ok=True)


def main():
    caps = load_real_dataset(os.path.join(HERE, "data", "Data two targets"))
    freq = caps[0]["raw"]["freq"]
    est = estimate_wall_params(caps[0]["raw"]["S21"], freq)
    df, B, dr, Rmax = freq_axis_params(freq)

    # Generate ONE representative scene with a fixed seed (two targets, metal+wood)
    rng = np.random.default_rng(7)
    targets = [{"x": -0.10, "y": 1.6, "material": "metal"},
               {"x": 0.08, "y": 3.0, "material": "wood"}]
    S21, meta = generate_scene(freq, n_pos=28, targets=targets,
                               eps_w=est["eps_w"], d_w=est["d_w"],
                               standoff=est["standoff"], rng=rng)
    n_pos = S21.shape[0]
    xpos = meta["xpos"]

    # ---------- Stage 1: raw complex S21 at the centre position ----------
    mid = n_pos // 2
    fghz = freq / 1e9
    fig, ax = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
    ax[0].plot(fghz, S21[mid].real, lw=1.2, color="#1f77b4")
    ax[0].set_ylabel("Re{S21}"); ax[0].set_title(f"Stage 1 — Raw complex S21 (scan position {mid})")
    ax[0].grid(alpha=0.3)
    ax[1].plot(fghz, S21[mid].imag, lw=1.2, color="#d62728")
    ax[1].set_ylabel("Im{S21}"); ax[1].set_xlabel("Frequency (GHz)"); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "stage1_raw_S21.png"), dpi=140); plt.close()

    # ---------- Stage 2: |S21| frequency-domain B-scan (all positions) ----------
    plt.figure(figsize=(7, 4))
    magdb = 20*np.log10(np.abs(S21) + 1e-12)
    im = plt.imshow(magdb, aspect="auto", cmap="magma",
                    extent=[fghz[0], fghz[-1], xpos[-1], xpos[0]])
    plt.colorbar(im, label="|S21| (dB)")
    plt.xlabel("Frequency (GHz)"); plt.ylabel("Scan position (m)")
    plt.title("Stage 2 — Frequency-domain B-scan |S21| (28 positions × 201 freqs)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "stage2_freq_bscan.png"), dpi=140); plt.close()

    # ---------- Stage 3: component breakdown (wall-only / target-only / total) ----------
    Hwall = _wall_response(freq, est["eps_w"], est["d_w"])
    # wall-only S21 (no targets)
    S_wall = np.tile(Hwall * 0.5, (n_pos, 1))
    # target-only S21 (no wall)
    S_tgt = np.zeros_like(S21)
    for i, xa in enumerate(xpos):
        for tg in targets:
            m = MATERIALS[tg["material"]]
            R = np.hypot(tg["x"] - xa, tg["y"] + est["standoff"])
            S_tgt[i] += _target_response(freq, R, m["refl"], m["loss"],
                                         est["eps_w"], est["d_w"])
    def to_img(s):
        img, rngax = to_range_image(s, freq, bg_subtract=False)
        return img, rngax
    img_wall, rngax = to_img(S_wall)
    img_tgt, _ = to_img(S_tgt)
    img_tot, _ = to_img(S21)
    keep = rngax <= 6
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for a, im_, ti in zip(ax, [img_wall, img_tgt, img_tot],
                          ["Wall-only return", "Target-only return", "Total (wall + targets + noise)"]):
        h = a.imshow(im_[:, keep], aspect="auto", cmap="viridis",
                     extent=[rngax[keep][0], rngax[keep][-1], n_pos, 0])
        a.set_title(ti, fontsize=10); a.set_xlabel("Range (m)"); a.set_ylabel("Scan position")
    plt.suptitle("Stage 3 — Physics decomposition (this is the key difference from idealized data)",
                 fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "stage3_components.png"), dpi=140); plt.close()

    # ---------- Stage 4: per-position range profiles (waterfall) ----------
    img_full, rngax = to_range_image(S21, freq, bg_subtract=False, log=False)
    keep = rngax <= 6
    plt.figure(figsize=(7, 5))
    offset = 0
    step = np.max(img_full[:, keep]) * 0.6 + 1e-9
    for i in range(0, n_pos, 2):  # every other position for clarity
        prof = img_full[i, keep]
        plt.plot(rngax[keep], prof + offset, lw=0.9, color=plt.cm.viridis(i/n_pos))
        offset += step
    plt.xlabel("Range (m)"); plt.ylabel("Scan position (stacked / offset)")
    plt.title("Stage 4 — Per-position range profiles after inverse-FFT (range compression)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "stage4_range_profiles.png"), dpi=140); plt.close()

    # ---------- Stage 5 & 6: assembled dB image vs final normalized patch ----------
    img_db, rngax = to_range_image(S21, freq, bg_subtract=True, log=True)
    keep = rngax <= 6
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    h0 = ax[0].imshow(img_db[:, keep], aspect="auto", cmap="viridis",
                      extent=[rngax[keep][0], rngax[keep][-1], n_pos, 0])
    plt.colorbar(h0, ax=ax[0], label="dB")
    ax[0].set_title("Stage 5 — Assembled range image (dB, after clutter subtraction)")
    ax[0].set_xlabel("Range (m)"); ax[0].set_ylabel("Scan position")
    norm = normalize01(img_db[:, keep])
    h1 = ax[1].imshow(norm, aspect="auto", cmap="viridis",
                      extent=[rngax[keep][0], rngax[keep][-1], n_pos, 0])
    plt.colorbar(h1, ax=ax[1], label="[0,1]")
    ax[1].set_title("Stage 6 — Final normalized image (what the CNN sees)")
    ax[1].set_xlabel("Range (m)"); ax[1].set_ylabel("Scan position")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "stage5_6_assembled_vs_final.png"), dpi=140); plt.close()

    # ---------- Combined storyboard ----------
    print("Backend visualization scene:")
    print(f"  wall: eps_w={est['eps_w']:.2f}, d_w={est['d_w']:.3f} m, standoff={est['standoff']:.2f} m")
    print(f"  targets: {[(t['material'], t['x'], t['y']) for t in targets]}")
    print(f"  range resolution {dr*100:.1f} cm, max range {Rmax:.1f} m")
    print("Saved backend stages to:", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  -", f)


if __name__ == "__main__":
    main()
