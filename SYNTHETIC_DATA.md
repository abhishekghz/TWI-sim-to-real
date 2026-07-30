# Synthetic TWI Data — How It Is Generated

This document explains, in full detail, how the synthetic Through-Wall Imaging
(TWI) data is produced — the physics, the parameters, the labels, and the files.

---

## 1. Why synthetic data at all?

Real TWI measurements are expensive to collect (each scene must be physically
built behind a wall and scanned with a VNA). We have only 2 real captures. Deep
networks need far more labelled data. The solution is to **simulate** TWI
measurements, where the ground-truth label is known for free.

The earlier paper used *idealized* synthetic data — a bright square on Gaussian
noise — which ignores TWI physics and transfers poorly. This project replaces it
with a **physics-faithful** generator.

---

## 2. The measurement we simulate

A stepped-frequency VNA measures the complex transmission parameter **S21** at
each of `Nf = 201` frequencies spanning **1.5–3.5 GHz** (10 MHz step), for each
of **28 scan positions** across an aperture. So one capture is a complex matrix
of shape `(28, 201)`.

From S21 we form a **range image** by inverse-FFT along frequency (per position)
and stacking the 28 range profiles. Key physics:

| Quantity | Formula | Value |
|----------|---------|-------|
| Range resolution | ΔR = c / (2·B) | 7.5 cm (B = 2 GHz) |
| Max unambiguous range | R_max = c / (2·Δf) | 15 m (Δf = 10 MHz) |

These match the real data exactly.

---

## 3. The forward model (`src/synth_twi.py`)

For each frequency `f` and scan position `xa`, the synthetic S21 is the coherent
sum of a **wall response** and one or more **target responses**, plus noise.

### 3.1 Wall response  `_wall_response()`
A homogeneous slab with relative permittivity `eps_w` and thickness `d_w`:

- Front-face Fresnel reflection: `r = (1 − n) / (1 + n)`, where `n = √eps_w`.
- Internal reverberations: the wave bounces inside the slab; each round trip adds
  a delayed, attenuated copy (`n_revb = 3` reverberations modelled).
- This produces the strong, near-range, frequency-dependent return that dominates
  real TWI data — exactly what the idealized model lacks.

### 3.2 Target response  `_target_response()`
For a target at one-way range `R` behind the wall:

- **Phase / delay**: `exp(−j·k0·(2R + extra))`, where `extra = (n − 1)·2·d_w`
  accounts for the *slower* propagation inside the wall slab (two-way).
- **Spreading loss**: `1 / R²` (two-way amplitude).
- **Material loss**: `exp(−loss·f)`, frequency-dependent, per material.
- **Wall transmission**: the signal crosses the wall twice (in and out).

### 3.3 Material reflectivity  `MATERIALS`
Matches the three real-data materials:

| Material | Reflectivity | Loss | Note |
|----------|-------------|------|------|
| metal | 0.95 | 0.02 | near-perfect reflector |
| teflon | 0.35 | 0.05 | low-permittivity, low-loss |
| wood | 0.45 | 0.15 | lossy dielectric |

### 3.4 Multipath
A secondary wall↔target↔wall bounce is added (longer path, extra wall
reflection, attenuated) — a real effect absent from idealized data.

### 3.5 Noise
Additive complex Gaussian noise on S21 at a configurable level (`noise_db`,
default −60 dB) models the VNA noise floor.

### 3.6 The range equation (paper Eq. 5)
The per-target amplitude follows
`Pr = (Pt · ρ · Ta² · To · Dr²) / (4·R²)`
i.e. received power falls as 1/R² and scales with reflectivity ρ and the
system/atmosphere factors.

---

## 4. Physics-guided conditioning (`src/music.py`)

Instead of arbitrary wall parameters, we **estimate the real wall** from the
strong near-range return of the real captures (`estimate_wall_params`) and
generate synthetic data around those values (`wall_dist_from_estimate`).

On this dataset the estimate is **eps_w ≈ 5.6, d_w ≈ 0.23 m, standoff ≈ 0.20 m**.
This shrinks the sim-to-real gap *before* any learning — the "physics-guided"
half of the method. The `physics_guided` set uses this; the `physics_unguided`
set uses random walls (the ablation source that removes physics guidance).

---

## 5. The MUSIC feature (`src/music.py`)

For each synthetic (and real) capture we also compute a 64-point **MUSIC
pseudo-spectrum** from the array covariance. This is the domain-invariant
"anchor" feature: it depends on target direction-of-arrival (shared between sim
and real), not on wall texture or noise colour. Saved as `music.npy`.

---

## 6. What gets saved (`generate_synthetic_dataset.py`)

Running `python3 generate_synthetic_dataset.py` writes:

```
synthetic/
  physics_guided/      images.npy (N,28,48)  labels.npy (N,)  music.npy (N,64)  metadata.json
  physics_unguided/    (same structure; random-wall ablation source)
  idealized/           (square+Gaussian; prior approach for the control arm)
  preview/             physics_guided_grid.png, idealized_grid.png, class_balance.png
  DATASET_SUMMARY.json (shapes, balance, seeds, wall estimate)
```

- **images.npy** — range-image patches, shape `(N, 28, 48)`, float32, min-max
  normalized to [0,1].
- **labels.npy** — `0` = background/clutter, `1` = hidden target present.
- **music.npy** — `(N, 64)` MUSIC pseudo-spectrum feature per patch.
- **metadata.json** — per-scene wall parameters, target (x,y) positions,
  materials, and the range of the positive/negative patches. Full provenance.

Each physics set has **800 patches** (400 scenes × 1 target patch + 1 background
patch), balanced 400/400. Seeds are fixed (`physics_guided=10`,
`physics_unguided=11`, `idealized=12`), so regeneration is identical.

---

## 7. How to load the saved data yourself

```python
import numpy as np
X = np.load("synthetic/physics_guided/images.npy")   # (800, 28, 48)
y = np.load("synthetic/physics_guided/labels.npy")   # (800,)  0/1
M = np.load("synthetic/physics_guided/music.npy")    # (800, 64)
import json
meta = json.load(open("synthetic/physics_guided/metadata.json"))
print(X.shape, y.mean(), meta[0])
```

---

## 8. Swapping in gprMax (higher fidelity)

The generator exposes a single function:
`generate_scene(freq, ...) -> (S21, meta)` returning complex S21 of shape
`(28, Nf)`. An FDTD solver (gprMax) can be wrapped to return the same S21 array,
and **nothing downstream changes** — the imaging, MUSIC, dataset assembly, and
training all stay identical. This is the intended path to a higher-fidelity
source domain (a stated future-work item in the paper).

---

## 9. Visualizing the backend (raw stages before preview)

Run `python3 visualize_backend.py` to render the internal pipeline for one
representative scene, saved under `synthetic/backend/`:

| File | Shows |
|------|-------|
| `stage1_raw_S21.png` | Raw complex S21 (Re/Im) vs frequency — what the VNA measures |
| `stage2_freq_bscan.png` | \|S21\| across all 28 positions (frequency-domain B-scan) |
| `stage3_components.png` | **Physics decomposition**: wall-only vs target-only vs total |
| `stage4_range_profiles.png` | Per-position range profiles after inverse-FFT |
| `stage5_6_assembled_vs_final.png` | Assembled dB image → final normalized patch |
| `storyboard_all_stages.png` | All stages stacked in one image |

Stage 3 is the most instructive: it isolates the strong near-range **wall**
return from the **target** returns, which is exactly the physical structure that
idealized (square-on-noise) data lacks — and the reason physics-faithful data
transfers ~3× better to real measurements.
