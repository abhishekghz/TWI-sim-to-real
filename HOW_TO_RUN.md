# HOW TO RUN — Step by Step

This guide takes you from a fresh machine to the final results and paper figures.

---

## 0. Folder layout

```
twi_project/
├── src/
│   ├── data_io.py            # load real .mat captures + frequency→range imaging
│   ├── synth_twi.py          # PHYSICS-BASED synthetic TWI forward model
│   ├── music.py              # MUSIC subspace + wall-parameter estimation
│   ├── models.py             # CNN baseline + MUSIC-anchored domain-adversarial net
│   ├── datasets.py           # real patch extraction + synthetic patch assembly
│   └── run_experiments.py    # 5 arms + 3 ablations, leave-one-capture-out
├── generate_synthetic_dataset.py   # STEP 1: make & save all synthetic data
├── train_from_saved.py             # STEP 2 (alt): train from saved synthetic data
├── synthetic/                      # (created by step 1) the saved synthetic dataset
├── outputs/                        # metrics.json, efficiency_curve.csv
└── figures/                        # all result figures (.png)
```

The REAL data is expected at:  `Data two targets/1809/data.mat` and
`Data two targets/2359/data.mat`. Edit the `REAL_ROOT` / `ROOT` path at the top of
`generate_synthetic_dataset.py` and `src/run_experiments.py` if you move it.

---

## 1. Install dependencies

```bash
pip install torch scipy numpy scikit-learn matplotlib
```

GPU is optional. Everything runs on CPU (just slower).

---

## 2. Generate and SAVE the synthetic dataset (STEP 1)

```bash
cd twi_project
python3 generate_synthetic_dataset.py
```

This:
1. loads the real captures and estimates the wall parameters,
2. generates three synthetic sets (physics-guided, physics-unguided, idealized),
3. saves them under `synthetic/` as `.npy` + `metadata.json`,
4. writes preview PNGs and `DATASET_SUMMARY.json`.

Inspect `synthetic/preview/physics_guided_grid.png` to SEE the synthetic data.

---

## 3. Run the experiments (STEP 2)

The simplest, self-contained way (regenerates data internally from the same
seeds and runs everything):

```bash
python3 src/run_experiments.py
```

This trains all five arms with leave-one-capture-out and writes:
- `outputs/metrics.json`         — all accuracy/F1 numbers
- `outputs/efficiency_curve.csv` — F1 vs real-data fraction
- `figures/efficiency_curve.png`, `figures/ablation.png`, `figures/zeroshot_bars.png`

On CPU this takes ~15–25 min at the default small settings. To make it faster or
higher quality, edit these near the top of `src/run_experiments.py`:
- `N_SYN`        (synthetic scenes; higher = better, slower)
- `epochs=...`   in `train_plain` / `train_dann` (higher = better, slower)
- `FRACS`        (real fine-tuning fractions to sweep)

For best published numbers: use a GPU (set `DEV = "cuda"`), raise `N_SYN` to
~1500 and epochs back to 60/80.

---

## 4. Read the results

```bash
python3 - << 'PY'
import json
m = json.load(open("outputs/metrics.json"))
for k,v in m.items():
    if k=="meta": continue
    print(k, "F1:", {f:round(x,3) for f,x in v["f1"].items()})
PY
```

Expected pattern (small-run numbers; trends are what matter):
- Idealized transfers poorly at 0% real (~0.16)
- Physics-synthetic jumps (~0.45) — **physics matters**
- Proposed best with fine-tuning (~0.80 F1, ~0.90 accuracy)
- Ablation: removing physics guidance hurts most; MUSIC anchor adds a clear gain

---

## 4b. (Optional) Visualize the synthetic-data BACKEND

To see the raw generation stages *before* the final preview (raw complex S21,
frequency B-scan, the wall/target physics decomposition, range profiles, and the
assembled-vs-normalized image):

```bash
python3 visualize_backend.py
```

Writes `synthetic/backend/stage1..6_*.png` and a combined
`synthetic/backend/storyboard_all_stages.png`.

---

## Troubleshooting

- **"synthetic/ not found"** → run step 2 first (`generate_synthetic_dataset.py`).
- **".mat not found"** → fix the real-data path at the top of the scripts.
- **Runs too slow** → lower `N_SYN` and epochs, or use a GPU.
- **All-one-class real patches** → the patch labeller keys off the two dominant
  post-wall returns; if you change the wall standoff a lot, adjust the
  `post = rng > 0.7` threshold in `src/datasets.py::extract_real_patches`.
