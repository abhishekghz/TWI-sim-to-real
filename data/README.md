# Real Through-Wall Imaging (TWI) Data

Two captures of the **two-target** case, collected with a 2-port VNA + Vivaldi
antenna (the setup from Yadav et al., DSJ 2026).

```
Data two targets/
  1809/data.mat
  2359/data.mat
```

Each `data.mat` contains:
| variable | shape | meaning |
|----------|-------|---------|
| `frequencies` | (1, 201) | 1.5–3.5 GHz, 10 MHz step |
| `dataMeasuredReal` | (28, 201) | Re{S21}, 28 scan positions |
| `dataMeasuredImag` | (28, 201) | Im{S21} |
| `dataMeasured1` | (28, 201) | calibrated |S21| in dB |

The scripts expect this folder at `data/Data two targets/`. If you move it,
update `REAL_ROOT` / `ROOT` at the top of `generate_synthetic_dataset.py` and
`src/run_experiments.py`.
