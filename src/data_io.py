"""
data_io.py
==========
Loading and SAR-style imaging for the real Through-Wall Imaging (TWI) VNA data.

The measured data are stepped-frequency S21 measurements:
    - frequencies      : (1, Nf) Hz, 1.5 - 3.5 GHz, 201 points (df = 10 MHz, B = 2 GHz)
    - dataMeasuredReal  : (Npos, Nf) real part of S21
    - dataMeasuredImag  : (Npos, Nf) imag part of S21
    - dataMeasured1     : (Npos, Nf) |S21| in dB (calibrated)
where Npos = 28 antenna/scan positions.

Physics:
    Range resolution  dr   = c / (2 B)        = 7.5 cm   (matches the paper)
    Max unambig range Rmax = c / (2 df)        = 15 m     (matches Fig. 6/7 axes)

A range profile per scan position is obtained by an inverse DFT over frequency
(after windowing). Stacking the 28 range profiles gives a range vs cross-range
B-scan image, which is the input modality for both the baseline and the
deep-learning pipeline.
"""

import os
import numpy as np
import scipy.io as sio

C = 299_792_458.0  # speed of light (m/s)


def load_mat(path):
    """Load one TWI .mat capture into a clean dict of numpy arrays."""
    m = sio.loadmat(path)
    f = np.asarray(m["frequencies"]).ravel().astype(np.float64)          # (Nf,)
    real = np.asarray(m["dataMeasuredReal"]).astype(np.float64)          # (Npos,Nf)
    imag = np.asarray(m["dataMeasuredImag"]).astype(np.float64)
    s21 = real + 1j * imag                                              # complex
    mag_db = np.asarray(m["dataMeasured1"]).astype(np.float64)
    return {
        "freq": f,
        "S21": s21,
        "mag_db": mag_db,
        "Npos": s21.shape[0],
        "Nf": s21.shape[1],
        "path": path,
    }


def freq_axis_params(freq):
    df = float(freq[1] - freq[0])
    B = float(freq[-1] - freq[0])
    dr = C / (2.0 * B)
    Rmax = C / (2.0 * df)
    return df, B, dr, Rmax


def background_subtract(s21):
    """
    Coherent background (clutter) subtraction.
    The dominant early-range return is the wall + antenna coupling, common to all
    scan positions. Subtracting the per-frequency mean across positions
    suppresses this stationary clutter (a standard TWI pre-processing step,
    analogous to mean/SVD subtraction in GPR).
    """
    bg = s21.mean(axis=0, keepdims=True)
    return s21 - bg


def to_range_image(s21, freq, n_pad=512, r_max_keep=8.0,
                   window=True, bg_subtract=True, log=True):
    """
    Convert (Npos, Nf) complex S21 -> (Npos, Nr) range image (B-scan).

    Returns
    -------
    img   : (Npos, Nr) float32, magnitude (dB if log=True) range image
    rng   : (Nr,) range axis in metres (cropped to r_max_keep)
    """
    df, B, dr, Rmax = freq_axis_params(freq)
    x = s21.copy()
    if bg_subtract:
        x = background_subtract(x)
    if window:
        w = np.hanning(x.shape[1])[None, :]
        x = x * w
    rp = np.fft.ifft(x, n=n_pad, axis=1)                  # (Npos, n_pad)
    rng_full = np.arange(n_pad) * (C / (2.0 * df * n_pad))
    keep = rng_full <= r_max_keep
    mag = np.abs(rp[:, keep])
    rng = rng_full[keep]
    if log:
        mag = 20.0 * np.log10(mag + 1e-12)
    img = mag.astype(np.float32)
    return img, rng


def normalize01(img):
    """Min-max normalize an image to [0,1] (per-image)."""
    lo, hi = np.nanmin(img), np.nanmax(img)
    if hi - lo < 1e-9:
        return np.zeros_like(img)
    return ((img - lo) / (hi - lo)).astype(np.float32)


def load_real_dataset(root):
    """
    Walk the 'Data two targets' directory and return a list of captures, each
    already imaged and normalized.

    Each capture is one (28 x Nf) acquisition -> one range image (a 'scene').
    Both folders correspond to the TWO-TARGET case (label = 2 targets present).
    """
    captures = []
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        mp = os.path.join(d, "data.mat")
        if os.path.isfile(mp):
            raw = load_mat(mp)
            img, rng = to_range_image(raw["S21"], raw["freq"])
            captures.append({
                "id": sub,
                "raw": raw,
                "image": img,            # (28, Nr) dB
                "image01": normalize01(img),
                "range": rng,
                "n_targets": 2,          # this dataset = two-target case
            })
    return captures


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Data two targets")
    caps = load_real_dataset(root)
    for c in caps:
        df, B, dr, Rmax = freq_axis_params(c["raw"]["freq"])
        print(f"[{c['id']}] image {c['image'].shape}  dr={dr*100:.1f} cm  "
              f"Rmax={Rmax:.1f} m  range_kept=[{c['range'][0]:.2f},{c['range'][-1]:.2f}] m")
