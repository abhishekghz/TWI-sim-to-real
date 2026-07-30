"""
music.py
========
MUSIC signal-subspace front-end + wall-parameter estimation.

Two roles in the proposed pipeline:

1. MUSIC subspace anchor (feature-level novelty):
   The array covariance R = (1/Nf) S S^H is decomposed; the signal subspace
   (top-K eigenvectors) is a representation governed by target geometry/DOA,
   which is physically shared between sim and real domains, while nuisance
   domain factors (wall texture, noise colour) live in the noise subspace.
   We expose the MUSIC pseudo-spectrum and signal-subspace projection as a
   domain-invariant feature fed alongside the range image.

2. Wall-parameter estimation (physics-guided conditioning):
   The strong near-range wall return lets us estimate effective (eps_w, d_w,
   standoff) from the real data, which then conditions the synthetic generator.
"""

import numpy as np

C = 299_792_458.0


def array_covariance(S21):
    """
    R from the (Npos, Nf) snapshots. Treat each frequency as a snapshot of the
    Npos-element spatial array: x(f) in C^{Npos}. R = (1/Nf) sum_f x x^H.
    """
    X = S21  # (Npos, Nf)
    R = (X @ X.conj().T) / X.shape[1]
    return R  # (Npos, Npos)


def music_spectrum(S21, n_sources=2, n_grid=181, dmax=0.5):
    """
    1-D spatial MUSIC pseudo-spectrum over cross-range steering angles.
    Returns (angles, spectrum, signal_subspace).
    """
    R = array_covariance(S21)
    Npos = R.shape[0]
    w, V = np.linalg.eigh(R)            # ascending eigenvalues
    idx = np.argsort(w)[::-1]
    V = V[:, idx]
    Es = V[:, :n_sources]              # signal subspace
    En = V[:, n_sources:]             # noise subspace

    # steering vectors over a normalized cross-range/DOA grid
    angles = np.linspace(-1, 1, n_grid)        # sin(theta)-like normalized axis
    pos = np.arange(Npos)
    Pmusic = np.zeros(n_grid)
    for i, a in enumerate(angles):
        sv = np.exp(1j * np.pi * pos * a)       # ULA steering vector
        sv = sv / np.linalg.norm(sv)
        denom = np.abs(sv.conj() @ (En @ En.conj().T) @ sv)
        Pmusic[i] = 1.0 / (denom + 1e-12)
    Pmusic = Pmusic / Pmusic.max()
    return angles, Pmusic, Es


def music_feature(S21, n_sources=2, n_grid=64):
    """
    Compact, real-valued MUSIC feature vector for the network:
    the (log) pseudo-spectrum resampled to n_grid points. Domain-invariant
    because it depends on target DOA, not on wall/noise texture.
    """
    angles, P, Es = music_spectrum(S21, n_sources=n_sources, n_grid=n_grid)
    feat = np.log(P + 1e-6)
    feat = (feat - feat.mean()) / (feat.std() + 1e-9)
    return feat.astype(np.float32)


def estimate_wall_params(S21, freq, standoff_guess=0.5):
    """
    Estimate effective wall parameters from the strong near-range return.

    Method:
      * Form the mean range profile (coherent over positions keeps the wall).
      * Locate the first dominant peak  -> round-trip delay -> standoff.
      * Use the spectral slope/decay of the early-time return as a proxy for
        eps_w (denser wall -> larger delay spread) and d_w.

    Returns dict(eps_w, d_w, standoff). These are EFFECTIVE values used only to
    condition the synthetic source domain (physics-guided), not exact inversion.
    """
    df = float(freq[1] - freq[0])
    n_pad = 1024
    w = np.hanning(S21.shape[1])[None, :]
    rp = np.fft.ifft((S21.mean(axis=0, keepdims=True)) * w, n=n_pad, axis=1).ravel()
    rng = np.arange(n_pad) * (C / (2 * df * n_pad))
    mag = np.abs(rp)
    near = rng < 3.0
    pk = np.argmax(mag[near])
    standoff = max(0.1, float(rng[near][pk]) / 1.0)  # round-trip already in range

    # crude eps_w proxy from the width of the wall return (reverberation tail)
    region = mag[near]
    half = region.max() * 0.5
    width_bins = np.sum(region > half)
    d_w = np.clip(0.10 + 0.01 * width_bins, 0.10, 0.40)
    eps_w = np.clip(3.0 + 0.2 * width_bins, 3.0, 7.0)
    return {"eps_w": float(eps_w), "d_w": float(d_w),
            "standoff": float(np.clip(standoff, 0.2, 1.2))}


def wall_dist_from_estimate(est, jitter=0.15):
    """
    Build a sampler that draws (eps_w, d_w, standoff) around the estimated wall
    parameters, for physics-guided synthetic generation.
    """
    def sampler(rng):
        eps_w = est["eps_w"] * (1 + jitter * rng.standard_normal())
        d_w = est["d_w"] * (1 + jitter * rng.standard_normal())
        standoff = est["standoff"] * (1 + jitter * rng.standard_normal())
        return (float(np.clip(eps_w, 2.5, 8.0)),
                float(np.clip(d_w, 0.08, 0.45)),
                float(np.clip(standoff, 0.2, 1.2)))
    return sampler
