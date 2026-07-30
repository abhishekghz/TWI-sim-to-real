"""
synth_twi.py
============
Physics-based synthetic Through-Wall Imaging (TWI) data generator.

This is the SOURCE domain for the sim-to-real study. Unlike the idealized
"white square on Gaussian noise" used in the baseline paper (Yadav et al., 2026),
this generator reproduces the stepped-frequency S21 measurement physics so that
the synthetic range images share the same modality and clutter structure as the
real VNA data:

  * Stepped-frequency forward model over the SAME band (1.5-3.5 GHz, 201 pts).
  * Front-wall response: a strong, near-range, frequency-dependent reflection
    (relative permittivity eps_w, thickness d_w), incl. internal reverberation.
  * Targets behind the wall: point/extended scatterers with material-dependent
    complex reflectivity (metal / Teflon / wood), correct two-way path delay
    through the wall (slower propagation inside the slab), and 1/R^2 spreading.
  * Multipath: wall-target-wall secondary bounce (attenuated, delayed).
  * Measurement noise: additive complex Gaussian on S21 (VNA noise floor).

The physics knobs (eps_w, d_w, standoff, target positions/materials) are exposed
so the generator can be CONDITIONED on wall parameters estimated from the real
data -> this realises the "physics-guided" half of the proposed method.

NOTE: This forward model is intentionally lightweight and fully reproducible.
The identical interface (generate_scene -> S21 (Npos,Nf)) is what an FDTD
solver (gprMax) would expose, so a higher-fidelity gprMax source can be swapped
in without changing any downstream code.
"""

import numpy as np

C = 299_792_458.0

# Material complex reflectivity priors (magnitude, loss) at microwave band.
# Metal ~ perfect reflector; Teflon low-eps low-loss; wood lossy dielectric.
MATERIALS = {
    "metal":  {"refl": 0.95, "loss": 0.02, "eps": 1.0},
    "teflon": {"refl": 0.35, "loss": 0.05, "eps": 2.1},
    "wood":   {"refl": 0.45, "loss": 0.15, "eps": 2.5},
}
MAT_LIST = ["metal", "teflon", "wood"]


def _wall_response(freq, eps_w, d_w, sigma_w=0.02, n_revb=3):
    """
    Complex frequency response of a homogeneous wall slab (front + internal
    reverberations). standoff handled separately.
    """
    k0 = 2 * np.pi * freq / C
    n = np.sqrt(eps_w)
    # Fresnel reflection (normal incidence, air->wall)
    r = (1 - n) / (1 + n)
    t = 1 - r**2
    # propagation phase/atten inside slab (two-way per reverberation)
    prop = np.exp(-1j * k0 * n * 2 * d_w) * np.exp(-sigma_w * 2 * d_w * freq / 1e9)
    H = np.zeros_like(freq, dtype=complex)
    H += r  # front-face reflection
    rev = t * r
    for _ in range(n_revb):
        H += rev * prop
        rev = rev * (r**2) * prop
    return H


def _target_response(freq, R_oneway, refl, loss, eps_w, d_w):
    """
    Complex response of a single target behind the wall at one-way range R_oneway
    (antenna->target straight-line distance, including the wall slab on the path).
    """
    k0 = 2 * np.pi * freq / C
    n = np.sqrt(eps_w)
    # extra electrical path from slower propagation inside the wall (two-way)
    extra = (n - 1) * 2 * d_w
    phase = np.exp(-1j * k0 * (2 * R_oneway + extra))
    spread = 1.0 / (R_oneway**2 + 1e-3)            # 1/R^2 two-way amplitude
    atten = np.exp(-loss * freq / 1e9)             # frequency-dependent material loss
    twall = (1 - ((1 - n) / (1 + n))**2)           # two-way wall transmission
    return refl * spread * atten * (twall**2) * phase


def generate_scene(freq, n_pos=28, array_len=0.54,
                   eps_w=4.5, d_w=0.20, standoff=0.50,
                   targets=None, n_targets=None,
                   noise_db=-60.0, multipath=True, rng=None):
    """
    Generate one synthetic TWI capture.

    Parameters
    ----------
    freq      : (Nf,) frequency axis (Hz) -- use the SAME axis as the real data.
    n_pos     : number of scan positions (28 to match the real array).
    array_len : physical aperture length (m); positions span [-L/2, L/2] cross-range.
    eps_w,d_w : wall permittivity / thickness (physics-guided: set from real data).
    standoff  : antenna-to-wall distance (m).
    targets   : list of dicts {x, y, material}; if None, sampled randomly.
    n_targets : if targets is None, how many to sample (default 1 or 2).

    Returns
    -------
    S21    : (n_pos, Nf) complex
    meta   : dict with ground-truth labels (target positions, materials, count, wall)
    """
    if rng is None:
        rng = np.random.default_rng()
    Nf = len(freq)
    xpos = np.linspace(-array_len / 2, array_len / 2, n_pos)  # cross-range positions

    # ---- sample a scene if not given -------------------------------------
    if targets is None:
        if n_targets is None:
            n_targets = int(rng.integers(1, 3))  # 1 or 2 targets
        targets = []
        for _ in range(n_targets):
            tx = rng.uniform(-0.20, 0.20)                 # cross-range (m)
            ty = rng.uniform(standoff + d_w + 0.3, 4.0)   # downrange behind wall (m)
            mat = MAT_LIST[int(rng.integers(0, len(MAT_LIST)))]
            targets.append({"x": float(tx), "y": float(ty), "material": mat})
    n_targets = len(targets)

    # ---- forward model per scan position ---------------------------------
    S21 = np.zeros((n_pos, Nf), dtype=complex)
    Hwall = _wall_response(freq, eps_w, d_w)
    for i, xa in enumerate(xpos):
        # wall reverberation (slightly position-jittered to mimic roughness)
        jitter = 1.0 + 0.05 * rng.standard_normal()
        S = Hwall * jitter * 0.5
        for tg in targets:
            m = MATERIALS[tg["material"]]
            R = np.hypot(tg["x"] - xa, tg["y"] + standoff)   # antenna->target
            S += _target_response(freq, R, m["refl"], m["loss"], eps_w, d_w)
            if multipath:
                # wall<->target secondary bounce: longer path, extra wall refl
                Rm = R + 2 * (standoff)
                rwall = ((1 - np.sqrt(eps_w)) / (1 + np.sqrt(eps_w)))**2
                S += 0.4 * rwall * _target_response(
                    freq, Rm, m["refl"], m["loss"], eps_w, d_w)
        S21[i] = S

    # ---- additive VNA noise ----------------------------------------------
    sig_pow = np.mean(np.abs(S21)**2)
    npow = sig_pow * 10**(noise_db / 10.0)
    noise = np.sqrt(npow / 2) * (rng.standard_normal(S21.shape)
                                 + 1j * rng.standard_normal(S21.shape))
    S21 = S21 + noise

    meta = {
        "targets": targets,
        "n_targets": n_targets,
        "eps_w": eps_w, "d_w": d_w, "standoff": standoff,
        "materials": [t["material"] for t in targets],
        "xpos": xpos,
    }
    return S21, meta


def make_synthetic_set(freq, n_scenes, imager, label_mode="count",
                       wall_dist=None, seed=0, **scene_kwargs):
    """
    Generate a labelled synthetic dataset of range images.

    label_mode:
        "count"    -> label = number of targets (0,1,2)  [detection task]
        "material" -> label = material of the strongest target (0/1/2)
    wall_dist: optional callable -> (eps_w, d_w, standoff) for physics-guided
               conditioning (e.g. sample around values estimated from real data).
    imager: function S21,freq -> (img, range)  [use data_io.to_range_image wrapper]
    """
    rng = np.random.default_rng(seed)
    X, y, metas = [], [], []
    for s in range(n_scenes):
        if wall_dist is not None:
            eps_w, d_w, standoff = wall_dist(rng)
            scene_kwargs.update(eps_w=eps_w, d_w=d_w, standoff=standoff)
        S21, meta = generate_scene(freq, rng=rng, **scene_kwargs)
        img, rngax = imager(S21, freq)
        X.append(img)
        if label_mode == "count":
            y.append(meta["n_targets"])
        else:
            # strongest target material (first listed as proxy)
            y.append(MAT_LIST.index(meta["materials"][0]) if meta["materials"] else 0)
        metas.append(meta)
    return np.stack(X), np.array(y), metas
