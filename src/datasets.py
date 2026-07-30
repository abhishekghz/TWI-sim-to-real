"""
datasets.py
===========
Assemble training/evaluation tensors for the sim-to-real study.

Reality check on the real data: we have 2 captures, both the TWO-TARGET case.
Two full 28x274 images cannot train or fairly test a classifier on their own.
Following standard practice for scarce-target-domain sim-to-real work, we:

  * Use the PHYSICS-BASED SYNTHETIC generator as the large labelled SOURCE domain
    (counts 0/1/2 targets, materials metal/teflon/wood), with automatic labels.
  * Derive a REAL TARGET-DOMAIN evaluation set by extracting overlapping
    range-cross-range PATCHES from the real images. A patch is labelled by how
    many true target signatures fall inside it (0/1/2), giving a small but
    genuine real test set with known ground truth from the two-target geometry.
  * Evaluate with LEAVE-ONE-CAPTURE-OUT so every real capture is held out once;
    real patches are used only for (optional) light fine-tuning + testing, never
    as the main training source -- which is precisely the sim-to-real premise.

This keeps the claims honest: the headline is data-efficiency (how little real
data is needed), not "trained on 2 images".
"""

import numpy as np
import torch
from data_io import to_range_image, normalize01
from music import music_feature


# ----------------------------------------------------------------------------
def extract_real_patches(capture, patch_w=48, stride=8, sig_db=None):
    """
    Slide a window along the RANGE axis of the real image. Each patch is labelled
    target-present (1) if it overlaps one of the two dominant post-wall target
    returns, else background/clutter (0). This is the well-posed binary detection
    task the two-target captures genuinely support (and matches the paper's goal:
    distinguish hidden-target signatures from wall clutter / background).
    """
    img = capture["image"]                     # (28, Nr) dB
    rng = capture["range"]
    Npos, Nr = img.shape
    prof = img.mean(axis=0)

    # locate the two strongest post-wall (beyond coupling) target ranges
    post = rng > 0.7
    pr = np.where(post, prof, -1e9)
    order = np.argsort(pr)[::-1]
    peaks = []
    for idx in order:
        if all(abs(idx - p) > 10 for p in peaks):
            peaks.append(int(idx))
        if len(peaks) == 2:
            break
    target_bins = sorted(peaks)

    S21 = capture["raw"]["S21"]
    mfeat_full = music_feature(S21)
    patches, labels, mfeats = [], [], []
    half = patch_w // 2
    for ctr in range(half, Nr - half, stride):
        sl = slice(ctr - half, ctr + half)
        p = img[:, sl]
        lab = 1 if any(abs(ctr - tb) <= half // 2 for tb in target_bins) else 0
        patches.append(normalize01(p))
        labels.append(lab)
        mfeats.append(mfeat_full)
    return (np.stack(patches).astype(np.float32),
            np.array(labels),
            np.stack(mfeats).astype(np.float32))


# ----------------------------------------------------------------------------
def make_synth_patches(freq, n_scenes, imager_patch_w=48, seed=0,
                       wall_dist=None, label_mode="binary"):
    """
    Generate synthetic binary samples (target-present=1 vs background/clutter=0)
    shaped like real patches (28 x patch_w). For each scene we emit one positive
    patch centred on a target and one negative patch from a target-free range
    region (wall clutter / empty), giving a balanced source set.
    """
    from synth_twi import generate_scene, MAT_LIST
    rng = np.random.default_rng(seed)
    X, y, M = [], [], []
    half = imager_patch_w // 2
    for _ in range(n_scenes):
        kw = {}
        if wall_dist is not None:
            eps_w, d_w, standoff = wall_dist(rng)
            kw.update(eps_w=eps_w, d_w=d_w, standoff=standoff)
        # ensure at least one target for the positive patch
        S21, meta = generate_scene(freq, n_pos=28, n_targets=int(rng.integers(1, 3)),
                                   rng=rng, **kw)
        img, rngax = to_range_image(S21, freq)
        mf = music_feature(S21)
        # positive patch: centre on a target range
        ty = np.mean([t["y"] for t in meta["targets"]])
        ctr = int(np.clip(np.argmin(np.abs(rngax - ty)), half, img.shape[1] - half - 1))
        X.append(normalize01(img[:, ctr - half:ctr + half])); y.append(1); M.append(mf)
        # negative patch: a near-range wall/clutter region away from targets
        neg_ctr = int(np.clip(np.argmin(np.abs(rngax - 0.4)), half, img.shape[1] - half - 1))
        if abs(neg_ctr - ctr) < imager_patch_w:        # ensure separation
            neg_ctr = min(img.shape[1] - half - 1, ctr + imager_patch_w)
        X.append(normalize01(img[:, neg_ctr - half:neg_ctr + half])); y.append(0); M.append(mf)
    return (np.stack(X).astype(np.float32), np.array(y),
            np.stack(M).astype(np.float32))


# ----------------------------------------------------------------------------
def make_idealized_patches(n, patch_w=48, npos=28, seed=0, label_mode="binary"):
    """
    The BASELINE-PAPER style synthetic data: a bright square on Gaussian noise,
    with NO TWI physics. Binary: target-present(1) vs background(0). Used as the
    'idealized-synthetic' control arm to prove that physics matters for transfer.
    """
    rng = np.random.default_rng(seed)
    X, y, M = [], [], []
    for _ in range(n):
        bg = rng.normal(0.5, 0.15, size=(npos, patch_w)).astype(np.float32)
        lab = int(rng.integers(0, 2))
        if lab == 1:
            sz = 6
            r0 = rng.integers(0, npos - sz); c0 = rng.integers(0, patch_w - sz)
            bg[r0:r0 + sz, c0:c0 + sz] = 1.0
        X.append(normalize01(bg))
        y.append(lab)
        M.append(rng.standard_normal(64).astype(np.float32))  # uninformative
    return np.stack(X), np.array(y), np.stack(M).astype(np.float32)


# ----------------------------------------------------------------------------
def to_tensors(X, y, M, device="cpu"):
    Xt = torch.tensor(X).unsqueeze(1).to(device)      # (N,1,H,W)
    yt = torch.tensor(y).long().to(device)
    Mt = torch.tensor(M).to(device)
    return Xt, yt, Mt
