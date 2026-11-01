"""
B1: Real video dataset for DD-MADUN v3 training.

Loads DAVIS 2017 image sequences (or fallback to synthetic with Perlin-like
textures), extracts random 256x256x8 clips, and mixes with the original
synthetic complex-scene generator.

Usage:
    train_data, test_data = create_mixed_dataset(
        n_train=2000, n_test=20, T=8, H=256, W=256,
        real_ratio=0.7, davis_root="DAVIS", seed=42)
"""
import os
import glob
import numpy as np
import torch
from PIL import Image

# Reuse the synthetic generator for the fallback / mixing
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
from dataset import make_complex_scene


def _load_davis_sequences(davis_root):
    """Scan DAVIS/JPEGImages/480p/*/ and return a list of sorted frame paths
    per sequence. Returns [] if DAVIS not found."""
    img_dir = os.path.join(davis_root, "JPEGImages", "480p")
    if not os.path.isdir(img_dir):
        # Try without the 480p subdir
        img_dir = os.path.join(davis_root, "JPEGImages")
        if not os.path.isdir(img_dir):
            return []
    sequences = []
    for seq_name in sorted(os.listdir(img_dir)):
        seq_path = os.path.join(img_dir, seq_name)
        if not os.path.isdir(seq_path):
            continue
        frames = sorted(glob.glob(os.path.join(seq_path, "*.jpg")))
        if len(frames) >= 8:
            sequences.append(frames)
    return sequences


def _load_clip(frame_paths, T, H, W, start_idx=None):
    """Load T consecutive frames from frame_paths starting at start_idx,
    randomly crop to HxW, convert to grayscale [0,1].
    Returns: (T, H, W) float32 numpy array."""
    n_frames = len(frame_paths)
    if start_idx is None:
        start_idx = np.random.randint(0, n_frames - T + 1)
    else:
        start_idx = min(start_idx, n_frames - T)

    frames = []
    for i in range(T):
        img = Image.open(frame_paths[start_idx + i]).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        frames.append(arr)

    # Stack to (T, H0, W0)
    clip = np.stack(frames, axis=0)
    H0, W0 = clip.shape[1], clip.shape[2]

    # Random crop to (H, W)
    if H0 > H:
        y0 = np.random.randint(0, H0 - H + 1)
    else:
        y0 = 0
    if W0 > W:
        x0 = np.random.randint(0, W0 - W + 1)
    else:
        x0 = 0
    clip = clip[:, y0:y0+H, x0:x0+W]

    # If smaller than target, pad with reflection
    if clip.shape[1] < H or clip.shape[2] < W:
        pad_h = max(0, H - clip.shape[1])
        pad_w = max(0, W - clip.shape[2])
        clip = np.pad(clip, ((0,0),(0,pad_h),(0,pad_w)), mode="reflect")

    return np.clip(clip, 0, 1)


def _augment_clip(clip):
    """Random augmentation: flip H, flip W, 90-degree rotation, brightness jitter."""
    T, H, W = clip.shape
    # Horizontal flip
    if np.random.rand() > 0.5:
        clip = clip[:, :, ::-1].copy()
    # Vertical flip
    if np.random.rand() > 0.5:
        clip = clip[:, ::-1, :].copy()
    # 90-degree rotation (random k=0,1,2,3)
    k = np.random.randint(0, 4)
    if k > 0:
        # 90/270-degree rotation would swap H and W; skip for non-square masks
        if k in (1, 3) and H != W:
            k = 0
        if k > 0:
            clip = np.rot90(clip, k=k, axes=(1, 2)).copy()
    # Brightness/contrast jitter
    if np.random.rand() > 0.5:
        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.uniform(-0.05, 0.05)
        clip = np.clip(clip * alpha + beta, 0, 1)
    return clip.astype(np.float32)


def generate_perlin_texture(H, W, n_octaves=4):
    """Generate a Perlin-like noise texture using random Gaussian filters.
    Not true Perlin noise but produces natural-looking texture patterns."""
    import scipy.ndimage as ndi
    texture = np.zeros((H, W), dtype=np.float32)
    for octave in range(n_octaves):
        scale = 2 ** (octave + 2)
        small = np.random.randn(max(2, H // scale), max(2, W // scale)).astype(np.float32)
        zoom_h = H / small.shape[0]
        zoom_w = W / small.shape[1]
        layer = ndi.zoom(small, (zoom_h, zoom_w), order=3)
        texture += layer * (0.5 ** octave)
    # Normalize to [0, 0.3] range for background texture
    t = texture - texture.min()
    if t.max() > 1e-8:
        t = t / t.max()
    return t * 0.3


def make_realistic_synthetic(B, T, H, W, seed=None, n_objects=7):
    """Enhanced synthetic scene with Perlin-like texture background.
    More realistic than plain Gaussian background in make_complex_scene."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    I = np.zeros((B, T, H, W), dtype=np.float32)
    yy, xx = np.ogrid[:H, :W]

    for b in range(B):
        # Perlin-like textured background (0.05~0.25)
        bg = generate_perlin_texture(H, W, n_octaves=rng.randint(3, 5))

        # Add a few moving textured objects
        objs = []
        for _ in range(n_objects):
            kind = rng.choice(["circle", "square", "gauss"])
            objs.append({
                "kind": kind,
                "cx": rng.uniform(W * 0.15, W * 0.85),
                "cy": rng.uniform(H * 0.15, H * 0.85),
                "r": rng.uniform(8, 28),
                "vx": rng.uniform(-12, 12),
                "vy": rng.uniform(-12, 12),
                "vr": rng.uniform(-12, 12),
                "ang": rng.uniform(0, 360),
                "int": rng.uniform(0.6, 1.0),
            })

        for t in range(T):
            frame = bg.copy()
            for o in objs:
                cx = int(np.clip(o["cx"] + o["vx"] * t, 0, W - 1))
                cy = int(np.clip(o["cy"] + o["vy"] * t, 0, H - 1))
                ang = np.radians(o["ang"] + o["vr"] * t)
                r = int(o["r"])
                if o["kind"] == "circle":
                    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
                    frame[mask] = np.maximum(frame[mask], o["int"])
                elif o["kind"] == "square":
                    x1, x2 = max(0, cx - r), min(W, cx + r)
                    y1, y2 = max(0, cy - r), min(H, cy + r)
                    frame[y1:y2, x1:x2] = np.maximum(frame[y1:y2, x1:x2], o["int"])
                else:
                    xr = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
                    yr = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
                    g = o["int"] * np.exp(-((xr / (r * 1.6)) ** 2 + (yr / (r * 0.7)) ** 2))
                    frame = np.maximum(frame, g.astype(np.float32))
            I[b, t] = frame
    return torch.from_numpy(np.clip(I, 0, 1))


def create_mixed_dataset(n_train=2000, n_test=20, T=8, H=256, W=256,
                         real_ratio=0.7, davis_root="DAVIS", seed=42,
                         n_objects=7):
    """Create a mixed dataset of real video clips and synthetic scenes.

    Args:
        real_ratio: fraction of samples from real video (rest from enhanced synthetic)
        davis_root: path to DAVIS dataset root (containing JPEGImages/480p/)
        seed: random seed

    Returns:
        train_data: (n_train, T, H, W) tensor
        test_data: (n_test, T, H, W) tensor
    """
    rng = np.random.RandomState(seed)
    sequences = _load_davis_sequences(davis_root)
    n_real_train = int(n_train * real_ratio)
    n_synth_train = n_train - n_real_train
    n_real_test = 0  # Keep test all synthetic for fair comparison with benchmark
    n_synth_test = n_test

    print(f"  [MixedDataset] DAVIS sequences: {len(sequences)}")
    print(f"  [MixedDataset] Train: {n_real_train} real + {n_synth_train} synth = {n_train}")
    print(f"  [MixedDataset] Test: {n_real_test} real + {n_synth_test} synth = {n_test}")

    # ---- Training data ----
    train_clips = []

    # Real video clips
    if n_real_train > 0 and len(sequences) > 0:
        for i in range(n_real_train):
            seq = sequences[rng.randint(0, len(sequences))]
            clip = _load_clip(seq, T, H, W)
            clip = _augment_clip(clip)
            train_clips.append(clip)
        print(f"  [MixedDataset] Loaded {n_real_train} real video clips")
    elif n_real_train > 0:
        print(f"  [MixedDataset] DAVIS not found, using enhanced synthetic instead")
        n_synth_train = n_train

    # Synthetic clips (enhanced with Perlin texture)
    if n_synth_train > 0:
        synth = make_realistic_synthetic(n_synth_train, T, H, W,
                                          seed=rng.randint(0, 10000),
                                          n_objects=n_objects)
        for i in range(n_synth_train):
            train_clips.append(synth[i].numpy())

    train_data = torch.from_numpy(np.stack(train_clips, axis=0))
    perm = torch.randperm(n_train)
    train_data = train_data[perm]

    # ---- Test data (all synthetic, same as original for fair comparison) ----
    test_seed = rng.randint(10000, 20000)
    test_data = make_complex_scene(n_synth_test, T, H, W, seed=test_seed,
                                    n_objects=n_objects)

    return train_data, test_data


if __name__ == "__main__":
    # Quick test
    print("Testing mixed dataset...")
    train, test = create_mixed_dataset(n_train=10, n_test=4, T=8, H=256, W=256,
                                        real_ratio=0.7, davis_root="DAVIS")
    print(f"Train: {train.shape}, range [{train.min():.3f}, {train.max():.3f}]")
    print(f"Test:  {test.shape}, range [{test.min():.3f}, {test.max():.3f}]")
    print("Done.")
