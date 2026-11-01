"""
Synthetic dataset for CUP reconstruction training and testing.
Generates diverse 3D (x,y,t) data cubes with known ground truth.
"""

import torch
import numpy as np
from scipy.ndimage import rotate as nd_rotate
from scipy.ndimage import shift as nd_shift


def generate_moving_shape(B, T, H, W, shape_type='random', seed=None):
    """
    Generate a batch of moving shapes.

    Args:
        B: batch size
        T: number of frames
        H, W: spatial dimensions
        shape_type: 'random', 'circle', 'square', 'line'
        seed: random seed

    Returns:
        I: tensor (B, T, H, W) with values in [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)

    I = np.zeros((B, T, H, W), dtype=np.float32)

    for b in range(B):
        # Random shape type
        if shape_type == 'random':
            st = np.random.choice(['circle', 'square', 'line', 'gaussian'])
        else:
            st = shape_type

        # Random motion parameters
        start_x = np.random.randint(W // 4, 3 * W // 4)
        start_y = np.random.randint(H // 4, 3 * H // 4)
        vel_x = np.random.uniform(-3, 3)
        vel_y = np.random.uniform(-3, 3)
        radius = np.random.randint(3, min(H, W) // 6)
        intensity = np.random.uniform(0.5, 1.0)

        for t in range(T):
            cx = int(start_x + vel_x * t)
            cy = int(start_y + vel_y * t)
            cx = np.clip(cx, radius, W - radius - 1)
            cy = np.clip(cy, radius, H - radius - 1)

            frame = np.zeros((H, W), dtype=np.float32)

            if st == 'circle':
                yy, xx = np.ogrid[:H, :W]
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
                frame[mask] = intensity

            elif st == 'square':
                x1 = max(0, cx - radius)
                x2 = min(W, cx + radius)
                y1 = max(0, cy - radius)
                y2 = min(H, cy + radius)
                frame[y1:y2, x1:x2] = intensity

            elif st == 'line':
                # Rotating line
                angle = t * 15 + np.random.uniform(0, 360)
                line_radius = min(radius * 2, min(H, W) // 3)
                for r in range(line_radius):
                    lx = int(cx + r * np.cos(np.radians(angle)))
                    ly = int(cy + r * np.sin(np.radians(angle)))
                    if 0 <= lx < W and 0 <= ly < H:
                        frame[ly, lx] = intensity

            elif st == 'gaussian':
                yy, xx = np.ogrid[:H, :W]
                sigma = radius / 2.0
                frame = intensity * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

            I[b, t] = frame

        # Add subtle background variation
        I[b] += np.random.uniform(0, 0.05, size=(T, H, W)).astype(np.float32)

    return torch.from_numpy(np.clip(I, 0, 1))


def generate_shepp_logan_motion(B, T, H, W, seed=None):
    """
    Generate simplified Shepp-Logan phantom with translational/rotational motion.

    Uses elliptical superposition to create phantom-like patterns.
    """
    if seed is not None:
        np.random.seed(seed)

    I = np.zeros((B, T, H, W), dtype=np.float32)

    for b in range(B):
        n_ellipses = np.random.randint(3, 8)
        ellipses = []
        for _ in range(n_ellipses):
            ellipses.append({
                'cx': np.random.uniform(0.2, 0.8) * W,
                'cy': np.random.uniform(0.2, 0.8) * H,
                'a': np.random.uniform(0.05, 0.2) * W,
                'b': np.random.uniform(0.05, 0.2) * H,
                'angle': np.random.uniform(0, np.pi),
                'intensity': np.random.uniform(0.3, 1.0),
                'vx': np.random.uniform(-1.5, 1.5),
                'vy': np.random.uniform(-1.5, 1.5),
                'vr': np.random.uniform(-3, 3),  # rotation velocity
            })

        for t in range(T):
            frame = np.zeros((H, W), dtype=np.float32)
            for e in ellipses:
                cx = e['cx'] + e['vx'] * t
                cy = e['cy'] + e['vy'] * t
                angle = e['angle'] + np.radians(e['vr']) * t

                yy, xx = np.ogrid[:H, :W]
                xr = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
                yr = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
                mask = (xr / e['a']) ** 2 + (yr / e['b']) ** 2 <= 1.0
                frame[mask] = np.maximum(frame[mask], e['intensity'])

            I[b, t] = frame

    return torch.from_numpy(np.clip(I, 0, 1))


def generate_fringe_patterns(B, T, H, W, seed=None):
    """
    Generate moving fringe/interference patterns.
    Simulates wave-like phenomena often captured by CUP.
    """
    if seed is not None:
        np.random.seed(seed)

    I = np.zeros((B, T, H, W), dtype=np.float32)
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')

    for b in range(B):
        freq = np.random.uniform(2, 8)
        phase_shift = np.random.uniform(0, 2 * np.pi)
        direction = np.random.uniform(0, 2 * np.pi)

        for t in range(T):
            phase = phase_shift + t * 0.2 * np.random.uniform(0.5, 2.0)
            rotated = xx * np.cos(direction + t * 0.05) + yy * np.sin(direction + t * 0.05)
            fringe = 0.5 + 0.5 * np.cos(freq * np.pi * rotated + phase)

            # Add envelope
            envelope = np.exp(-(xx ** 2 + yy ** 2) / 1.5)
            frame = fringe * envelope * np.random.uniform(0.7, 1.0)
            I[b, t] = frame.T  # Transpose because meshgrid is (ij)

    return torch.from_numpy(np.clip(I, 0, 1))


def create_dataset(n_train=500, n_test=20, T=8, H=64, W=64, seed=42):
    """
    Create a complete training and test dataset.

    Mixes multiple scene types for diversity:
      - Moving shapes (40%)
      - Shepp-Logan phantom motion (30%)
      - Fringe patterns (30%)

    Returns:
        train_data: (n_train, T, H, W)
        test_data: (n_test, T, H, W)
    """
    rng = np.random.RandomState(seed)

    # Training data
    n_per_type_train = [n_train * 4 // 10, n_train * 3 // 10, n_train * 3 // 10]
    n_per_type_train = [max(1, n) for n in n_per_type_train]
    # Adjust to match n_train
    n_per_type_train[0] += n_train - sum(n_per_type_train)

    train_shapes = generate_moving_shape(n_per_type_train[0], T, H, W, 'random', rng.randint(0, 10000))
    train_phantoms = generate_shepp_logan_motion(n_per_type_train[1], T, H, W, rng.randint(0, 10000))
    train_fringes = generate_fringe_patterns(n_per_type_train[2], T, H, W, rng.randint(0, 10000))

    train_data = torch.cat([train_shapes, train_phantoms, train_fringes], dim=0)
    # Shuffle
    perm = torch.randperm(n_train)
    train_data = train_data[perm]

    # Test data (separate seed)
    test_seed = rng.randint(10000, 20000)
    n_per_type_test = [n_test * 4 // 10, n_test * 3 // 10, n_test * 3 // 10]
    n_per_type_test = [max(1, n) for n in n_per_type_test]
    n_per_type_test[0] += n_test - sum(n_per_type_test)

    test_shapes = generate_moving_shape(n_per_type_test[0], T, H, W, 'random', test_seed)
    test_phantoms = generate_shepp_logan_motion(n_per_type_test[1], T, H, W, test_seed + 1)
    test_fringes = generate_fringe_patterns(n_per_type_test[2], T, H, W, test_seed + 2)

    test_data = torch.cat([test_shapes, test_phantoms, test_fringes], dim=0)

    return train_data, test_data


def make_complex_scene(B, T, H, W, seed=None, n_objects=7):
    """Complex scene: multiple objects with independent translation + rotation,
    low-frequency background texture, and inter-object overlap. Harder than the
    single-shape / phantom / fringe mixtures in create_dataset().
    Returns: tensor (B, T, H, W) with values in [0, 1]."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    I = np.zeros((B, T, H, W), dtype=np.float32)
    yy, xx = np.ogrid[:H, :W]   # precompute once per scene (not per frame!)
    for b in range(B):
        # Low-frequency background texture (a few large gaussians), 0.08~0.22
        bg = np.zeros((H, W), dtype=np.float32)
        for _ in range(4):
            cy, cx = rng.uniform(0, H), rng.uniform(0, W)
            sy, sx = rng.uniform(H / 6, H / 2), rng.uniform(W / 6, W / 2)
            bg += rng.uniform(0.04, 0.12) * np.exp(
                -(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        bg = np.clip(bg, 0, 0.22)

        objs = []
        for _ in range(n_objects):
            objs.append({
                'kind': rng.choice(['circle', 'square', 'gauss']),
                'cx': rng.uniform(W * 0.15, W * 0.85),
                'cy': rng.uniform(H * 0.15, H * 0.85),
                'r': rng.uniform(8, 28),
                'vx': rng.uniform(-12, 12),
                'vy': rng.uniform(-12, 12),
                'vr': rng.uniform(-12, 12),    # rotation speed (deg/frame)
                'ang': rng.uniform(0, 360),
                'int': rng.uniform(0.6, 1.0),
            })

        for t in range(T):
            frame = bg.copy()
            for o in objs:
                cx = int(np.clip(o['cx'] + o['vx'] * t, 0, W - 1))
                cy = int(np.clip(o['cy'] + o['vy'] * t, 0, H - 1))
                ang = np.radians(o['ang'] + o['vr'] * t)
                r = int(o['r'])
                if o['kind'] == 'circle':
                    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
                    frame[mask] = np.maximum(frame[mask], o['int'])
                elif o['kind'] == 'square':
                    x1, x2 = max(0, cx - r), min(W, cx + r)
                    y1, y2 = max(0, cy - r), min(H, cy + r)
                    frame[y1:y2, x1:x2] = np.maximum(frame[y1:y2, x1:x2], o['int'])
                else:  # rotating elliptical gaussian
                    xr = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
                    yr = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
                    g = o['int'] * np.exp(-((xr / (r * 1.6)) ** 2 + (yr / (r * 0.7)) ** 2))
                    frame = np.maximum(frame, g.astype(np.float32))
            I[b, t] = frame
    return torch.from_numpy(np.clip(I, 0, 1))


def create_complex_dataset(n_train=1000, n_test=20, T=8, H=256, W=256,
                           seed=42, n_objects=7):
    """All-complex-scene dataset (train + test), so evaluation is fair for the
    harder distribution. Train and test use disjoint seeds."""
    rng = np.random.RandomState(seed)
    train_seed = rng.randint(0, 10000)
    test_seed = rng.randint(10000, 20000)
    train_data = make_complex_scene(n_train, T, H, W, seed=train_seed, n_objects=n_objects)
    test_data = make_complex_scene(n_test, T, H, W, seed=test_seed, n_objects=n_objects)
    return train_data, test_data


if __name__ == '__main__':
    train, test = create_dataset(n_train=20, n_test=5, T=8, H=64, W=64)
    print(f"Train: {train.shape}, Test: {test.shape}")
    print(f"Train range: [{train.min():.3f}, {train.max():.3f}]")
    print(f"Test range:  [{test.min():.3f}, {test.max():.3f}]")
