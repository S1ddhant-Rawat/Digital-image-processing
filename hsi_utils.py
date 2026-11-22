"""Shared helpers: load an image, convert RGB <-> HSI, slide a kernel over an image, show results.
Keep this file in the same folder as the filter files."""
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

IMAGE_PATH = None        # e.g. r"C:\Users\LENOVOO\photo.jpg"  (None = use the built-in test image)


def load_image():
    """Return an RGB image as floats from 0 to 1."""
    if IMAGE_PATH is None:                                       # built-in test image
        n = 256
        y, x = np.mgrid[0:n, 0:n] / (n - 1)
        sig = lambda u: 1 / (1 + np.exp(-u / 0.005))             # soft step = slightly soft edges
        circle = sig(0.14 - np.hypot(x - .7, y - .3))
        square = sig(x - .16) * sig(.43 - x) * sig(y - .59) * sig(.82 - y)
        img = np.dstack([x, y, 1 - 0.7 * x])
        for mask, colour in ((circle, [1, .9, .2]), (square, [.1, .2, .6])):
            img = img * (1 - mask[..., None]) + np.array(colour) * mask[..., None]
        return img
    from PIL import Image                                        # needs: pip install pillow
    im = Image.open(IMAGE_PATH).convert("RGB")
    im.thumbnail((256, 256))                                     # keep it small so filters stay quick
    return np.asarray(im) / 255.0


def rgb_to_hsi(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    eps = 1e-8
    i = (r + g + b) / 3                                          # Intensity
    s = 1 - np.minimum(np.minimum(r, g), b) / (i + eps)          # Saturation
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + eps
    theta = np.arccos(np.clip(num / den, -1, 1))
    h = np.where(b <= g, theta, 2 * np.pi - theta)               # Hue
    return h, s, i


def hsi_to_rgb(h, s, i):
    i = np.clip(i, 0, 1)
    f = lambda hh: i * (1 + s * np.cos(hh) / np.cos(np.pi / 3 - hh))
    a, b_ = 2 * np.pi / 3, 4 * np.pi / 3
    s1, s2 = h < a, (h >= a) & (h < b_)
    b1 = i * (1 - s); r1 = f(h); g1 = 3 * i - (r1 + b1)
    r2 = i * (1 - s); g2 = f(h - a); b2 = 3 * i - (r2 + g2)
    g3 = i * (1 - s); b3 = f(h - b_); r3 = 3 * i - (g3 + b3)
    r, g, b = (np.select([s1, s2], [x, y], z) for x, y, z in ((r1, r2, r3), (g1, g2, g3), (b1, b2, b3)))
    return np.clip(np.dstack([r, g, b]), 0, 1)


def windows(img, k):
    """Every k x k neighbourhood of the image (edges are padded by mirroring)."""
    return sliding_window_view(np.pad(img, k // 2, mode="reflect"), (k, k))


def kernel_filter(img, kernel):
    """Multiply each neighbourhood by the kernel and add up (this is the filtering step)."""
    return (windows(img, kernel.shape[0]) * kernel).sum(axis=(-2, -1))


def add_salt_pepper(i, amount=0.08, seed=0):
    """Add random black and white dots (noise) to the intensity channel."""
    noisy = i.copy()
    r = np.random.default_rng(seed).random(i.shape)
    noisy[r < amount / 2], noisy[r > 1 - amount / 2] = 0, 1
    return noisy


def edge_view(x):
    """Scale an edge map to 0..1 so it can be displayed."""
    x = np.abs(x)
    return np.clip(x / (np.percentile(x, 99.5) + 1e-9), 0, 1)


def show(panels):
    """panels = list of (title, image) or (title, image, cmap). Shows them side by side."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    for axis, p in zip(np.atleast_1d(axes), panels):
        axis.imshow(p[1], cmap=p[2] if len(p) > 2 else None, vmin=0, vmax=1)
        axis.set_title(p[0])
        axis.axis("off")
    plt.tight_layout()
    plt.show()
