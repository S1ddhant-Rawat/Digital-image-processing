import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

IMAGE_PATH = None        # e.g. r"C:\Users\LENOVOO\photo.jpg"  (None = use a built-in test image)

# ---------- image ----------
def make_test_image(n=256):
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    img = np.dstack([x, y, 1 - 0.7 * x])                       # colour gradient
    img[(x - .7) ** 2 + (y - .3) ** 2 < .02] = [1, .9, .2]     # yellow circle (sharp edge)
    img[150:210, 40:110] = [.1, .2, .6]                        # dark blue square (sharp edge)
    return img

def load_image():
    if IMAGE_PATH is None:
        return make_test_image()
    from PIL import Image
    im = Image.open(IMAGE_PATH).convert("RGB")
    im.thumbnail((256, 256))                                   # keep it small so the median is quick
    return np.asarray(im) / 255.0

# ---------- RGB <-> HSI ----------
def rgb_to_hsi(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    eps = 1e-8
    i = (r + g + b) / 3                                        # Intensity
    s = 1 - np.minimum(np.minimum(r, g), b) / (i + eps)        # Saturation
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + eps
    theta = np.arccos(np.clip(num / den, -1, 1))
    h = np.where(b <= g, theta, 2 * np.pi - theta)             # Hue (0 to 2*pi)
    return h, s, i

def hsi_to_rgb(h, s, i):
    i = np.clip(i, 0, 1)
    f = lambda hh: i * (1 + s * np.cos(hh) / np.cos(np.pi / 3 - hh))
    a, b_ = 2 * np.pi / 3, 4 * np.pi / 3
    s1, s2 = h < a, (h >= a) & (h < b_)                        # three 120-degree sectors
    b1 = i * (1 - s); r1 = f(h); g1 = 3 * i - (r1 + b1)                    # sector 1
    r2 = i * (1 - s); g2 = f(h - a); b2 = 3 * i - (r2 + g2)                # sector 2
    g3 = i * (1 - s); b3 = f(h - b_); r3 = 3 * i - (g3 + b3)               # sector 3
    r, g, b = (np.select([s1, s2], [x, y], z) for x, y, z in ((r1, r2, r3), (g1, g2, g3), (b1, b2, b3)))
    return np.clip(np.dstack([r, g, b]), 0, 1)

# ---------- smoothing filters (applied to the Intensity channel) ----------
def windows(img, k):                                           # every k x k neighbourhood
    return sliding_window_view(np.pad(img, k // 2, mode="reflect"), (k, k))

def kernel_filter(img, kernel):                                # weighted sum of each neighbourhood
    return (windows(img, kernel.shape[0]) * kernel).sum(axis=(-2, -1))

mean_k = np.ones((3, 3)) / 9                                   # mean: all weights equal
weighted_k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16  # weighted: centre counts more
ax = np.arange(-2, 3)
gauss_k = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / 2)   # Gaussian, sigma = 1
gauss_k /= gauss_k.sum()

def median_filter(img, k=3):                                   # median: middle value, not an average
    return np.median(windows(img, k), axis=(-2, -1))

def psnr(a, b):                                                # higher = closer to the original
    return 10 * np.log10(1 / (np.mean((a - b) ** 2) + 1e-12))

# ---------- run ----------
rgb = load_image()
h, s, i = rgb_to_hsi(rgb)                                      # split into H, S, I

rng = np.random.default_rng(0)                                 # add salt-and-pepper noise to I
noisy = i.copy()
r = rng.random(i.shape)
noisy[r < .04], noisy[r > .96] = 0, 1

results = [("Original", i), ("Noisy intensity", noisy),
           ("Mean 3x3", kernel_filter(noisy, mean_k)),
           ("Weighted average 3x3", kernel_filter(noisy, weighted_k)),
           ("Gaussian 5x5", kernel_filter(noisy, gauss_k)),
           ("Median 3x3", median_filter(noisy))]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for axis, (name, intensity) in zip(axes.flat, results):
    axis.imshow(hsi_to_rgb(h, s, intensity))                   # put H, S and the new I back together
    axis.set_title(name if name == "Original" else f"{name}   PSNR {psnr(i, intensity):.1f} dB")
    axis.axis("off")
plt.tight_layout()
plt.show()