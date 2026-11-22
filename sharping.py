import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

IMAGE_PATH = None        # e.g. r"C:\Users\LENOVOO\photo.jpg"  (None = use a built-in test image)
BOOST_K = 3              # high-boost strength (k = 1 is plain unsharp masking, k > 1 is high-boost)

# ---------- image ----------
def make_test_image(n=256):
    """Colour gradient with a circle and a square. Their edges are soft (a bit blurry) on purpose."""
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    sig = lambda u: 1 / (1 + np.exp(-u / 0.005))                       # soft step
    circle = sig(0.14 - np.hypot(x - .7, y - .3))
    square = sig(x - .16) * sig(.43 - x) * sig(y - .59) * sig(.82 - y)
    img = np.dstack([x, y, 1 - 0.7 * x])
    for mask, colour in ((circle, [1, .9, .2]), (square, [.1, .2, .6])):
        img = img * (1 - mask[..., None]) + np.array(colour) * mask[..., None]
    return img

def load_image():
    if IMAGE_PATH is None:
        return make_test_image()
    from PIL import Image
    im = Image.open(IMAGE_PATH).convert("RGB")
    im.thumbnail((256, 256))
    return np.asarray(im) / 255.0

# ---------- RGB <-> HSI ----------
def rgb_to_hsi(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    eps = 1e-8
    i = (r + g + b) / 3                                                # Intensity
    s = 1 - np.minimum(np.minimum(r, g), b) / (i + eps)                # Saturation
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + eps
    theta = np.arccos(np.clip(num / den, -1, 1))
    h = np.where(b <= g, theta, 2 * np.pi - theta)                     # Hue
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

# ---------- filtering helper ----------
def kernel_filter(img, kernel):                                        # weighted sum of each neighbourhood
    k = kernel.shape[0]
    return (sliding_window_view(np.pad(img, k // 2, mode="reflect"), (k, k)) * kernel).sum(axis=(-2, -1))

laplacian_k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])          # second derivative: finds edges
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])               # first derivative in x
sobel_y = sobel_x.T                                                    # first derivative in y
ax = np.arange(-2, 3)
gauss_k = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / 2)
gauss_k /= gauss_k.sum()                                               # blur, used by high-boost

def show_edges(x):                                                     # scale an edge map to 0..1 for display
    x = np.abs(x)
    return np.clip(x / (np.percentile(x, 99.5) + 1e-9), 0, 1)

# ---------- run: all filters work on the Intensity channel only ----------
rgb = load_image()
h, s, i = rgb_to_hsi(rgb)

lap = kernel_filter(i, laplacian_k)                                    # Laplacian
lap_sharp = i + lap                                                    # original + edges = sharper

sobel = np.hypot(kernel_filter(i, sobel_x), kernel_filter(i, sobel_y)) # Sobel: gradient magnitude

mask = i - kernel_filter(i, gauss_k)                                   # what the blur removed (the details)
unsharp = i + 1 * mask                                                 # unsharp masking
boost = i + BOOST_K * mask                                             # high-boost: add the details k times

panels = [("Original", hsi_to_rgb(h, s, i), None),
          ("Laplacian (edges only)", show_edges(lap), "gray"),
          ("Laplacian sharpened", hsi_to_rgb(h, s, lap_sharp), None),
          ("Sobel (edges only)", show_edges(sobel), "gray"),
          ("Unsharp mask (k = 1)", hsi_to_rgb(h, s, unsharp), None),
          (f"High-boost (k = {BOOST_K})", hsi_to_rgb(h, s, boost), None)]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for axis, (title, image, cmap) in zip(axes.flat, panels):
    axis.imshow(image, cmap=cmap, vmin=0, vmax=1)
    axis.set_title(title)
    axis.axis("off")
plt.tight_layout()
plt.show()