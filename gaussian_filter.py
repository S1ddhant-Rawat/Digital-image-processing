# GAUSSIAN FILTER - smoothing
# Weights follow a bell curve: close neighbours count a lot, far ones count very little.
import numpy as np
from hsi_utils import load_image, rgb_to_hsi, hsi_to_rgb, kernel_filter, add_salt_pepper, show

SIZE, SIGMA = 5, 1.0                   # kernel size, and how wide the bell curve is

rgb = load_image()
h, s, i = rgb_to_hsi(rgb)
noisy = add_salt_pepper(i)

ax = np.arange(SIZE) - SIZE // 2                                   # -2, -1, 0, 1, 2
x, y = np.meshgrid(ax, ax)
kernel = np.exp(-(x ** 2 + y ** 2) / (2 * SIGMA ** 2))             # Gaussian formula
kernel /= kernel.sum()                                             # make the weights add up to 1
result = kernel_filter(noisy, kernel)

show([("Original", hsi_to_rgb(h, s, i)),
      ("Noisy", hsi_to_rgb(h, s, noisy)),
      (f"Gaussian filter {SIZE}x{SIZE}, sigma={SIGMA}", hsi_to_rgb(h, s, result))])
