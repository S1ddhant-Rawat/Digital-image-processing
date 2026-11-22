# WEIGHTED AVERAGE FILTER - smoothing
# Like the mean filter, but the centre pixel counts more than its neighbours.
import numpy as np
from hsi_utils import load_image, rgb_to_hsi, hsi_to_rgb, kernel_filter, add_salt_pepper, show

rgb = load_image()
h, s, i = rgb_to_hsi(rgb)
noisy = add_salt_pepper(i)

kernel = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]]) / 16    # centre weight 4, sides 2, corners 1 (total 16)
result = kernel_filter(noisy, kernel)

show([("Original", hsi_to_rgb(h, s, i)),
      ("Noisy", hsi_to_rgb(h, s, noisy)),
      ("Weighted average filter 3x3", hsi_to_rgb(h, s, result))])
