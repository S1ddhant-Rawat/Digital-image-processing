# MEDIAN FILTER - smoothing
# Each pixel becomes the MIDDLE value of its 3x3 neighbourhood (sort the 9 values, take the 5th).
# It is not an average, so it cannot be written as a kernel. Great against salt-and-pepper noise.
import numpy as np
from hsi_utils import load_image, rgb_to_hsi, hsi_to_rgb, windows, add_salt_pepper, show

rgb = load_image()
h, s, i = rgb_to_hsi(rgb)
noisy = add_salt_pepper(i)

result = np.median(windows(noisy, 3), axis=(-2, -1))   # median of every 3x3 neighbourhood

show([("Original", hsi_to_rgb(h, s, i)),
      ("Noisy", hsi_to_rgb(h, s, noisy)),
      ("Median filter 3x3", hsi_to_rgb(h, s, result))])
