# MEAN (AVERAGE) FILTER - smoothing
# Each pixel becomes the plain average of its 3x3 neighbourhood. Every neighbour counts equally.
import numpy as np
from hsi_utils import load_image, rgb_to_hsi, hsi_to_rgb, kernel_filter, add_salt_pepper, show

rgb = load_image()
h, s, i = rgb_to_hsi(rgb)              # split into Hue, Saturation, Intensity
noisy = add_salt_pepper(i)             # add noise to the intensity so smoothing has something to remove

kernel = np.ones((3, 3)) / 9           # nine equal weights that add up to 1
result = kernel_filter(noisy, kernel)  # slide the kernel over the intensity channel

show([("Original", hsi_to_rgb(h, s, i)),
      ("Noisy", hsi_to_rgb(h, s, noisy)),
      ("Mean filter 3x3", hsi_to_rgb(h, s, result))])
