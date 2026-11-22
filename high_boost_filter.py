# HIGH-BOOST FILTERING - sharpening
# 1) blur the image   2) mask = original - blurred (the fine details)   3) result = original + k * mask
# k = 1 is "unsharp masking". k > 1 is high-boost: the details are boosted more strongly.
import numpy as np
from hsi_utils import load_image, rgb_to_hsi, hsi_to_rgb, kernel_filter, edge_view, show

K = 3                                  # boost strength (try 1, 2, 3, 5)

rgb = load_image()
h, s, i = rgb_to_hsi(rgb)

ax = np.arange(5) - 2                                      # blur with a 5x5 Gaussian
x, y = np.meshgrid(ax, ax)
blur_kernel = np.exp(-(x ** 2 + y ** 2) / 2)
blur_kernel /= blur_kernel.sum()
blurred = kernel_filter(i, blur_kernel)

mask = i - blurred                     # what the blur removed
boosted = i + K * mask                 # add the details back, k times

show([("Original", hsi_to_rgb(h, s, i)),
      ("Blurred", hsi_to_rgb(h, s, blurred)),
      ("Mask (details)", edge_view(mask), "gray"),
      (f"High-boost, k = {K}", hsi_to_rgb(h, s, boosted))])
