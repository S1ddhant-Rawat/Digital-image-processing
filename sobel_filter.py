# SOBEL FILTER - sharpening / edge detection
# A first-derivative filter. One kernel finds vertical edges (changes in x), one finds horizontal
# edges (changes in y). Combining them gives the edge strength at every pixel.
import numpy as np
from hsi_utils import load_image, rgb_to_hsi, hsi_to_rgb, kernel_filter, edge_view, show

rgb = load_image()
h, s, i = rgb_to_hsi(rgb)

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])       # change from left to right
sobel_y = sobel_x.T                    # change from top to bottom

gx = kernel_filter(i, sobel_x)
gy = kernel_filter(i, sobel_y)
magnitude = np.sqrt(gx ** 2 + gy ** 2)  # edge strength
sharpened = i + magnitude               # optional: add the edges back to the original

show([("Original", hsi_to_rgb(h, s, i)),
      ("Sobel Gx (vertical edges)", edge_view(gx), "gray"),
      ("Sobel Gy (horizontal edges)", edge_view(gy), "gray"),
      ("Sobel magnitude (edges)", edge_view(magnitude), "gray"),
      ("Original + Sobel edges", hsi_to_rgb(h, s, sharpened))])
