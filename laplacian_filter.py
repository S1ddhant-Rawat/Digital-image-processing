# LAPLACIAN FILTER - sharpening
# A second-derivative filter: it responds where the intensity changes quickly (edges).
# Adding the edges back to the original makes the image look sharper.
import numpy as np
from hsi_utils import load_image, rgb_to_hsi, hsi_to_rgb, kernel_filter, edge_view, show

rgb = load_image()
h, s, i = rgb_to_hsi(rgb)

kernel = np.array([[ 0, -1,  0],
                   [-1,  4, -1],
                   [ 0, -1,  0]])      # weights add up to 0, so flat areas give 0
edges = kernel_filter(i, kernel)       # Laplacian response (edges)
sharpened = i + edges                  # original + edges

show([("Original", hsi_to_rgb(h, s, i)),
      ("Laplacian (edges only)", edge_view(edges), "gray"),
      ("Laplacian sharpened", hsi_to_rgb(h, s, sharpened))])
