"""Series colors shared by the Ni(OH)F notebooks.

n = 2 is bright pink and black.
n = 3 is violet, bright pink, and black.
Any other n is an even sample of the six-color palette
(orange, crimson, dark magenta, bright pink, violet, black).
"""

import numpy as np

PALETTE = ["#F4A000", "#DC143C", "#8B1874", "#FF3EB5", "#5E2B97", "#1A1A1A"]
BRIGHT_PINK = "#FF3EB5"


def series_colors(n, palette=None):
    """Return n colors. n = 2 and n = 3 are fixed; other n sample the palette."""
    colors = PALETTE if palette is None else list(palette)
    n = int(n)
    if n <= 0:
        return []
    if n == 1:
        return [colors[0]]
    if n == 2:
        return [BRIGHT_PINK, colors[-1]]
    if n == 3:
        return ["#5E2B97", BRIGHT_PINK, colors[-1]]
    idx = np.linspace(0, len(colors) - 1, n).round().astype(int)
    return [colors[i] for i in idx]
