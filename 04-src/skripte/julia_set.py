import numpy as np
import matplotlib as mpl

mpl.use("pgf")
import matplotlib.pyplot as plt

plt.style.use("style.mplstyle")
from matplotlib.colors import AsinhNorm


def julia(mesh, c, num_iter=1000, radius=2):
    """Compute the Julia set of z^2 + c for a given complex constant c.

    Parameters
    ----------
    mesh : np.ndarray
        A 2D array of complex numbers representing the grid of points in the complex plane.
    c : complex
        The complex constant used in the Julia set formula.
    num_iter : int, optional
        The maximum number of iterations to perform (default is 1000).
    radius : float, optional
        The escape radius; points with absolute value greater than this will be considered to have escaped (default is 2).
    """

    z = mesh.copy()
    diverge_len = np.full(z.shape, float(num_iter), dtype=float)
    not_escaped = np.ones(z.shape, dtype=bool)

    for i in range(num_iter):
        z[not_escaped] = z[not_escaped] ** 2 + c

        newly_escaped = not_escaped & (np.abs(z) >= radius)
        if newly_escaped.any():
            abs_z = np.abs(z[newly_escaped])

            # normalized iteration count for smooth coloring
            log_ratio = np.log2(np.maximum(np.log2(abs_z), 1e-10))
            diverge_len[newly_escaped] = np.maximum(i + 1.0 - log_ratio, 0.0)

            not_escaped[newly_escaped] = False

    return diverge_len


# from https://commons.wikimedia.org/wiki/File:Julia_set_%28red%29.png
c = -0.512511498387847167 + 0.521295573094847167j

x = np.linspace(-1.5, 1.5, 3543)
y = np.linspace(-1.0, 1.0, 2362)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

try:
    fractal = np.load("julia.npy")
except FileNotFoundError:
    fractal = julia(Z, c=c)
    np.save("julia.npy", fractal)

fig, ax = plt.subplots(
    figsize=(5.905511811, 3.937007874), constrained_layout=True
)  # 150 mm × 100 mm
ax.set_aspect("equal")

ax.set_xlabel(r"$\operatorname{Re}(z)$")
ax.set_ylabel(r"$\operatorname{Im}(z)$")

im = ax.imshow(
    fractal,
    extent=[-1.5, 1.5, -1, 1],
    origin="lower",
    cmap="viridis",
    norm=AsinhNorm(),
)

fig.savefig("julia.pgf", dpi=600)
plt.close()
