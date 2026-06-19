import locale

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("pgf")
plt.style.use("style.mplstyle")
locale.setlocale(locale.LC_NUMERIC, "sl_SI.UTF-8")

ticks = [-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi]
labels = [r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"]

STEP = 0.005
Y_RANGE = (-9.0, 9.0)
CENTER = 3 + 0j
RADIUS = 1.5

PANELS = [
    (-2.0, 6.0, 2),
    (-1.0, 3.0, 1),
    (-1.0, 7.0, 0),
]

fig, axes = plt.subplots(
    1, len(PANELS), figsize=(5.905511811, 3), constrained_layout=True
)

for ax, (x_min, x_max, n_iter) in zip(axes, PANELS):
    x = np.arange(x_min, x_max, STEP)
    y = np.arange(*Y_RANGE, STEP)
    X, Y = np.meshgrid(x, y)
    W = X + 1j * Y
    for _ in range(n_iter):
        W = np.exp(W)
    mask = np.abs(W - CENTER) < RADIUS

    ax.set_aspect("equal")
    ax.set_xlabel(r"$\operatorname{Re}(z)$")
    ax.set_ylabel(r"$\operatorname{Im}(z)$")
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))
    ax.contourf(X, Y, mask.astype(float), levels=[0.5, 1.5], cmap="gray")

plt.savefig("praslike.pgf")
plt.close(fig)
