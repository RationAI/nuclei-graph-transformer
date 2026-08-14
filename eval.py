import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import matplotlib

# Silently apply memory optimization for rendering tasks
matplotlib.rcParams.update({"figure.max_open_warning": 0})

# 1. Load the computed stats from both environments
mmci_data = np.load("mmci_stats.npz")
panda_data = np.load("panda_stats.npz")

mmci_circ = mmci_data["circularity"]
panda_circ = panda_data["circularity"]

# 2. Calculate Wasserstein distance
print("--- Polygon Detail Shift Metrics ---")
print(
    f"Circularity (Roughness) Wasserstein: {wasserstein_distance(mmci_circ, panda_circ):.4f}"
)

# 3. Maximum Resolution Plotting
fig, ax = plt.subplots(figsize=(8, 5))

# Plot distributions
ax.hist(
    mmci_circ,
    bins=50,
    alpha=0.5,
    label="MMCI (0.23 MPP - Smooth)",
    density=True,
    color="royalblue",
)
ax.hist(
    panda_circ,
    bins=50,
    alpha=0.5,
    label="PANDA (0.48 MPP - Jagged)",
    density=True,
    color="darkorange",
)

ax.set_title("Polygon Circularity Distribution (Surface Roughness)")
ax.set_xlabel("Circularity (1.0 = Perfect Circle, <0.5 = Highly Jagged)")
ax.legend()

plt.tight_layout()

# Force maximum resolution (1200 DPI) for clear visual inspection of the shape distributions
plt.savefig("polygon_detail_comparison.png", dpi=1200, bbox_inches="tight")
print(
    "\nHigh-resolution polygon detail plots saved successfully to 'polygon_detail_comparison.png'."
)
