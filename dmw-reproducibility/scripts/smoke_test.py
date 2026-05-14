"""Small synthetic smoke test for the DMW experiment code.

This script avoids dataset downloads and full benchmark runs.  It checks that
the core DMW utilities import correctly and produce finite values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from dmw_core import (  # noqa: E402
    circle_points,
    ellipse_points,
    multiscale_sdmw,
    pairwise_distances_point_cloud,
    sample_distance_vectors,
    sliced_dmw,
)


def main() -> None:
    rng = np.random.default_rng(7)
    x = circle_points(80, radius=1.0, rng=rng, noise=0.01)
    y = ellipse_points(80, axes=(1.0, 0.75), rng=rng, noise=0.01)
    dx = pairwise_distances_point_cloud(x)
    dy = pairwise_distances_point_cloud(y)

    vx = sample_distance_vectors(dx, n=4, k=32, rng=rng)
    vy = sample_distance_vectors(dy, n=4, k=32, rng=rng)
    sdmw = sliced_dmw(vx, vy, l=16, rng=rng, p=1)
    ms = multiscale_sdmw(dx, dy, orders=[2, 3, 4], weights=[1 / 3] * 3, k=24, l=8, rng=rng, p=1)

    if not np.isfinite(sdmw) or not np.isfinite(ms):
        raise RuntimeError("Smoke test failed: non-finite DMW value.")

    print(f"smoke_sdmw={sdmw:.6f}")
    print(f"smoke_multiscale_sdmw={ms:.6f}")
    print("[ok] DMW smoke test completed.")


if __name__ == "__main__":
    main()

