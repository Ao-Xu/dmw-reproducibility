"""Core utilities for the DMW experiments.

The implementation is intentionally dependency-light: numpy/scipy/sklearn/
networkx/matplotlib are enough to reproduce the figures in the paper draft.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

try:
    import ot
    import ot.gromov
except Exception:  # pragma: no cover - optional dependency
    ot = None


def pairwise_distances_point_cloud(x: np.ndarray) -> np.ndarray:
    return cdist(x, x)


def sample_distance_vectors(
    dist: np.ndarray,
    n: int,
    k: int,
    rng: np.random.Generator,
    replace: bool = True,
) -> np.ndarray:
    """Sample K random n-point distance vectors from a finite metric."""
    m = dist.shape[0]
    pairs = np.triu_indices(n, 1)
    out = np.empty((k, n * (n - 1) // 2), dtype=float)
    for row in range(k):
        idx = rng.choice(m, size=n, replace=replace)
        sub = dist[np.ix_(idx, idx)]
        out[row] = sub[pairs]
    return out


def empirical_dmw(a: np.ndarray, b: np.ndarray, p: int = 1) -> float:
    """Exact empirical DMW for equal-size empirical laws via assignment."""
    if len(a) != len(b):
        raise ValueError("This helper expects equal empirical sample sizes.")
    if p == 1:
        cost = cdist(a, b, metric="cityblock") / a.shape[1]
    else:
        diff = np.abs(a[:, None, :] - b[None, :, :]) ** p
        cost = diff.mean(axis=2)
    rows, cols = linear_sum_assignment(cost)
    val = cost[rows, cols].mean()
    return float(val if p == 1 else val ** (1.0 / p))


def sliced_dmw(
    a: np.ndarray,
    b: np.ndarray,
    l: int,
    rng: np.random.Generator,
    p: int = 1,
) -> float:
    """Monte Carlo sliced DMW for equal empirical sample sizes."""
    dim = a.shape[1]
    vals = []
    for _ in range(l):
        theta = rng.normal(size=dim)
        norm = np.linalg.norm(theta)
        if norm == 0:
            theta[0] = 1.0
            norm = 1.0
        theta = theta / norm
        pa = np.sort(a @ theta)
        pb = np.sort(b @ theta)
        vals.append(np.mean(np.abs(pa - pb) ** p))
    return float(np.mean(vals) ** (1.0 / p))


def multiscale_sdmw(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    orders: Sequence[int],
    weights: Sequence[float],
    k: int,
    l: int,
    rng: np.random.Generator,
    p: int = 1,
) -> float:
    total = 0.0
    for n, w in zip(orders, weights):
        va = sample_distance_vectors(dist_a, n, k, rng)
        vb = sample_distance_vectors(dist_b, n, k, rng)
        total += w * sliced_dmw(va, vb, l, rng, p=p)
    return float(total)


def circle_points(num: int, radius: float, rng: np.random.Generator, noise: float = 0.0) -> np.ndarray:
    theta = rng.uniform(0, 2 * np.pi, size=num)
    pts = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    if noise > 0:
        pts += noise * rng.normal(size=pts.shape)
    return pts


def ellipse_points(num: int, axes: tuple[float, float], rng: np.random.Generator, noise: float = 0.0) -> np.ndarray:
    theta = rng.uniform(0, 2 * np.pi, size=num)
    pts = np.column_stack([axes[0] * np.cos(theta), axes[1] * np.sin(theta)])
    if noise > 0:
        pts += noise * rng.normal(size=pts.shape)
    return pts


def sphere_points(num: int, radius: float, rng: np.random.Generator, noise: float = 0.0) -> np.ndarray:
    z = rng.uniform(-1, 1, size=num)
    phi = rng.uniform(0, 2 * np.pi, size=num)
    r = np.sqrt(np.maximum(0, 1 - z**2))
    pts = radius * np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if noise > 0:
        pts += noise * rng.normal(size=pts.shape)
    return pts


def sbm_graph(n: int, p_in: float, p_out: float, rng: np.random.Generator) -> nx.Graph:
    sizes = [n // 2, n - n // 2]
    probs = [[p_in, p_out], [p_out, p_in]]
    seed = int(rng.integers(0, 2**31 - 1))
    g = nx.stochastic_block_model(sizes, probs, seed=seed)
    if not nx.is_connected(g):
        comps = [list(c) for c in nx.connected_components(g)]
        for c1, c2 in zip(comps[:-1], comps[1:]):
            g.add_edge(c1[0], c2[0])
    return g


def er_graph(n: int, p_edge: float, rng: np.random.Generator) -> nx.Graph:
    seed = int(rng.integers(0, 2**31 - 1))
    g = nx.erdos_renyi_graph(n, p_edge, seed=seed)
    if not nx.is_connected(g):
        comps = [list(c) for c in nx.connected_components(g)]
        for c1, c2 in zip(comps[:-1], comps[1:]):
            g.add_edge(c1[0], c2[0])
    return g


def graph_shortest_path_metric(g: nx.Graph) -> np.ndarray:
    nodes = list(g.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    d = np.zeros((len(nodes), len(nodes)), dtype=float)
    lengths = dict(nx.all_pairs_shortest_path_length(g))
    diameter = 0
    for u in nodes:
        for v, val in lengths[u].items():
            d[idx[u], idx[v]] = val
            diameter = max(diameter, val)
    d[d == 0] = 0
    return d / max(1.0, float(diameter))


def shortest_path_hist_features(g: nx.Graph, bins: int = 8) -> np.ndarray:
    d = graph_shortest_path_metric(g)
    vals = d[np.triu_indices_from(d, 1)]
    hist, _ = np.histogram(vals, bins=bins, range=(0, 1), density=False)
    return hist.astype(float) / max(1, hist.sum())


def degree_hist_features(g: nx.Graph, bins: int = 8) -> np.ndarray:
    deg = np.array([d for _, d in g.degree()], dtype=float)
    if deg.max() > 0:
        deg = deg / deg.max()
    hist, _ = np.histogram(deg, bins=bins, range=(0, 1), density=False)
    return hist.astype(float) / max(1, hist.sum())


def entropic_gw_proxy(
    dx: np.ndarray,
    dy: np.ndarray,
    iters: int = 25,
    epsilon: float = 0.05,
) -> float:
    """Entropic GW baseline, using POT when available."""
    m, n = dx.shape[0], dy.shape[0]
    px = np.ones(m) / m
    py = np.ones(n) / n
    if ot is not None:
        return float(
        ot.gromov.entropic_gromov_wasserstein2(
                dx,
                dy,
                px,
                py,
                loss_fun="square_loss",
                epsilon=epsilon,
                max_iter=iters,
                tol=1e-7,
                verbose=False,
            )
        )
    t = np.outer(px, py)
    loss = np.abs(dx[:, None, :, None] - dy[None, :, None, :])
    for _ in range(iters):
        grad = 2.0 * np.einsum("ikjl,kl->ij", loss, t)
        k_mat = np.exp(-grad / max(epsilon, 1e-6))
        u = np.ones(m)
        v = np.ones(n)
        for _ in range(60):
            u = px / np.maximum(k_mat @ v, 1e-300)
            v = py / np.maximum(k_mat.T @ u, 1e-300)
        t = (u[:, None] * k_mat) * v[None, :]
    obj = np.einsum("ikjl,ij,kl->", loss, t, t)
    return float(obj)


def exact_gw_pot(dx: np.ndarray, dy: np.ndarray, iters: int = 50) -> float:
    """POT conditional-gradient GW baseline for small uniform spaces."""
    if ot is None:
        return entropic_gw_proxy(dx, dy, iters=min(iters, 20), epsilon=0.05)
    m, n = dx.shape[0], dy.shape[0]
    px = np.ones(m) / m
    py = np.ones(n) / n
    return float(
        ot.gromov.gromov_wasserstein2(
            dx,
            dy,
            px,
            py,
            loss_fun="square_loss",
            numItermax=iters,
            verbose=False,
        )
    )


@dataclass
class TimerResult:
    value: float
    seconds: float


def timed(fn, *args, **kwargs) -> TimerResult:
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    return TimerResult(value=float(value), seconds=time.perf_counter() - start)
