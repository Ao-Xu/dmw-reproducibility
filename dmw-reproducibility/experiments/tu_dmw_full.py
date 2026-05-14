"""JMLR-style experiment suite for DMW.

This script is intentionally self-contained. It downloads TU datasets, parses
them into NetworkX graphs, builds reusable sliced-DMW features, runs nested-CV
classification, two-sample testing, scalability, and ablations.
"""

from __future__ import annotations

import itertools
import json
import ssl
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ot
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG = ROOT / "figures_full"
RES = ROOT / "results_full"
for p in [DATA, FIG, RES]:
    p.mkdir(exist_ok=True)


def rng(seed=0):
    return np.random.default_rng(seed)


def savefig(name):
    plt.savefig(FIG / name, bbox_inches="tight", dpi=220)
    plt.close()


def download_tu(name: str) -> Path:
    url = f"https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip"
    zpath = DATA / f"{name}.zip"
    outdir = DATA / name
    if outdir.exists():
        return outdir
    print(f"Downloading {name}...")
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=ctx, timeout=120) as r:
        zpath.write_bytes(r.read())
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(DATA)
    return outdir


def read_ints(path: Path):
    return [int(x.strip()) for x in path.read_text().splitlines() if x.strip()]


def load_tu(name: str, max_graphs: int | None = None):
    d = download_tu(name)
    graph_indicator = read_ints(d / f"{name}_graph_indicator.txt")
    labels = np.array(read_ints(d / f"{name}_graph_labels.txt"))
    n_graphs = max(graph_indicator)
    graphs = [nx.Graph() for _ in range(n_graphs)]
    node_to_graph = {}
    local_id = {}
    counters = [0] * n_graphs
    for global_idx, gid in enumerate(graph_indicator, start=1):
        gidx = gid - 1
        lid = counters[gidx]
        counters[gidx] += 1
        node_to_graph[global_idx] = gidx
        local_id[global_idx] = lid
        graphs[gidx].add_node(lid)
    with open(d / f"{name}_A.txt", "r") as f:
        for line in f:
            a, b = [int(x.strip()) for x in line.split(",")]
            ga, gb = node_to_graph[a], node_to_graph[b]
            if ga == gb:
                graphs[ga].add_edge(local_id[a], local_id[b])
    # Relabel labels to 0..c-1.
    uniq = {v: i for i, v in enumerate(sorted(set(labels)))}
    y = np.array([uniq[v] for v in labels])
    if max_graphs is not None and len(graphs) > max_graphs:
        # Stratified deterministic truncation for fast smoke tests.
        keep = []
        for cls in sorted(set(y)):
            idx = np.where(y == cls)[0]
            keep.extend(idx[: max_graphs // len(set(y))])
        keep = sorted(keep)
        graphs = [graphs[i] for i in keep]
        y = y[keep]
    return graphs, y


def graph_metric(g: nx.Graph) -> np.ndarray:
    if not nx.is_connected(g):
        comps = [list(c) for c in nx.connected_components(g)]
        for c1, c2 in zip(comps[:-1], comps[1:]):
            g = g.copy()
            g.add_edge(c1[0], c2[0])
    nodes = list(g.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    d = np.zeros((len(nodes), len(nodes)), dtype=float)
    lengths = dict(nx.all_pairs_shortest_path_length(g))
    for u, dd in lengths.items():
        for v, val in dd.items():
            d[idx[u], idx[v]] = val
    mx = d.max()
    return d / mx if mx > 0 else d


def sample_vectors(dist, n, k, gen):
    m = dist.shape[0]
    pairs = np.triu_indices(n, 1)
    out = np.empty((k, len(pairs[0])), dtype=np.float32)
    for i in range(k):
        ids = gen.choice(m, size=n, replace=True)
        out[i] = dist[np.ix_(ids, ids)][pairs]
    return out


def sdmw_features(metrics, orders, k=40, l=32, seed=0):
    gen = rng(seed)
    directions = {}
    for n in orders:
        dim = n * (n - 1) // 2
        th = gen.normal(size=(l, dim)).astype(np.float32)
        th /= np.maximum(np.linalg.norm(th, axis=1, keepdims=True), 1e-12)
        directions[n] = th
    feats = []
    for dist in metrics:
        parts = []
        for n in orders:
            vec = sample_vectors(dist, n, k, gen)
            proj = vec @ directions[n].T
            proj.sort(axis=0)
            parts.append(proj.T.reshape(-1))
        feats.append(np.concatenate(parts))
    return np.vstack(feats).astype(np.float32)


def l1_distance_matrix(feat):
    return cdist(feat, feat, metric="cityblock") / feat.shape[1]


def sp_hist(g, bins=12):
    d = graph_metric(g)
    vals = d[np.triu_indices_from(d, 1)]
    h, _ = np.histogram(vals, bins=bins, range=(0, 1))
    return h / max(1, h.sum())


def degree_hist(g, bins=12):
    vals = np.array([d for _, d in g.degree()], dtype=float)
    if vals.max() > 0:
        vals /= vals.max()
    h, _ = np.histogram(vals, bins=bins, range=(0, 1))
    return h / max(1, h.sum())


def wl_features(graphs, h=4):
    """Simple WL subtree count features."""
    vocab = {}
    rows = []
    labels = [{u: str(g.degree(u)) for u in g.nodes()} for g in graphs]
    for it in range(h + 1):
        row_dicts = []
        for gi, g in enumerate(graphs):
            counts = {}
            for u in g.nodes():
                lab = f"{it}:{labels[gi][u]}"
                if lab not in vocab:
                    vocab[lab] = len(vocab)
                counts[vocab[lab]] = counts.get(vocab[lab], 0) + 1
            row_dicts.append(counts)
        rows.append(row_dicts)
        if it < h:
            new_labels = []
            mapper = {}
            for gi, g in enumerate(graphs):
                nl = {}
                for u in g.nodes():
                    sig = (labels[gi][u], tuple(sorted(labels[gi][v] for v in g.neighbors(u))))
                    if sig not in mapper:
                        mapper[sig] = str(len(mapper))
                    nl[u] = mapper[sig]
                new_labels.append(nl)
            labels = new_labels
    x = np.zeros((len(graphs), len(vocab)), dtype=float)
    for row_dicts in rows:
        for i, counts in enumerate(row_dicts):
            for j, c in counts.items():
                x[i, j] += c
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x


def graphlet_features(graphs):
    """Counts of small motifs: edges, wedges, triangles, squares proxy."""
    x = []
    for g in graphs:
        deg = np.array([d for _, d in g.degree()], dtype=float)
        edges = g.number_of_edges()
        wedges = np.sum(deg * (deg - 1) / 2)
        triangles = sum(nx.triangles(g).values()) / 3
        clustering = nx.average_clustering(g) if g.number_of_nodes() > 2 else 0
        x.append([edges, wedges, triangles, clustering, deg.mean(), deg.std()])
    x = np.asarray(x, dtype=float)
    return (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-12)


def nested_precomputed(dmat, y, outer_splits=10, seed=0):
    n = len(y)
    splits = min(outer_splits, np.min(np.bincount(y)))
    outer = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    lambdas = [0.25, 0.5, 1, 2, 4, 8]
    cs = [0.1, 1, 10]
    scores = []
    for train, test in outer.split(np.zeros(n), y):
        inner_splits = min(3, np.min(np.bincount(y[train])))
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed + 1)
        best = (-1, None, None)
        for lam, c in itertools.product(lambdas, cs):
            kfull = np.exp(-lam * dmat)
            vals = []
            for tr0, va0 in inner.split(np.zeros(len(train)), y[train]):
                tr, va = train[tr0], train[va0]
                clf = SVC(kernel="precomputed", C=c)
                clf.fit(kfull[np.ix_(tr, tr)], y[tr])
                vals.append(accuracy_score(y[va], clf.predict(kfull[np.ix_(va, tr)])))
            score = float(np.mean(vals))
            if score > best[0]:
                best = (score, lam, c)
        _, lam, c = best
        kfull = np.exp(-lam * dmat)
        clf = SVC(kernel="precomputed", C=c)
        clf.fit(kfull[np.ix_(train, train)], y[train])
        scores.append(accuracy_score(y[test], clf.predict(kfull[np.ix_(test, train)])))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def nested_linear_features(x, y, outer_splits=10, seed=0):
    dmat = cdist(x, x, metric="sqeuclidean")
    return nested_precomputed(dmat, y, outer_splits, seed)


def experiment_tu_classification():
    rows = []
    budgets = {
        "MUTAG": dict(k=48, l=32),
        "PROTEINS": dict(k=24, l=16),
        "IMDB-BINARY": dict(k=24, l=16),
    }
    for name in ["MUTAG", "PROTEINS", "IMDB-BINARY"]:
        max_graphs = None
        graphs, y = load_tu(name, max_graphs=max_graphs)
        metrics = [graph_metric(g) for g in graphs]
        feat = sdmw_features(metrics, orders=[2, 3, 4, 6], seed=10, **budgets[name])
        d_sdmw = l1_distance_matrix(feat)
        acc, std = nested_precomputed(d_sdmw, y, seed=11)
        rows.append([name, "MS-SDMW", len(graphs), acc, std])
        for method, x in [
            ("Shortest-path", np.vstack([sp_hist(g) for g in graphs])),
            ("Degree", np.vstack([degree_hist(g) for g in graphs])),
            ("WL subtree", wl_features(graphs, h=4)),
            ("Graphlet", graphlet_features(graphs)),
        ]:
            acc, std = nested_linear_features(x, y, seed=12)
            rows.append([name, method, len(graphs), acc, std])
    df = pd.DataFrame(rows, columns=["dataset", "method", "num_graphs", "accuracy", "std"])
    df.to_csv(RES / "tu_classification.csv", index=False)
    return df


def experiment_two_sample():
    gen = rng(21)
    rows = []
    sample_sizes = [8, 16, 32, 64]
    shifts = [0.00, 0.08, 0.16, 0.24]
    for shift in shifts:
        for m in sample_sizes:
            rejections = 0
            trials = 50
            for t in range(trials):
                metrics = []
                labels = []
                for _ in range(m):
                    theta = gen.uniform(0, 2 * np.pi, 80)
                    x = np.column_stack([np.cos(theta), np.sin(theta)]) + 0.025 * gen.normal(size=(80, 2))
                    metrics.append(cdist(x, x) / np.max(cdist(x, x)))
                    labels.append(0)
                    theta = gen.uniform(0, 2 * np.pi, 80)
                    y = np.column_stack([(1 + shift) * np.cos(theta), (1 - 0.5 * shift) * np.sin(theta)])
                    y += 0.025 * gen.normal(size=(80, 2))
                    metrics.append(cdist(y, y) / np.max(cdist(y, y)))
                    labels.append(1)
                feat = sdmw_features(metrics, [2, 3, 4, 6], k=36, l=24, seed=int(gen.integers(1e9)))
                d = l1_distance_matrix(feat)
                med = np.median(d[np.triu_indices_from(d, 1)])
                gamma = 1.0 / max(med, 1e-6)
                k_mat = np.exp(-gamma * d)
                lab = np.array(labels)
                stat = mmd2(k_mat, lab)
                perm = []
                for _ in range(99):
                    perm.append(mmd2(k_mat, gen.permutation(lab)))
                pval = (1 + np.sum(np.array(perm) >= stat)) / (1 + len(perm))
                rejections += pval < 0.05
            rows.append([shift, m, rejections / trials])
    df = pd.DataFrame(rows, columns=["ellipse_shift", "sample_size", "power"])
    df.to_csv(RES / "two_sample_power.csv", index=False)
    plt.figure(figsize=(6, 4))
    for shift, sub in df.groupby("ellipse_shift"):
        plt.plot(sub.sample_size, sub.power, marker="o", label=f"shift={shift}")
    plt.xlabel("Metric spaces per group")
    plt.ylabel("Permutation-test power")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.title("Two-sample testing power")
    savefig("two_sample_power.pdf")
    return df


def mmd2(k, labels):
    a = np.where(labels == 0)[0]
    b = np.where(labels == 1)[0]
    return k[np.ix_(a, a)].mean() + k[np.ix_(b, b)].mean() - 2 * k[np.ix_(a, b)].mean()


def experiment_weight_ablation():
    gen = rng(31)
    graphs, y = load_tu("MUTAG")
    metrics = [graph_metric(g) for g in graphs]
    orders = [2, 3, 4, 6, 8]
    schemes = {
        "single n=2": [2],
        "single n=4": [4],
        "single n=8": [8],
        "uniform": orders,
        "inverse order": orders,
    }
    rows = []
    for scheme, ords in schemes.items():
        feat = sdmw_features(metrics, ords, k=36, l=24, seed=32)
        if scheme == "inverse order":
            # Rescale blocks by inverse-order weights.
            blocks = []
            start = 0
            weights = np.array([1 / n for n in ords], dtype=float)
            weights /= weights.sum()
            for n, w in zip(ords, weights):
                size = 24 * 36
                blocks.append(feat[:, start:start + size] * w * len(ords))
                start += size
            feat = np.hstack(blocks)
        d = l1_distance_matrix(feat)
        acc, std = nested_precomputed(d, y, outer_splits=10, seed=33)
        rows.append([scheme, acc, std])
    df = pd.DataFrame(rows, columns=["scheme", "accuracy", "std"])
    df.to_csv(RES / "weight_ablation.csv", index=False)
    plt.figure(figsize=(6, 3.8))
    plt.bar(df.scheme, df.accuracy, yerr=df["std"], capsize=3)
    plt.ylim(0.4, 1.0)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Multi-scale weight ablation on MUTAG")
    savefig("weight_ablation.pdf")
    return df


def experiment_sliced_unsliced_runtime():
    gen = rng(41)
    ns = [30, 50, 80, 120, 200, 350]
    rows = []
    for n_nodes in ns:
        g1 = nx.stochastic_block_model([n_nodes // 2, n_nodes - n_nodes // 2], [[0.18, 0.03], [0.03, 0.18]], seed=int(gen.integers(1e9)))
        g2 = nx.stochastic_block_model([n_nodes // 2, n_nodes - n_nodes // 2], [[0.15, 0.06], [0.06, 0.15]], seed=int(gen.integers(1e9)))
        d1, d2 = graph_metric(g1), graph_metric(g2)
        for method in ["POT-GW", "POT-entropic-GW", "Full-DMW", "Sliced-DMW"]:
            if method == "POT-GW" and n_nodes > 120:
                rows.append([n_nodes, method, np.nan])
                continue
            t0 = time.perf_counter()
            if method == "POT-GW":
                p = np.ones(n_nodes) / n_nodes
                ot.gromov.gromov_wasserstein2(d1, d2, p, p, "square_loss", max_iter=30)
            elif method == "POT-entropic-GW":
                p = np.ones(n_nodes) / n_nodes
                ot.gromov.entropic_gromov_wasserstein2(d1, d2, p, p, "square_loss", epsilon=0.05, max_iter=30)
            else:
                a = sample_vectors(d1, 6, 96, gen)
                b = sample_vectors(d2, 6, 96, gen)
                if method == "Full-DMW":
                    cost = np.abs(a[:, None, :] - b[None, :, :]).mean(axis=2)
                    linear_sum_assignment(cost)
                else:
                    th = gen.normal(size=(64, a.shape[1]))
                    th /= np.linalg.norm(th, axis=1, keepdims=True)
                    pa, pb = a @ th.T, b @ th.T
                    pa.sort(axis=0); pb.sort(axis=0)
                    np.abs(pa - pb).mean()
            rows.append([n_nodes, method, time.perf_counter() - t0])
    df = pd.DataFrame(rows, columns=["nodes", "method", "seconds"])
    df.to_csv(RES / "runtime_scaling.csv", index=False)
    plt.figure(figsize=(6, 4))
    for method, sub in df.groupby("method"):
        plt.plot(sub.nodes, sub.seconds, marker="o", label=method)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Number of nodes")
    plt.ylabel("Seconds")
    plt.legend()
    plt.title("Runtime scaling")
    savefig("runtime_scaling.pdf")
    return df


def experiment_parameter_scaling():
    gen = rng(51)
    g1 = nx.stochastic_block_model([60, 60], [[0.18, 0.03], [0.03, 0.18]], seed=1)
    g2 = nx.stochastic_block_model([60, 60], [[0.15, 0.06], [0.06, 0.15]], seed=2)
    d1, d2 = graph_metric(g1), graph_metric(g2)
    rows = []
    for k in [16, 32, 64, 128, 256]:
        t0 = time.perf_counter()
        a = sample_vectors(d1, 6, k, gen); b = sample_vectors(d2, 6, k, gen)
        th = gen.normal(size=(64, a.shape[1])); th /= np.linalg.norm(th, axis=1, keepdims=True)
        pa, pb = a @ th.T, b @ th.T; pa.sort(axis=0); pb.sort(axis=0)
        np.abs(pa - pb).mean()
        rows.append(["K", k, time.perf_counter() - t0])
    for l in [8, 16, 32, 64, 128, 256]:
        t0 = time.perf_counter()
        a = sample_vectors(d1, 6, 96, gen); b = sample_vectors(d2, 6, 96, gen)
        th = gen.normal(size=(l, a.shape[1])); th /= np.linalg.norm(th, axis=1, keepdims=True)
        pa, pb = a @ th.T, b @ th.T; pa.sort(axis=0); pb.sort(axis=0)
        np.abs(pa - pb).mean()
        rows.append(["L", l, time.perf_counter() - t0])
    df = pd.DataFrame(rows, columns=["variable", "value", "seconds"])
    df.to_csv(RES / "parameter_scaling.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, var in zip(axes, ["K", "L"]):
        sub = df[df.variable == var]
        ax.plot(sub.value, sub.seconds, marker="o")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel(var)
        ax.set_ylabel("Seconds")
        ax.set_title(f"Runtime vs {var}")
    savefig("parameter_scaling.pdf")
    return df


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    results = {
        "tu_classification": experiment_tu_classification().to_dict("records"),
        "two_sample": experiment_two_sample().to_dict("records"),
        "weight_ablation": experiment_weight_ablation().to_dict("records"),
        "runtime": experiment_sliced_unsliced_runtime().to_dict("records"),
        "parameter_scaling": experiment_parameter_scaling().to_dict("records"),
    }
    (RES / "full_summary.json").write_text(json.dumps(results, indent=2))
    print("done")


if __name__ == "__main__":
    main()
