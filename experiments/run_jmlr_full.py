"""Expanded JMLR-style experiment suite for Distance-Matrix Wasserstein.

The script is deliberately cache-heavy.  The TU datasets and graph metrics are
expensive enough that a failed long run should not force us to start from zero.
It produces publication-style PDF figures and CSV tables under
experiments/results_jmlr and experiments/figures_jmlr.
"""

from __future__ import annotations

import itertools
import json
import math
import ssl
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ot
import pandas as pd
import seaborn as sns
from scipy.linalg import eigvalsh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache_jmlr"
FIG = ROOT / "figures_jmlr"
RES = ROOT / "results_jmlr"
for folder in [DATA, CACHE, FIG, RES]:
    folder.mkdir(exist_ok=True)


DATASETS = [
    "MUTAG",
    "PTC_MR",
    "BZR",
    "COX2",
    "PROTEINS",
    "ENZYMES",
    "IMDB-BINARY",
    "IMDB-MULTI",
    "NCI1",
    "REDDIT-BINARY",
]

SMALL_GW_DATASETS = {"MUTAG", "PTC_MR", "BZR", "COX2"}


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def set_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=1.05,
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8.5,
        },
    )
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]
    )


def savefig(name: str) -> None:
    plt.savefig(FIG / name, bbox_inches="tight")
    plt.close()


def download_tu(name: str) -> Path:
    outdir = DATA / name
    if outdir.exists() and (outdir / f"{name}_A.txt").exists():
        return outdir
    url = f"https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip"
    zpath = DATA / f"{name}.zip"
    print(f"[download] {name}")
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=ctx, timeout=240) as response:
        zpath.write_bytes(response.read())
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(DATA)
    return outdir


def read_int_file(path: Path) -> list[int]:
    return [int(x.strip()) for x in path.read_text().splitlines() if x.strip()]


def load_tu(name: str):
    cache_path = CACHE / f"{name}_graphs.pkl"
    if cache_path.exists():
        import pickle

        with cache_path.open("rb") as f:
            return pickle.load(f)

    d = download_tu(name)
    graph_indicator = read_int_file(d / f"{name}_graph_indicator.txt")
    raw_labels = np.array(read_int_file(d / f"{name}_graph_labels.txt"))
    n_graphs = max(graph_indicator)
    graphs = [nx.Graph() for _ in range(n_graphs)]
    node_to_graph: dict[int, int] = {}
    local_id: dict[int, int] = {}
    counters = [0] * n_graphs
    for global_idx, gid in enumerate(graph_indicator, start=1):
        gidx = gid - 1
        lid = counters[gidx]
        counters[gidx] += 1
        node_to_graph[global_idx] = gidx
        local_id[global_idx] = lid
        graphs[gidx].add_node(lid)

    node_label_path = d / f"{name}_node_labels.txt"
    if node_label_path.exists():
        node_labels = read_int_file(node_label_path)
        for global_idx, lab in enumerate(node_labels, start=1):
            graphs[node_to_graph[global_idx]].nodes[local_id[global_idx]]["label"] = int(lab)

    with open(d / f"{name}_A.txt", "r") as f:
        for line in f:
            if not line.strip():
                continue
            a, b = [int(x.strip()) for x in line.split(",")]
            ga, gb = node_to_graph[a], node_to_graph[b]
            if ga == gb:
                graphs[ga].add_edge(local_id[a], local_id[b])

    label_map = {v: i for i, v in enumerate(sorted(set(raw_labels)))}
    y = np.array([label_map[v] for v in raw_labels], dtype=int)

    import pickle

    with cache_path.open("wb") as f:
        pickle.dump((graphs, y), f)
    return graphs, y


def connected_copy(g: nx.Graph) -> nx.Graph:
    if nx.is_connected(g):
        return g
    h = g.copy()
    comps = [list(c) for c in nx.connected_components(h)]
    for c1, c2 in zip(comps[:-1], comps[1:]):
        h.add_edge(c1[0], c2[0])
    return h


def graph_metric(g: nx.Graph, max_nodes: int | None = None, seed: int = 0) -> np.ndarray:
    h = connected_copy(g)
    nodes = list(h.nodes())
    if max_nodes is not None and len(nodes) > max_nodes:
        gen = rng(seed + len(nodes))
        nodes = list(gen.choice(nodes, size=max_nodes, replace=False))
        h = connected_copy(h.subgraph(nodes).copy())
        nodes = list(h.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    d = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    lengths = dict(nx.all_pairs_shortest_path_length(h))
    for u in nodes:
        for v, val in lengths[u].items():
            if v in idx:
                d[idx[u], idx[v]] = val
    mx = float(d.max())
    return d / mx if mx > 0 else d


def metric_cache_path(name: str, max_nodes: int | None) -> Path:
    suffix = "full" if max_nodes is None else f"cap{max_nodes}"
    return CACHE / f"{name}_metrics_{suffix}.npz"


def load_metrics(name: str, graphs: list[nx.Graph]) -> list[np.ndarray]:
    max_nodes = 220 if name in {"REDDIT-BINARY", "COLLAB"} else None
    path = metric_cache_path(name, max_nodes)
    if path.exists():
        arr = np.load(path, allow_pickle=True)["metrics"]
        return list(arr)
    print(f"[metrics] {name}")
    metrics = [graph_metric(g, max_nodes=max_nodes, seed=i) for i, g in enumerate(graphs)]
    np.savez_compressed(path, metrics=np.array(metrics, dtype=object))
    return metrics


def sample_vectors(dist: np.ndarray, order: int, k: int, gen: np.random.Generator) -> np.ndarray:
    m = dist.shape[0]
    pairs = np.triu_indices(order, 1)
    out = np.empty((k, len(pairs[0])), dtype=np.float32)
    for i in range(k):
        ids = gen.choice(m, size=order, replace=True)
        out[i] = dist[np.ix_(ids, ids)][pairs]
    return out


def sdmw_features(metrics: list[np.ndarray], orders: list[int], k: int, l: int, seed: int) -> np.ndarray:
    gen = rng(seed)
    directions = {}
    for order in orders:
        dim = order * (order - 1) // 2
        theta = gen.normal(size=(l, dim)).astype(np.float32)
        theta /= np.maximum(np.linalg.norm(theta, axis=1, keepdims=True), 1e-12)
        directions[order] = theta
    features = []
    for dist in metrics:
        blocks = []
        for order in orders:
            vecs = sample_vectors(dist, order, k, gen)
            proj = vecs @ directions[order].T
            proj.sort(axis=0)
            blocks.append(proj.T.reshape(-1))
        features.append(np.concatenate(blocks))
    return np.vstack(features).astype(np.float32)


def l1_distance_matrix(x: np.ndarray) -> np.ndarray:
    return cdist(x, x, metric="cityblock") / x.shape[1]


def sp_hist(g: nx.Graph, bins: int = 16) -> np.ndarray:
    d = graph_metric(g)
    vals = d[np.triu_indices_from(d, 1)]
    hist, _ = np.histogram(vals, bins=bins, range=(0, 1))
    return hist.astype(float) / max(1, hist.sum())


def degree_hist(g: nx.Graph, bins: int = 16) -> np.ndarray:
    vals = np.array([d for _, d in g.degree()], dtype=float)
    if vals.size == 0:
        vals = np.array([0.0])
    if vals.max() > 0:
        vals = vals / vals.max()
    hist, _ = np.histogram(vals, bins=bins, range=(0, 1))
    return hist.astype(float) / max(1, hist.sum())


def graphlet_features(graphs: list[nx.Graph]) -> np.ndarray:
    rows = []
    for g in graphs:
        deg = np.array([d for _, d in g.degree()], dtype=float)
        if len(deg) == 0:
            deg = np.array([0.0])
        edges = g.number_of_edges()
        wedges = float(np.sum(deg * (deg - 1) / 2))
        triangles = float(sum(nx.triangles(g).values()) / 3)
        clustering = nx.average_clustering(g) if g.number_of_nodes() > 2 else 0.0
        density = nx.density(g)
        rows.append([g.number_of_nodes(), edges, wedges, triangles, clustering, density, deg.mean(), deg.std()])
    x = np.asarray(rows, dtype=float)
    return (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-12)


def wl_features(graphs: list[nx.Graph], h: int = 5) -> np.ndarray:
    vocab: dict[str, int] = {}
    row_dicts_all = []
    labels = []
    for g in graphs:
        lab = {}
        for u in g.nodes():
            lab[u] = str(g.nodes[u].get("label", g.degree(u)))
        labels.append(lab)
    for it in range(h + 1):
        row_dicts = []
        for gi, g in enumerate(graphs):
            counts = {}
            for u in g.nodes():
                key = f"{it}:{labels[gi][u]}"
                if key not in vocab:
                    vocab[key] = len(vocab)
                j = vocab[key]
                counts[j] = counts.get(j, 0) + 1
            row_dicts.append(counts)
        row_dicts_all.append(row_dicts)
        if it < h:
            mapper = {}
            next_labels = []
            for gi, g in enumerate(graphs):
                new = {}
                for u in g.nodes():
                    sig = (labels[gi][u], tuple(sorted(labels[gi][v] for v in g.neighbors(u))))
                    if sig not in mapper:
                        mapper[sig] = str(len(mapper))
                    new[u] = mapper[sig]
                next_labels.append(new)
            labels = next_labels
    x = np.zeros((len(graphs), len(vocab)), dtype=np.float32)
    for row_dicts in row_dicts_all:
        for i, counts in enumerate(row_dicts):
            for j, c in counts.items():
                x[i, j] += c
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x


def netlsd_features(graphs: list[nx.Graph], times: np.ndarray | None = None) -> np.ndarray:
    if times is None:
        times = np.logspace(-2, 2, 32)
    rows = []
    for g in graphs:
        h = connected_copy(g)
        n = h.number_of_nodes()
        if n == 0:
            rows.append(np.zeros(len(times)))
            continue
        lap = nx.normalized_laplacian_matrix(h).toarray().astype(float)
        vals = eigvalsh(lap)
        heat = np.exp(-np.outer(times, vals)).sum(axis=1) / n
        rows.append(heat)
    x = np.asarray(rows, dtype=float)
    return (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-12)


def nested_precomputed(
    dmat: np.ndarray,
    y: np.ndarray,
    outer_splits: int = 10,
    seed: int = 0,
    lambdas: list[float] | None = None,
    cs: list[float] | None = None,
    inner_splits_cap: int = 3,
):
    n = len(y)
    class_counts = np.bincount(y)
    splits = max(2, min(outer_splits, int(class_counts.min())))
    outer = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    if lambdas is None:
        lambdas = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    if cs is None:
        cs = [0.1, 1, 10]
    scores = []
    train_times = []
    for train, test in outer.split(np.zeros(n), y):
        inner_splits = max(2, min(inner_splits_cap, int(np.bincount(y[train]).min())))
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed + 1)
        best = (-np.inf, None, None)
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
        t0 = time.perf_counter()
        clf = SVC(kernel="precomputed", C=c)
        clf.fit(kfull[np.ix_(train, train)], y[train])
        train_times.append(time.perf_counter() - t0)
        scores.append(accuracy_score(y[test], clf.predict(kfull[np.ix_(test, train)])))
    return float(np.mean(scores)), float(np.std(scores, ddof=1)), float(np.mean(train_times))


def nested_feature_kernel(
    x: np.ndarray,
    y: np.ndarray,
    outer_splits: int = 10,
    seed: int = 0,
    lambdas: list[float] | None = None,
    cs: list[float] | None = None,
    inner_splits_cap: int = 3,
):
    dmat = cdist(x, x, metric="sqeuclidean")
    return nested_precomputed(
        dmat,
        y,
        outer_splits=outer_splits,
        seed=seed,
        lambdas=lambdas,
        cs=cs,
        inner_splits_cap=inner_splits_cap,
    )


def protocol(name: str):
    if name in {"NCI1", "REDDIT-BINARY"}:
        return {
            "outer_splits": 5,
            "inner_splits_cap": 2,
            "lambdas": [0.5, 1, 2],
            "cs": [1, 10],
        }
    return {
        "outer_splits": 10,
        "inner_splits_cap": 3,
        "lambdas": None,
        "cs": None,
    }


def dataset_budget(name: str) -> tuple[int, int, list[int]]:
    if name in {"NCI1", "REDDIT-BINARY", "IMDB-MULTI"}:
        return 16, 12, [2, 3, 4, 6]
    if name in {"PROTEINS", "ENZYMES", "IMDB-BINARY"}:
        return 24, 16, [2, 3, 4, 6]
    return 48, 32, [2, 3, 4, 6, 8]


def feature_or_compute(name: str, method: str, compute_fn):
    path = CACHE / f"{name}_{method}.npz"
    if path.exists():
        return np.load(path, allow_pickle=True)["x"]
    x = compute_fn()
    np.savez_compressed(path, x=x)
    return x


def run_tu_classification() -> pd.DataFrame:
    out_path = RES / "tu_classification_full.csv"
    if out_path.exists():
        rows = pd.read_csv(out_path).values.tolist()
    else:
        rows = []
    stats = []
    completed = {(r[0], r[1]) for r in rows}
    for name in DATASETS:
        print(f"[classification] {name}", flush=True)
        graphs, y = load_tu(name)
        metrics = load_metrics(name, graphs)
        cfg = protocol(name)
        sizes = np.array([g.number_of_nodes() for g in graphs])
        stats.append(
            {
                "dataset": name,
                "num_graphs": len(graphs),
                "num_classes": len(np.unique(y)),
                "avg_nodes": float(sizes.mean()),
                "median_nodes": float(np.median(sizes)),
                "max_nodes": int(sizes.max()),
            }
        )
        k, l, orders = dataset_budget(name)
        if (name, "MS-SDMW") not in completed:
            print(f"  method=MS-SDMW", flush=True)
            t0 = time.perf_counter()
            sdmw_feat = feature_or_compute(
                name,
                f"MS_SDMW_k{k}_l{l}",
                lambda metrics=metrics, orders=orders, k=k, l=l, name=name: sdmw_features(
                    metrics, orders, k=k, l=l, seed=101 + len(name)
                ),
            )
            d_sdmw = l1_distance_matrix(sdmw_feat)
            kernel_time = time.perf_counter() - t0
            acc, std, train_time = nested_precomputed(d_sdmw, y, seed=111, **cfg)
            rows.append([name, "MS-SDMW", len(graphs), acc, std, kernel_time, train_time])
            pd.DataFrame(
                rows,
                columns=["dataset", "method", "num_graphs", "accuracy", "std", "kernel_seconds", "svm_seconds"],
            ).to_csv(out_path, index=False)
            completed.add((name, "MS-SDMW"))

        baselines = [
            ("Shortest-path", lambda graphs=graphs: np.vstack([sp_hist(g) for g in graphs])),
            ("Degree", lambda graphs=graphs: np.vstack([degree_hist(g) for g in graphs])),
            ("WL subtree", lambda graphs=graphs: wl_features(graphs, h=5)),
            ("Graphlet", lambda graphs=graphs: graphlet_features(graphs)),
            ("NetLSD", lambda graphs=graphs: netlsd_features(graphs)),
        ]
        for method, fn in baselines:
            if (name, method) in completed:
                continue
            print(f"  method={method}", flush=True)
            t0 = time.perf_counter()
            x = feature_or_compute(name, method.replace(" ", "_"), fn)
            kernel_time = time.perf_counter() - t0
            acc, std, train_time = nested_feature_kernel(x, y, seed=112, **cfg)
            rows.append([name, method, len(graphs), acc, std, kernel_time, train_time])
            pd.DataFrame(
                rows,
                columns=["dataset", "method", "num_graphs", "accuracy", "std", "kernel_seconds", "svm_seconds"],
            ).to_csv(out_path, index=False)
            completed.add((name, method))

        if name in SMALL_GW_DATASETS:
            if (name, "Entropic-GW") in completed:
                continue
            print(f"  method=Entropic-GW", flush=True)
            t0 = time.perf_counter()
            dgw_path = CACHE / f"{name}_entropic_gw_dmat.npz"
            if dgw_path.exists():
                dgw = np.load(dgw_path)["d"]
            else:
                dgw = pairwise_entropic_gw(metrics, max_pairs=None)
                np.savez_compressed(dgw_path, d=dgw)
            kernel_time = time.perf_counter() - t0
            acc, std, train_time = nested_precomputed(dgw, y, seed=113, **cfg)
            rows.append([name, "Entropic-GW", len(graphs), acc, std, kernel_time, train_time])
            pd.DataFrame(
                rows,
                columns=["dataset", "method", "num_graphs", "accuracy", "std", "kernel_seconds", "svm_seconds"],
            ).to_csv(out_path, index=False)
            completed.add((name, "Entropic-GW"))

    df = pd.DataFrame(
        rows,
        columns=["dataset", "method", "num_graphs", "accuracy", "std", "kernel_seconds", "svm_seconds"],
    )
    df.to_csv(out_path, index=False)
    pd.DataFrame(stats).to_csv(RES / "tu_dataset_stats.csv", index=False)
    plot_tu_results(df, pd.DataFrame(stats))
    return df


def pairwise_entropic_gw(metrics: list[np.ndarray], max_pairs: int | None = None) -> np.ndarray:
    n = len(metrics)
    dmat = np.zeros((n, n), dtype=np.float32)
    pairs = list(itertools.combinations(range(n), 2))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    for count, (i, j) in enumerate(pairs, start=1):
        if count % 500 == 0:
            print(f"  [entropic-gw] {count}/{len(pairs)}")
        a, b = metrics[i], metrics[j]
        pa = np.ones(a.shape[0]) / a.shape[0]
        pb = np.ones(b.shape[0]) / b.shape[0]
        val = ot.gromov.entropic_gromov_wasserstein2(
            a,
            b,
            pa,
            pb,
            loss_fun="square_loss",
            epsilon=0.08,
            max_iter=30,
            tol=1e-7,
            verbose=False,
        )
        dmat[i, j] = dmat[j, i] = float(max(val, 0.0)) ** 0.5
    return dmat


def plot_tu_results(df: pd.DataFrame, stats: pd.DataFrame) -> None:
    order = DATASETS
    method_order = ["MS-SDMW", "Entropic-GW", "Shortest-path", "Degree", "WL subtree", "Graphlet", "NetLSD"]
    present = [m for m in method_order if m in set(df.method)]
    plt.figure(figsize=(12.2, 5.2))
    ax = sns.barplot(
        data=df,
        x="dataset",
        y="accuracy",
        hue="method",
        order=order,
        hue_order=present,
        errorbar=None,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=6, rotation=90, padding=1)
    ax.set_ylim(0.35, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Nested-CV accuracy")
    ax.set_title("TU graph classification: metric DMW kernel versus structural baselines")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(ncol=4, frameon=True, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    savefig("tu_classification_full.pdf")

    plt.figure(figsize=(8.5, 4.0))
    sub = df[df.method.isin(["MS-SDMW", "WL subtree", "Graphlet", "NetLSD"])]
    sns.scatterplot(data=sub, x="kernel_seconds", y="accuracy", hue="method", style="dataset", s=70)
    plt.xscale("log")
    plt.xlabel("Kernel/feature construction time (s, log scale)")
    plt.ylabel("Nested-CV accuracy")
    plt.title("Accuracy--runtime profile on TU benchmarks")
    plt.legend(ncol=2, fontsize=7, frameon=True)
    savefig("tu_accuracy_runtime.pdf")

    plt.figure(figsize=(8.5, 3.5))
    sns.barplot(data=stats, x="dataset", y="avg_nodes", order=order, color="#4477AA")
    plt.yscale("log")
    plt.xlabel("")
    plt.ylabel("Average number of nodes")
    plt.title("Scale of the TU benchmark suite")
    plt.xticks(rotation=35, ha="right")
    savefig("tu_dataset_scales.pdf")


def circle_metric(num: int, axes: tuple[float, float], noise: float, gen: np.random.Generator) -> np.ndarray:
    theta = gen.uniform(0, 2 * np.pi, num)
    pts = np.column_stack([axes[0] * np.cos(theta), axes[1] * np.sin(theta)])
    pts += noise * gen.normal(size=pts.shape)
    d = cdist(pts, pts)
    return d / max(float(d.max()), 1e-12)


def sdmw_distance_between_metrics(dx: np.ndarray, dy: np.ndarray, order: int, k: int, l: int, seed: int) -> float:
    gen = rng(seed)
    a = sample_vectors(dx, order, k, gen)
    b = sample_vectors(dy, order, k, gen)
    dim = a.shape[1]
    theta = gen.normal(size=(l, dim))
    theta /= np.maximum(np.linalg.norm(theta, axis=1, keepdims=True), 1e-12)
    pa = a @ theta.T
    pb = b @ theta.T
    pa.sort(axis=0)
    pb.sort(axis=0)
    return float(np.abs(pa - pb).mean())


def run_theory_validation() -> pd.DataFrame:
    gen = rng(200)
    orders = [2, 3, 4, 5, 6, 8, 10, 12]
    reps = 40
    rows = []
    dx_ref = circle_metric(140, (1.0, 1.0), 0.015, gen)
    dy_ref = circle_metric(140, (1.28, 0.82), 0.015, gen)
    reference = sdmw_distance_between_metrics(dx_ref, dy_ref, order=12, k=900, l=256, seed=201)
    for order in orders:
        vals = []
        for rep in range(reps):
            dx = circle_metric(120, (1.0, 1.0), 0.015, gen)
            dy = circle_metric(120, (1.28, 0.82), 0.015, gen)
            vals.append(sdmw_distance_between_metrics(dx, dy, order=order, k=96, l=64, seed=3000 + 37 * rep + order))
        vals = np.array(vals)
        rows.append(
            {
                "order": order,
                "mean_estimate": vals.mean(),
                "std_estimate": vals.std(ddof=1),
                "reference_gap": abs(vals.mean() - reference),
                "total_error": np.mean(np.abs(vals - reference)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RES / "theory_tradeoff_full.csv", index=False)
    plot_theory_validation(df)
    return df


def plot_theory_validation(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex=True)
    ax = axes[0, 0]
    ax.plot(df.order, df.reference_gap, marker="o")
    ax.set_ylabel("Approximation gap")
    ax.set_title("(a) Population proxy improves with order")
    ax = axes[0, 1]
    ax.plot(df.order, df.std_estimate, marker="o", color="#EE6677")
    ax.set_ylabel("Std. over repetitions")
    ax.set_title("(b) Estimation variability grows")
    ax = axes[1, 0]
    ax.plot(df.order, df.total_error, marker="o", color="#228833")
    ax.set_xlabel("Distance-matrix order $n$")
    ax.set_ylabel("Total empirical error")
    ax.set_title("(c) Finite-sample tradeoff")
    ax = axes[1, 1]
    ax.errorbar(df.order, df.mean_estimate, yerr=df.std_estimate, marker="o", capsize=3, color="#AA3377")
    ax.set_xlabel("Distance-matrix order $n$")
    ax.set_ylabel("SDMW estimate")
    ax.set_title("(d) Estimates with uncertainty")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    fig.suptitle("Empirical validation of the approximation--estimation tradeoff", y=1.02)
    savefig("theory_tradeoff_2x2.pdf")


def mmd2(kernel: np.ndarray, labels: np.ndarray) -> float:
    a = np.where(labels == 0)[0]
    b = np.where(labels == 1)[0]
    return float(kernel[np.ix_(a, a)].mean() + kernel[np.ix_(b, b)].mean() - 2 * kernel[np.ix_(a, b)].mean())


def run_two_sample() -> pd.DataFrame:
    gen = rng(400)
    rows = []
    sample_sizes = [8, 16, 32, 64, 96]
    shifts = [0.00, 0.04, 0.08, 0.12, 0.16, 0.24]
    trials = 80
    permutations = 199
    for shift in shifts:
        for m in sample_sizes:
            rejects = 0
            for trial in range(trials):
                metrics = []
                labels = []
                for _ in range(m):
                    metrics.append(circle_metric(90, (1.0, 1.0), 0.025, gen))
                    labels.append(0)
                    metrics.append(circle_metric(90, (1.0 + shift, 1.0 - 0.45 * shift), 0.025, gen))
                    labels.append(1)
                feat = sdmw_features(metrics, [2, 3, 4, 6], k=40, l=32, seed=int(gen.integers(1_000_000_000)))
                d = l1_distance_matrix(feat)
                med = np.median(d[np.triu_indices_from(d, 1)])
                kernel = np.exp(-d / max(med, 1e-6))
                lab = np.array(labels)
                stat = mmd2(kernel, lab)
                null = [mmd2(kernel, gen.permutation(lab)) for _ in range(permutations)]
                pval = (1 + np.sum(np.array(null) >= stat)) / (1 + permutations)
                rejects += pval < 0.05
            rows.append({"ellipse_shift": shift, "sample_size": m, "power": rejects / trials})
            print(f"[two-sample] shift={shift} m={m} power={rejects / trials:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(RES / "two_sample_power_full.csv", index=False)
    plt.figure(figsize=(6.6, 4.2))
    for shift, sub in df.groupby("ellipse_shift"):
        plt.plot(sub.sample_size, sub.power, marker="o", label=f"$\\Delta={shift:.2f}$")
    plt.axhline(0.05, color="0.25", ls="--", lw=1, label="nominal $0.05$")
    plt.xlabel("Metric spaces per group")
    plt.ylabel("Rejection probability")
    plt.ylim(0, 1.03)
    plt.title("Two-sample power of the MS-SDMW kernel")
    plt.legend(ncol=3, frameon=True)
    savefig("two_sample_power_full.pdf")
    return df


def run_finite_direction() -> pd.DataFrame:
    gen = rng(450)
    rows = []
    signals = [0.03, 0.05, 0.08, 0.12]
    directions = [2, 4, 8, 16, 32, 64, 128]
    trials = 160
    order = 6
    k = 160
    for signal in signals:
        dx = circle_metric(130, (1.0, 1.0), 0.01, gen)
        dy = circle_metric(130, (1.0 + signal, 1.0 - 0.45 * signal), 0.01, gen)
        ref_vals = [
            sdmw_distance_between_metrics(dx, dy, order=order, k=k, l=512, seed=9000 + r + int(signal * 1000))
            for r in range(25)
        ]
        ref = float(np.mean(ref_vals))
        for l in directions:
            vals = [
                sdmw_distance_between_metrics(dx, dy, order=order, k=k, l=l, seed=10000 + t + 97 * l)
                for t in range(trials)
            ]
            vals = np.array(vals)
            rows.append(
                {
                    "signal": signal,
                    "directions": l,
                    "reference": ref,
                    "mean_estimate": vals.mean(),
                    "std_estimate": vals.std(ddof=1),
                    "success_prob": float(np.mean(np.abs(vals - ref) <= 0.20 * max(ref, 1e-12))),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(RES / "finite_direction_full.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for signal, sub in df.groupby("signal"):
        axes[0].plot(sub.directions, sub.success_prob, marker="o", label=f"$\\Delta={signal:.2f}$")
        axes[1].errorbar(sub.directions, sub.mean_estimate, yerr=sub.std_estimate, marker="o", capsize=2)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Number of directions $L$")
    axes[0].set_ylabel("Within $20\\%$ of high-$L$ reference")
    axes[0].set_ylim(-0.02, 1.03)
    axes[0].set_title("(a) Separation probability")
    axes[0].legend(frameon=True, ncol=2)
    axes[1].set_ylabel("Sliced statistic")
    axes[1].set_title("(b) Monte Carlo variability")
    fig.suptitle("Finite-direction concentration of sliced DMW", y=1.03)
    savefig("finite_direction_full.pdf")
    return df


def run_runtime_scaling() -> pd.DataFrame:
    gen = rng(500)
    node_sizes = [30, 50, 80, 120, 200, 350, 600, 1000]
    rows = []
    for n_nodes in node_sizes:
        print(f"[runtime] n={n_nodes}")
        g1 = nx.stochastic_block_model(
            [n_nodes // 2, n_nodes - n_nodes // 2],
            [[0.16, 0.03], [0.03, 0.16]],
            seed=int(gen.integers(1_000_000_000)),
        )
        g2 = nx.stochastic_block_model(
            [n_nodes // 2, n_nodes - n_nodes // 2],
            [[0.13, 0.06], [0.06, 0.13]],
            seed=int(gen.integers(1_000_000_000)),
        )
        d1, d2 = graph_metric(g1), graph_metric(g2)
        methods = ["POT-GW", "POT-entropic-GW", "Full-DMW", "Sliced-DMW"]
        for method in methods:
            if method == "POT-GW" and n_nodes > 120:
                rows.append([n_nodes, method, np.nan])
                continue
            if method == "POT-entropic-GW" and n_nodes > 600:
                rows.append([n_nodes, method, np.nan])
                continue
            t0 = time.perf_counter()
            if method == "POT-GW":
                p = np.ones(n_nodes) / n_nodes
                ot.gromov.gromov_wasserstein2(d1, d2, p, p, "square_loss", max_iter=30)
            elif method == "POT-entropic-GW":
                p = np.ones(n_nodes) / n_nodes
                ot.gromov.entropic_gromov_wasserstein2(
                    d1, d2, p, p, "square_loss", epsilon=0.06, max_iter=30, tol=1e-7
                )
            else:
                a = sample_vectors(d1, 6, 128, gen)
                b = sample_vectors(d2, 6, 128, gen)
                if method == "Full-DMW":
                    cost = np.abs(a[:, None, :] - b[None, :, :]).mean(axis=2)
                    linear_sum_assignment(cost)
                else:
                    theta = gen.normal(size=(96, a.shape[1]))
                    theta /= np.maximum(np.linalg.norm(theta, axis=1, keepdims=True), 1e-12)
                    pa, pb = a @ theta.T, b @ theta.T
                    pa.sort(axis=0)
                    pb.sort(axis=0)
                    np.abs(pa - pb).mean()
            rows.append([n_nodes, method, time.perf_counter() - t0])
    df = pd.DataFrame(rows, columns=["nodes", "method", "seconds"])
    df.to_csv(RES / "runtime_scaling_full.csv", index=False)
    plt.figure(figsize=(6.4, 4.2))
    for method, sub in df.groupby("method"):
        plt.plot(sub.nodes, sub.seconds, marker="o", label=method)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of graph nodes")
    plt.ylabel("Wall-clock time (s)")
    plt.title("Scalability: DMW avoids node-level GW optimization")
    plt.legend(frameon=True)
    savefig("runtime_scaling_full.pdf")
    return df


def run_parameter_scaling() -> pd.DataFrame:
    gen = rng(600)
    g1 = nx.stochastic_block_model([120, 120], [[0.16, 0.03], [0.03, 0.16]], seed=1)
    g2 = nx.stochastic_block_model([120, 120], [[0.13, 0.06], [0.06, 0.13]], seed=2)
    d1, d2 = graph_metric(g1), graph_metric(g2)
    rows = []
    for k in [16, 32, 64, 128, 256, 512, 1024]:
        t0 = time.perf_counter()
        a = sample_vectors(d1, 6, k, gen)
        b = sample_vectors(d2, 6, k, gen)
        theta = gen.normal(size=(128, a.shape[1]))
        theta /= np.maximum(np.linalg.norm(theta, axis=1, keepdims=True), 1e-12)
        pa, pb = a @ theta.T, b @ theta.T
        pa.sort(axis=0)
        pb.sort(axis=0)
        np.abs(pa - pb).mean()
        rows.append(["K", k, time.perf_counter() - t0])
    for l in [8, 16, 32, 64, 128, 256, 512]:
        t0 = time.perf_counter()
        a = sample_vectors(d1, 6, 256, gen)
        b = sample_vectors(d2, 6, 256, gen)
        theta = gen.normal(size=(l, a.shape[1]))
        theta /= np.maximum(np.linalg.norm(theta, axis=1, keepdims=True), 1e-12)
        pa, pb = a @ theta.T, b @ theta.T
        pa.sort(axis=0)
        pb.sort(axis=0)
        np.abs(pa - pb).mean()
        rows.append(["L", l, time.perf_counter() - t0])
    for order in [2, 3, 4, 6, 8, 10, 12]:
        t0 = time.perf_counter()
        a = sample_vectors(d1, order, 256, gen)
        b = sample_vectors(d2, order, 256, gen)
        theta = gen.normal(size=(128, a.shape[1]))
        theta /= np.maximum(np.linalg.norm(theta, axis=1, keepdims=True), 1e-12)
        pa, pb = a @ theta.T, b @ theta.T
        pa.sort(axis=0)
        pb.sort(axis=0)
        np.abs(pa - pb).mean()
        rows.append(["n", order, time.perf_counter() - t0])
    df = pd.DataFrame(rows, columns=["variable", "value", "seconds"])
    df.to_csv(RES / "parameter_scaling_full.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    for ax, var in zip(axes, ["K", "L", "n"]):
        sub = df[df.variable == var]
        ax.plot(sub.value, sub.seconds, marker="o")
        ax.set_xscale("log", base=2 if var != "n" else 10)
        ax.set_yscale("log")
        ax.set_xlabel(var if var != "n" else "Order $n$")
        ax.set_ylabel("Seconds")
        ax.set_title(f"Runtime vs {var}")
    savefig("parameter_scaling_full.pdf")
    return df


def run_weight_ablation() -> pd.DataFrame:
    graphs, y = load_tu("MUTAG")
    metrics = load_metrics("MUTAG", graphs)
    schemes = {
        "single n=2": ([2], None),
        "single n=3": ([3], None),
        "single n=4": ([4], None),
        "single n=6": ([6], None),
        "single n=8": ([8], None),
        "uniform": ([2, 3, 4, 6, 8], "uniform"),
        "inverse order": ([2, 3, 4, 6, 8], "inverse"),
        "sqrt inverse": ([2, 3, 4, 6, 8], "sqrt_inverse"),
    }
    rows = []
    for scheme, (orders, weight_mode) in schemes.items():
        feat = sdmw_features(metrics, orders, k=48, l=32, seed=777)
        if weight_mode is not None:
            blocks = []
            start = 0
            if weight_mode == "uniform":
                weights = np.ones(len(orders)) / len(orders)
            elif weight_mode == "inverse":
                weights = np.array([1 / n for n in orders], dtype=float)
                weights /= weights.sum()
            else:
                weights = np.array([1 / math.sqrt(n) for n in orders], dtype=float)
                weights /= weights.sum()
            for order, w in zip(orders, weights):
                size = 48 * 32
                blocks.append(feat[:, start : start + size] * w * len(orders))
                start += size
            feat = np.hstack(blocks)
        d = l1_distance_matrix(feat)
        acc, std, train_time = nested_precomputed(d, y, seed=778)
        rows.append([scheme, acc, std, train_time])
    df = pd.DataFrame(rows, columns=["scheme", "accuracy", "std", "svm_seconds"])
    df.to_csv(RES / "weight_ablation_full.csv", index=False)
    plt.figure(figsize=(7.2, 3.8))
    ax = sns.barplot(data=df, x="scheme", y="accuracy", color="#4477AA", errorbar=None)
    ax.errorbar(np.arange(len(df)), df.accuracy, yerr=df["std"], fmt="none", ecolor="0.15", capsize=3, lw=1)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Nested-CV accuracy")
    ax.set_title("Multi-scale weighting is helpful but dataset dependent")
    ax.tick_params(axis="x", rotation=30)
    savefig("weight_ablation_full.pdf")
    return df


def main() -> None:
    set_style()
    outputs = {
        "theory": run_theory_validation().to_dict("records"),
        "finite_direction": run_finite_direction().to_dict("records"),
        "classification": run_tu_classification().to_dict("records"),
        "two_sample": run_two_sample().to_dict("records"),
        "runtime": run_runtime_scaling().to_dict("records"),
        "parameter_scaling": run_parameter_scaling().to_dict("records"),
        "weight_ablation": run_weight_ablation().to_dict("records"),
    }
    (RES / "summary_jmlr.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print("[done] expanded JMLR experiment suite")


if __name__ == "__main__":
    main()
