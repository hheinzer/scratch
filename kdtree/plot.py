import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from kdtree import KDTree

rng = np.random.default_rng(0)

POINT = "grey"
EC = "orange"
FC = "none"
LINE = "orange"
QUERY = "purple"
BAR = "purple"


def create_uniform(num=1000, dim=2):
    return rng.uniform(-1, 1, (num, dim))


def create_meshgrid(num=32, dim=2, jitter=0.01):
    x = np.linspace(-1, 1, num)
    y = np.linspace(-1, 1, num)
    X, Y = np.meshgrid(x, y)
    centers = np.column_stack((X.flat, Y.flat))
    return centers + rng.normal(0, jitter, (num**2, dim))


def create_clusters(num=1000, dim=2, count=10):
    centers = rng.uniform(-0.7, 0.7, (count, dim))
    cluster = rng.integers(0, count, num)
    return centers[cluster] + rng.normal(0, 0.1, (num, dim))


def load_dump(tree):
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        path = tmp.name
    tree.dump(path)
    with open(path) as file:
        header = file.readline()
    dim = int(header.split("dim=")[1].split()[0])
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[np.newaxis]
    idx = data[:, 0].astype(int)
    num = data[:, 1].astype(int)
    axis = data[:, 2].astype(int)
    value = data[:, 3]
    left = data[:, 4].astype(int)
    right = data[:, 5].astype(int)
    beg = data[:, 6].astype(int)
    end = data[:, 7].astype(int)
    depth = data[:, 8].astype(int)
    bbox_min = data[:, 9 : 9 + dim]
    bbox_max = data[:, 9 + dim :]
    return dict(
        dim=dim,
        idx=idx,
        num=num,
        axis=axis,
        value=value,
        left=left,
        right=right,
        beg=beg,
        end=end,
        depth=depth,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )


def plot_tree(points, dump):
    _, ax = plt.subplots()
    depths = dump["depth"]
    cmap = plt.get_cmap("Oranges")
    for lo, hi, depth in zip(dump["bbox_min"], dump["bbox_max"], depths):
        color = cmap(0.5 * depth / depths.max())
        ax.add_patch(patches.Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1], fc=color))
    ax.scatter(points[:, 0], points[:, 1], s=2, c=QUERY, zorder=2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("tree")
    ax.set_aspect("equal")
    plt.show(block=False)


def plot_nearest(points, tree, query=(0, 0), cap=16):
    idx, _ = tree.nearest(query, cap=cap)

    _, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], s=2, c=POINT, zorder=2)
    ax.scatter(points[idx, 0], points[idx, 1], s=20, ec=EC, fc=FC, zorder=1)
    for nb in points[idx]:
        ax.plot([query[0], nb[0]], [query[1], nb[1]], c=LINE, lw=0.5, zorder=0)
    ax.scatter(query[0], query[1], s=80, c=QUERY, marker="*", zorder=3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("nearest")
    ax.set_aspect("equal")
    plt.show(block=False)


def plot_radius(points, tree, query=(0, 0), radius=0.2):
    idx, _ = tree.radius(query, radius)

    _, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], s=2, c=POINT, zorder=2)
    ax.scatter(points[idx, 0], points[idx, 1], s=20, ec=EC, fc=FC, zorder=1)
    ax.add_patch(patches.Circle(query, radius, fill=False, linestyle="--", zorder=3))
    ax.scatter(query[0], query[1], s=80, c=QUERY, marker="*", zorder=4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("radius")
    ax.set_aspect("equal")
    plt.show(block=False)


def plot_pairs(points, tree, radius=0.05):
    pairs = tree.pairs(radius)

    _, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], s=2, c=POINT, zorder=2)
    for i, j in pairs:
        ax.plot(
            [points[i, 0], points[j, 0]],
            [points[i, 1], points[j, 1]],
            c=LINE,
            lw=0.5,
            zorder=1,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("pairs")
    ax.set_aspect("equal")
    plt.show(block=False)


def plot_counts(tree, max_radius=1):
    radii = np.linspace(0, max_radius, 100)
    shell = tree.counts(radii, cumulative=False)
    cumul = tree.counts(radii, cumulative=True)

    _, ax1 = plt.subplots()
    ax1.bar(
        radii,
        shell,
        width=radii[1] - radii[0],
        align="edge",
        color=BAR,
        alpha=0.5,
        label="per shell",
    )
    ax2 = ax1.twinx()
    ax2.plot(radii, cumul, c=LINE, label="cumulative")
    ax1.set_xlabel("radius")
    ax1.set_ylabel("pairs per shell")
    ax2.set_ylabel("cumulative pairs")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    ax1.set_title("counts")
    ax1.set_box_aspect(1)
    plt.show(block=False)


def main():
    # points = create_uniform()
    # points = create_meshgrid()
    points = create_clusters()

    tree = KDTree(points, leaf_size=8)
    dump = load_dump(tree)

    plot_tree(points, dump)
    plot_nearest(points, tree)
    plot_radius(points, tree)
    plot_pairs(points, tree)
    plot_counts(tree)

    plt.show()


if __name__ == "__main__":
    main()
