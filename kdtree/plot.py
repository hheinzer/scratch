import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from kdtree import KDTree


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


def main():
    rng = np.random.default_rng(0)

    centers = rng.uniform(-0.7, 0.7, (10, 2))
    cluster = rng.integers(0, 10, 1000)
    points = centers[cluster] + rng.normal(0, 0.1, (1000, 2))

    tree = KDTree(points, leaf_size=8)
    dump = load_dump(tree)

    _, ax = plt.subplots()

    depth = dump["depth"]
    cmap = plt.get_cmap("Oranges")
    for lo, hi, dep in zip(dump["bbox_min"], dump["bbox_max"], depth):
        color = cmap(0.5 * (dep - depth.min()) / max(depth.max() - depth.min(), 1))
        ax.add_patch(patches.Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1], fc=color))

    ax.scatter(points[:, 0], points[:, 1], s=2, c="purple", zorder=2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
