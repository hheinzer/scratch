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
    bbox_min = data[:, 8 : 8 + dim]
    bbox_max = data[:, 8 + dim :]
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

    fig, ax = plt.subplots(figsize=(7, 7))

    for lo, hi in zip(dump["bbox_min"], dump["bbox_max"]):
        ax.add_patch(patches.Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1], ec="orange", fc="none"))

    ax.scatter(points[:, 0], points[:, 1], s=1, c="purple")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.tight_layout()
    plt.savefig("plot.pdf")


if __name__ == "__main__":
    main()
