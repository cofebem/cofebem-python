import os
import numpy as np
import meshio

from cofebem.hmatrices.cluster_tree import ClusterTree


# ---------------------------
# Frontier extraction (tree cut)
# ---------------------------
def clusters_at_order(root, order: int):
    """
    Frontier nodes: nodes at depth==order OR leaves encountered earlier.
    """
    frontier = []

    def dfs(node):
        if node is None:
            return
        if node.is_leaf or node.level == order:
            frontier.append(node)
            return
        dfs(node.left)
        dfs(node.right)

    dfs(root)
    return frontier


def point_cluster_tags_for_order(
    cluster_tree, order: int, start_id: int = 1, dtype=np.int32
):
    """
    tags[p] = cluster id for point p (categorical labels).
    """
    npts = cluster_tree.pts.shape[0]
    tags = np.zeros(npts, dtype=dtype)

    frontier = clusters_at_order(cluster_tree.root, order)

    for k, cl in enumerate(frontier, start=start_id):
        tags[cl.idx] = k

    # Safety (shouldn't happen unless something degenerate/unassigned)
    if np.any(tags == 0):
        tags[tags == 0] = -1

    return tags, frontier


# ---------------------------
# Cell tagging: majority vote
# ---------------------------
def cells_cluster_tags_from_point_tags_majority(
    cells: np.ndarray, point_tags: np.ndarray
):
    """
    Force each cell to belong to ONE cluster:
    take the cluster_id that appears most among the cell vertices.

    cells: (ncell, nv)
    point_tags: (npoints,)
    returns: (ncell,)
    """
    cell_pt_tags = point_tags[cells]  # (ncell, nv)
    out = np.empty(cell_pt_tags.shape[0], dtype=np.int32)

    # Majority label per cell
    for i, tags in enumerate(cell_pt_tags):
        vals, counts = np.unique(tags, return_counts=True)
        out[i] = vals[np.argmax(counts)]

    return out


# ---------------------------
# MeshIO export with cell_data
# ---------------------------
def export_cluster_tree_levels_cells_majority(
    mesh: meshio.Mesh,
    cluster_tree,
    out_prefix: str,
    max_order: int = None,
    ext: str = "vtu",
    cell_data_name: str = "cluster_id",
):
    """
    Export one mesh per order, preserving the original cells,
    with cell_data that contains the cluster id per cell block.
    """

    if max_order is None:
        max_order = cluster_tree._max_level()

    # Ensure output directory exists
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    points = mesh.points
    cells = mesh.cells  # list[meshio.CellBlock]

    for order in range(max_order + 1):
        pt_tags, frontier = point_cluster_tags_for_order(cluster_tree, order)

        # Build cell_data per block (meshio expects dict[name] = list aligned with mesh.cells)
        cell_tags_blocks = []
        for cb in cells:
            conn = cb.data
            cell_tags = cells_cluster_tags_from_point_tags_majority(conn, pt_tags)
            cell_tags_blocks.append(cell_tags)

        out = meshio.Mesh(
            points=points,
            cells=cells,
            cell_data={cell_data_name: cell_tags_blocks},
        )

        fn = f"{out_prefix}_level_{order:03d}.{ext}"
        meshio.write(fn, out)
        print(f"[ok] wrote {fn}  frontier clusters={len(frontier)}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    pikachu = meshio.read("./geo_files/Pikachu.msh")
    pts = pikachu.points.astype(np.float64)

    cluster_tree = ClusterTree(pts, leaf_size=64, split="pca")

    export_cluster_tree_levels_cells_majority(
        mesh=pikachu,
        cluster_tree=cluster_tree,
        out_prefix="./results/pikachu_cells_clusters",
        max_order=16,
        ext="vtu",
        cell_data_name="cluster_id",
    )
