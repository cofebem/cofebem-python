import numpy as np
from cofebem.hmatrices.cluster import ClusterTree, BlockClusterTree
from cofebem.hmatrices.low_rank_approx import truncated_svd


class HMatrix:
    def __init__(
        self, A, coords, compress_func, max_leaf_size=64, eta=1.0, tol=1e-6, max_rank=50
    ):

        self.A = A
        self.coords = coords
        self.compress_func = compress_func
        self.max_leaf_size = max_leaf_size
        self.eta = eta
        self.tol = tol
        self.max_rank = max_rank

        self.row_tree = ClusterTree(coords, max_leaf_size)
        self.col_tree = ClusterTree(coords, max_leaf_size)
        self.block_tree = BlockClusterTree(self.row_tree, self.col_tree, eta)

        self._compress()

    def _compress(self):
        for block in self.block_tree.admissible_blocks:
            block.compress_from_dense(
                self.A, self.tol, self.max_rank, self.compress_func
            )

    def matvec(self, x):
        y = np.zeros(self.A.shape[0])

        def apply_block(block):
            t_ids = block.row_cluster.ids
            s_ids = block.col_cluster.ids
            if block.is_admissible:
                Vx = block.V.T @ x[s_ids]
                y[t_ids] += block.U @ Vx
            elif block.is_leaf:
                A_block = self.A[np.ix_(t_ids, s_ids)]
                y[t_ids] += A_block @ x[s_ids]

        self.block_tree.traverse(apply_block)
        return y

    def __call__(self, x):
        return self.matvec(x)

    def print_summary(self):
        self.block_tree.print_summary()

    def to_dense(self):
        N = self.A.shape[0]
        A_approx = np.zeros((N, N))

        def fill_block(block):
            t_ids = block.row_cluster.ids
            s_ids = block.col_cluster.ids
            if block.is_admissible:
                A_approx[np.ix_(t_ids, s_ids)] = block.U @ block.V.T
            elif block.is_leaf:
                A_approx[np.ix_(t_ids, s_ids)] = self.A[np.ix_(t_ids, s_ids)]

        self.block_tree.traverse(fill_block)
        return A_approx

    def visualize(self, filename=None):
        import matplotlib.pyplot as plt

        # import matplotlib.patches as patches

        def tree_id_to_int(cl_id):
            return int(cl_id, 2)

        CT = {0: "white", 1: "skyblue", 2: "salmon"}

        all_blocks = self.block_tree.blocks
        max_level = max(len(block.row_cluster.cl_id) for block in all_blocks)
        block_sizes = [2 ** (max_level - l) for l in range(max_level + 1)]

        num_clusters = 2**max_level
        matrix_visual = np.full((num_clusters, num_clusters), -1)
        matrix_visual[np.diag_indices(num_clusters)] = 2

        fig, ax = plt.subplots(figsize=(10, 10))

        for ci in range(num_clusters):
            ax.add_patch(
                plt.Rectangle(
                    (ci - 0.5, ci - 0.5),
                    1,
                    1,
                    facecolor=CT[2],
                    edgecolor="black",
                    lw=0.5,
                    zorder=1,
                )
            )

        for block in all_blocks:
            cl1 = block.row_cluster
            cl2 = block.col_cluster
            level = len(cl1.cl_id) - 1
            block_size = block_sizes[level]

            idx1 = tree_id_to_int(cl1.cl_id) * block_size
            idx2 = tree_id_to_int(cl2.cl_id) * block_size

            if block.is_admissible:
                value = 1  # Compressed
            elif block.is_leaf:
                value = 2  # Dense
            else:
                value = 0  # No interaction (shouldn't happen)

            matrix_visual[idx1 : idx1 + block_size, idx2 : idx2 + block_size] = value
            matrix_visual[idx2 : idx2 + block_size, idx1 : idx1 + block_size] = value

            ct = CT[value]

            ax.add_patch(
                plt.Rectangle(
                    (idx2 - 0.5, idx1 - 0.5),
                    block_size,
                    block_size,
                    facecolor=ct,
                    edgecolor="black",
                    lw=0.5,
                    zorder=2,
                )
            )
            if idx1 != idx2:
                ax.add_patch(
                    plt.Rectangle(
                        (idx1 - 0.5, idx2 - 0.5),
                        block_size,
                        block_size,
                        facecolor=ct,
                        edgecolor="black",
                        lw=0.5,
                        zorder=2,
                    )
                )

        ax.matshow(matrix_visual, cmap="viridis", vmin=0, vmax=2, alpha=0.0)

        # Decorate
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Cluster Index")
        ax.set_ylabel("Cluster Index")
        ax.set_title("Hierarchical Matrix Structure")
        ax.set_aspect("equal")
        fig.tight_layout()

        if filename:
            fig.savefig(filename, dpi=300)
        else:
            plt.show()

        plt.close()
