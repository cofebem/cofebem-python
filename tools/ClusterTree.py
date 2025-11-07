"""
Cluster tree implementation for hierarchical matrix construction using PCA clustering.


Author: Vladislav A. Yastrebov
Affiliation: CNRS, Mines Paris, France
Created: May 2024
License: BSD 3-Clause License
"""
import numpy as np
import matplotlib.pyplot as plt

# from mpi4py import MPI
# from petsc4py import PETSc
# from numba import jit, prange


def tree_id_to_int(cl_id):
    return int(cl_id, 2)

class Cluster:
    def __init__(self, cl_id, ids, coords, parent=None):
        # Check that coords and ids have the same length
        assert len(ids) == len(coords)
        # Check that all coords have the same dimension
        assert (len(coords[0]) == 3 or len(coords[0]) == 2)

        self.children = None
        self.parent = parent 
        self.cl_id = cl_id       
        self.ids = ids
        self.coords = coords
        self.dim = len(coords[0])
        self.length = len(ids)
        self.maxx = np.max(coords[:,0])
        self.minx = np.min(coords[:,0])
        self.maxy = np.max(coords[:,1])
        self.miny = np.min(coords[:,1])
        if self.dim == 3:
            self.maxz = np.max(coords[:,2])
            self.minz = np.min(coords[:,2])

            self.size = np.sqrt((self.maxx - self.minx)**2 + (self.maxy - self.miny)**2 + (self.maxz - self.minz)**2)
            self.center = np.array([np.mean(self.coords[:,0]), np.mean(self.coords[:,1]), np.mean(self.coords[:,2])])
            assert self.center[2] >= self.minz and self.center[2] <= self.maxz
        elif self.dim == 2:
            self.size = np.sqrt((self.maxx - self.minx)**2 + (self.maxy - self.miny)**2)
            self.center = np.array([np.mean(self.coords[:,0]), np.mean(self.coords[:,1])])
        assert self.center[0] >= self.minx and self.center[0] <= self.maxx
        assert self.center[1] >= self.miny and self.center[1] <= self.maxy

        # self.cluster = None

    def left(self):
        if self.children is None:
            return None
        assert isinstance(self.children[0], Cluster)
        return self.children[0]
    def right(self):
        if self.children is None:
            return None
        assert isinstance(self.children[1], Cluster)
        return self.children[1]
    def add_child(self, cluster):
        if self.children is None:
            self.children = [cluster]
        else:
            self.children.append(cluster)
    def add_children(self, clusters):
        if self.children is None:
            self.children = clusters
        else:
            self.children.extend(clusters)
    def size():
        return self.size
    def center():
        return self.center
    
    def dist_to_cluster(self, cluster):
        dx = max(0, max(self.minx - cluster.maxx, cluster.minx - self.maxx))
        dy = max(0, max(self.miny - cluster.maxy, cluster.miny - self.maxy))
        dz = max(0, max(self.minz - cluster.maxz, cluster.minz - self.maxz))
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
class ClusterTree:
    def __init__(self, coords):
        self.coords = coords
        self.ids = np.arange(len(coords))
        self.clusters = {}  # Dict to store clusters at each level        
        self.depth = 0
        self.hierarchical_matrix = []
        self.h_matrix_value = []
    def get_minimal_depth(self,level):
        for cl in self.clusters[level]:
            if cl.children is None:
                return level
        for cl in self.clusters[level]:
            return self.get_minimal_depth(level+1)            
    def mergeup_clusters(self, level):
        for cl in self.clusters[level]:
            if cl.children is None:
                continue
            for child in cl.children:
                child.cluster = cl
            cl.children = None
        # self.clusters.pop(level)
        for l in range(level+1, self.depth+1):
            del self.clusters[l]
        self.depth = level
    def recursive_pca_clustering(self, cluster, min_cluster_size, level):
        if cluster.length < min_cluster_size:
            if self.clusters.get(level) is None:
                self.clusters[level] = []
            self.clusters[level].append(cluster)
            self.depth = max(self.depth, level)
        else:
            # Compute the mean of the data
            node_coords = cluster.coords
            node_ids = cluster.ids
            assert len(node_coords) == len(node_ids)
            assert node_coords.all() == self.coords[node_ids].all()

            mean = np.mean(node_coords, axis=0)
            # Compute the covariance matrix
            cov = np.cov(node_coords, rowvar=False)

            # Compute the eigenvalues and eigenvectors of the covariance matrix
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Sort the eigenvectors by decreasing eigenvalues
            indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[indices]
            eigenvectors = eigenvectors[:, indices]

            # Project the data onto the principal components
            coords_pca = np.dot(node_coords - mean, eigenvectors)

            cluster1_id = np.where(coords_pca[:, 0] > 0)[0]
            cluster2_id = np.where(coords_pca[:, 0] <= 0)[0]

            cluster1_ids = node_ids[cluster1_id]
            cluster1_coords = node_coords[cluster1_id]
            assert cluster1_coords.all() == self.coords[cluster1_ids].all()
            c1id = cluster.cl_id + "0"
            cl1 = Cluster(c1id, cluster1_ids, cluster1_coords, cluster)

            cluster2_ids = node_ids[cluster2_id]
            cluster2_coords = node_coords[cluster2_id]
            assert cluster2_coords.all() == self.coords[cluster2_ids].all()
            c2id = cluster.cl_id + "1"
            cl2 = Cluster(c2id, cluster2_ids, cluster2_coords, cluster)
            

            # Recursively cluster each sub-cluster
            # self.clusters[level].append(cl1)
            # self.clusters[level].append(cl2)
            if len(cluster1_ids) >= min_cluster_size and len(cluster2_ids) >= min_cluster_size:
                cluster.add_children([cl1, cl2])
                if self.clusters.get(level) is None :
                    self.clusters[level] = []
                    self.clusters[level].append(cluster)
                else:                
                    self.clusters[level].append(cluster)
                self.recursive_pca_clustering(cl1, min_cluster_size, level + 1)
                self.recursive_pca_clustering(cl2, min_cluster_size, level + 1)
            else:
                self.depth = max(self.depth, level)
                if self.clusters.get(level) is None :
                    self.clusters[level] = []
                    self.clusters[level].append(cluster)
                else:                
                    self.clusters[level].append(cluster)                

    def run_clustering(self, min_cluster_size):
        node_ids = np.arange(len(self.coords))  # Assign node IDs
        cl0 = Cluster("0",node_ids, self.coords, None)
        self.recursive_pca_clustering(cl0, min_cluster_size, 0)
        return self.clusters
    
    def are_descendants(self, cl1, cl2):
        # Simplify descendant checking
        current = cl2
        while current:
            if current == cl1:
                return True
            current = current.parent
        return False

    def h_matrix_vector_product(self, x):
        y = np.zeros_like(x)
        for hmatrix,structure in zip(self.h_matrix_value, self.hierarchical_matrix):
            type_interaction, cl1, cl2 = structure
            type,data = hmatrix
            if type == 'Direct':
                K_block = data
                y[np.ix_(cl1.ids)] += np.dot(K_block, x[cl2.ids])
            elif type == 'ACA':
                U,V = data
                y[np.ix_(cl1.ids)] += np.dot(U, np.dot(V, x[cl2.ids]))
        return y

    def construct_hmatrix(self, cl1, cl2, eta):
        # if self.are_descendants(cl1, cl2):
        #     return  # Do not process if one is a descendant of the other
        assert isinstance(cl1, Cluster)
        assert isinstance(cl2, Cluster)
        if cl1 is not cl2:
            dist = cl1.dist_to_cluster(cl2)
            max_size = max(cl1.size, cl2.size)
            if max_size < 1.e-10 * dist:
                raise ValueError("Should not be here. Finishing.")
                self.NoInteractionBlock(cl1, cl2)
            elif max_size < eta * dist:
                self.CompressBlock(cl1, cl2)  # Compressible
            else:
                if cl1.children is None or cl2.children is None:
                    self.DirectComputeBlock(cl1, cl2)  # Directly compute as no further subdivision possible
                else:
                    # Recursively process children clusters
                    self.construct_hmatrix(cl1.left(), cl2.left(), eta)
                    self.construct_hmatrix(cl1.left(), cl2.right(), eta)
                    self.construct_hmatrix(cl1.right(), cl2.left(), eta)
                    self.construct_hmatrix(cl1.right(), cl2.right(), eta)
        else:
            # Self interactions, check if further division is possible
            if cl1.children:
                self.construct_hmatrix(cl1.left(), cl1.left(), eta)
                self.construct_hmatrix(cl1.left(), cl1.right(), eta)
                self.construct_hmatrix(cl1.right(), cl1.right(), eta)

    def NoInteractionBlock(self, cl1, cl2):
        # Placeholder for no interaction logic
        self.hierarchical_matrix.append(('No Interaction', cl1, cl2))
        
    def CompressBlock(self, cl1, cl2):
        # Placeholder for compression logic
        self.hierarchical_matrix.append(('Compress', cl1, cl2))

    def DirectComputeBlock(self, cl1, cl2):
        # Placeholder for direct computation logic
        self.hierarchical_matrix.append(('Direct', cl1, cl2))

    def build_hierarchical_matrix(self, eta):
        # Initiates matrix construction from the root level
        root = self.clusters[0][0]  # Assumes level 0 is the root level
        assert isinstance(root, Cluster)
        self.construct_hmatrix(root, root, eta)

    def get_dof_indices(self, cl_id):
        cluster = self.clusters[0][0]
        for i,code in enumerate(cl_id):
            if cluster.children is None:
                break
            cluster = cluster.children[int(code)]
        return cluster.ids

    def get_block(self, cl1, cl2, matrix):
        # Get the block corresponding to the interaction between cl1 and cl2
        if cl1 is cl2:
            ids = cl1.ids
            return matrix[np.ix_(ids,ids)]
        else:
            return matrix[np.ix_(cl1.ids, cl2.ids)]

    def plot_hierarchical_matrix(self, filename, matrix, palette, value_min=0, value_max=0):
        # Determine the sizes of blocks at each level
        matrix = np.abs(matrix)
        max_level = self.depth
        block_sizes = [2**(max_level - level) for level in range(max_level + 1)]

        if value_min == 0 and value_max == 0:
            log_matrix = np.log(np.abs(matrix) + 1e-15)             
            mean_val = np.mean(log_matrix)
            std_val = np.std(log_matrix)
            value_min = mean_val - 1 * std_val
            value_max = mean_val + 1 * std_val

        # Create a matrix to represent the types of interactions
        num_clusters = 2**max_level
        matrix_visual = np.full((num_clusters, num_clusters), -1)  # Initialize with -1 for no interaction
        # Set to 2 for self-interaction
        matrix_visual[np.diag_indices(num_clusters)] = 2
        # Plotting the matrix
        fig, ax = plt.subplots(figsize=(10, 10))

        # Fill the matrix based on the type of interaction
        for ci in range(num_clusters):
            ax.add_patch(plt.Rectangle((ci-0.5, ci-0.5), 1, 1, fill=False, edgecolor='black', lw=0.5, zorder=2))
            coefficients = self.get_dof_indices(self.clusters[self.depth][ci].cl_id)
            ax.imshow(np.log(matrix[np.ix_(coefficients,coefficients)]), vmin = value_min, vmax = value_max, cmap=palette, alpha=1, extent=(ci-0.5, ci+0.5, ci-0.5, ci+0.5), zorder=1)
        for interaction in self.hierarchical_matrix:
            type_interaction, cl1, cl2 = interaction
            level = len(cl1.cl_id) - 1  # Assume level is deduced from the length of cl_id
            idx1 = tree_id_to_int(cl1.cl_id) * block_sizes[level]
            idx2 = tree_id_to_int(cl2.cl_id) * block_sizes[level]
            block_size = block_sizes[level]

            # Draw rectangles to represent the blocks
            ax.add_patch(plt.Rectangle((idx2-0.5, idx1-0.5), block_size, block_size, fill=False, edgecolor='black', lw=0.5, zorder=2))       
            ax.add_patch(plt.Rectangle((idx1-0.5, idx2-0.5), block_size, block_size, fill=False, edgecolor='black', lw=0.5, zorder=2))

            coefficients_x = self.get_dof_indices(cl1.cl_id)
            coefficients_y = self.get_dof_indices(cl2.cl_id)

            ax.imshow(np.log(matrix[np.ix_(coefficients_x,coefficients_y)]), cmap=palette, alpha=1, extent=(idx1-0.5, idx1+block_size-0.5, idx2-0.5, idx2+block_size-0.5), zorder=1, vmin= value_min, vmax = value_max)
            ax.imshow(np.log(matrix[np.ix_(coefficients_y,coefficients_x)]), cmap=palette, alpha=1, extent=(idx2-0.5, idx2+block_size-0.5, idx1-0.5, idx1+block_size-0.5), zorder=1, vmin= value_min, vmax = value_max)


            # Update the correct block in the matrix
            matrix_visual[idx1:idx1 + block_size, idx2:idx2 + block_size] = 1
            matrix_visual[idx2:idx2 + block_size, idx1:idx1 + block_size] = 1

        cax = ax.matshow(matrix_visual, cmap=palette, vmin=0, vmax=2, alpha=0.)  # Set vmin and vmax for color consistency

        # Set labels
        ax.set_xlabel('Cluster Index')
        ax.set_ylabel('Cluster Index')
        ax.set_title('Hierarchical Matrix')

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        fig.tight_layout()
        fig.savefig(filename)
        # plt.show()

    def plot_coarse_hierarchical_matrix_structure(self, filename, min_size):
        max_level = self.depth
        block_sizes = [2**(max_level - level) for level in range(max_level + 1)]
        CT = {0: 'white', 1: 'skyblue', 2: 'salmon'}
        
        num_clusters = 2**max_level
        fig, ax = plt.subplots(figsize=(10, 10))

        # Fill the matrix based on the type of interaction
        print("num_clusters = ", num_clusters)
        ax.set_xlim(num_clusters-0.5, -0.5)  # Invert x axis
        ax.set_ylim(-0.5, num_clusters-0.5)  # Regular y axis
        # ax.add_patch(plt.Rectangle((-0.5, -0.5), num_clusters, num_clusters, facecolor=CT[2], edgecolor='black', lw=0.5, zorder=0))
        if min_size == 0:
            for ci in range(num_clusters):
                ax.add_patch(plt.Rectangle((ci-0.5, ci-0.5), 1, 1, facecolor=CT[2], edgecolor='black', lw=0.5, zorder=2))
        for interaction in self.hierarchical_matrix:
            type_interaction, cl1, cl2 = interaction
            level = len(cl1.cl_id) - 1  # Assume level is deduced from the length of cl_id
            idx1 = tree_id_to_int(cl1.cl_id) * block_sizes[level]
            idx2 = tree_id_to_int(cl2.cl_id) * block_sizes[level]
            block_size = block_sizes[level]

            if type_interaction == 'No Interaction':
                value = 0
            elif type_interaction == 'Compress':
                value = 1
            elif type_interaction == 'Direct':
                value = 2
            ct = CT[value]

            # Draw rectangles to represent the blocks
            if False or block_size > min_size:
                ax.add_patch(plt.Rectangle((idx2-0.5, idx1-0.5), block_size, block_size, facecolor=ct, edgecolor='black', lw=0.5, zorder=2))       
                ax.add_patch(plt.Rectangle((idx1-0.5, idx2-0.5), block_size, block_size, facecolor=ct, edgecolor='black', lw=0.5, zorder=2))

        # Set labels
        ax.set_xlabel('Cluster Index')
        ax.set_ylabel('Cluster Index')
        ax.set_title('Hierarchical Matrix Structure')

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        fig.tight_layout()
        fig.savefig(filename)
        plt.show()

    def plot_hierarchical_matrix_structure(self, filename):
        palette = "viridis"
        # Determine the sizes of blocks at each level
        max_level = self.depth
        block_sizes = [2**(max_level - level) for level in range(max_level + 1)]
        CT = {0: 'white', 1: 'skyblue', 2: 'salmon'}
        
        num_clusters = 2**max_level
        matrix_visual = np.full((num_clusters, num_clusters), -1)  # Initialize with -1 for no interaction
        # Set to 2 for self-interaction
        matrix_visual[np.diag_indices(num_clusters)] = 2
        # Plotting the matrix
        fig, ax = plt.subplots(figsize=(10, 10))

        # Fill the matrix based on the type of interaction
        for ci in range(num_clusters):
            ax.add_patch(plt.Rectangle((ci-0.5, ci-0.5), 1, 1, facecolor=CT[2], edgecolor='black', lw=0.5, zorder=2))
        for interaction in self.hierarchical_matrix:
            type_interaction, cl1, cl2 = interaction
            level = len(cl1.cl_id) - 1  # Assume level is deduced from the length of cl_id
            idx1 = tree_id_to_int(cl1.cl_id) * block_sizes[level]
            idx2 = tree_id_to_int(cl2.cl_id) * block_sizes[level]
            block_size = block_sizes[level]

            if type_interaction == 'No Interaction':
                value = 0
            elif type_interaction == 'Compress':
                value = 1
            elif type_interaction == 'Direct':
                value = 2
            ct = CT[value]

            # Draw rectangles to represent the blocks
            ax.add_patch(plt.Rectangle((idx2-0.5, idx1-0.5), block_size, block_size, facecolor=ct, edgecolor='black', lw=0.5, zorder=2))       
            ax.add_patch(plt.Rectangle((idx1-0.5, idx2-0.5), block_size, block_size, facecolor=ct, edgecolor='black', lw=0.5, zorder=2))

            # Update the correct block in the matrix
            matrix_visual[idx1:idx1 + block_size, idx2:idx2 + block_size] = value
            matrix_visual[idx2:idx2 + block_size, idx1:idx1 + block_size] = value

        cax = ax.matshow(matrix_visual, cmap='viridis', vmin=0, vmax=2, alpha=0.)  # Set vmin and vmax for color consistency

        # Set labels
        ax.set_xlabel('Cluster Index')
        ax.set_ylabel('Cluster Index')
        ax.set_title('Hierarchical Matrix Structure')

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        fig.tight_layout()
        fig.savefig(filename)
        # plt.show()
