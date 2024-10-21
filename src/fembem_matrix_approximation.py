import ClusterTree as ct
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import aca


Test = "Cube"
# For the dragon
# min_cluster_size = 20
# For the cube
min_cluster_size = 5 
Eta = 1.25
FileName = f"{Test}_eta_{Eta}.pdf"
FileNameM = f"{Test}_matrix_eta_{Eta}.pdf"

"""
    Test the ClusterTree on a simple square mesh
"""

if Test == "Cube":
    fname = "out_elasticity/FlexData_30x30.npz"
    data = np.load(fname)
    K, coords, dofs = data["K"],data["coords"],data["dofs"]
    n = np.sqrt(K.shape[0]).astype(int)
                
    # Create a ClusterTree object
    ctree = ct.ClusterTree(coords)
    # Construct clusters
    clusters = ctree.run_clustering(min_cluster_size)
    # Balance the tree
    minimal_depth = ctree.get_minimal_depth(0)
    if minimal_depth != ctree.depth:
        ctree.mergeup_clusters(minimal_depth)
        clusters = ctree.clusters
    ctree.build_hierarchical_matrix(eta=Eta)
    # Plot
    if False:
        ctree.plot_hierarchical_matrix_structure(FileName)
        palette = "inferno"
        value_min = -2 #np.min(np.log(np.abs(K)))
        value_max = np.max(np.log(np.abs(K)))
        ctree.plot_hierarchical_matrix(FileNameM, K, palette, value_min, value_max)
    
    hmatrix = ctree.hierarchical_matrix
    matrix_block = 0
    relative_tolerance = 1e-2
    for block in hmatrix:
        type, cl1, cl2 = block
        if type == "Compress" and min(len(cl1.ids),len(cl2.ids)) > 20:
            print("ACA for ", cl1.cl_id, " and ", cl2.cl_id)
            print("Size cl1 = ", len(cl1.ids), ", size cl2 = ", len(cl2.ids))
            matrix_block = ctree.get_block(cl1, cl2, K)

            U,V,error,rank = aca.aca(matrix_block, relative_tolerance)
            reduced_size  = rank*(matrix_block.shape[0] + matrix_block.shape[1])
            original_size = matrix_block.shape[0]*matrix_block.shape[1]
            reduction = reduced_size/original_size
            print("|R| = ", error, ", rank = ", rank, ", compression = ", reduction)
        elif type == "Compress":
            # print("SVD for ", cl1.cl_id, " and ", cl2.cl_id, " due to small size")
            # print("Size cl1 = ", len(cl1.ids), ", size cl2 = ", len(cl2.ids))
            matrix_block = ctree.get_block(cl1, cl2, K)
            original_size = matrix_block.shape[0]*matrix_block.shape[1]
            
            U, s, V = np.linalg.svd(matrix_block)
            # Truncate to tolerance
            rank = np.sum(s > relative_tolerance*np.max(s))
            U = U[:, :rank]
            V = V[:rank, :]
            s = s[:rank]
            # print("SCF = ", np.sum(s**2), ", rank = ", rank, ", compression = ", rank*(matrix_block.shape[0] + matrix_block.shape[1])/original_size)

    # fig,ax = plt.subplots()
    # matrix_block_approx = U @ V
    # print(U.shape)
    # print(V.shape)

    # matrix_block_approx, ranks, R_norm = aca(matrix_block, 1e-4)
    # fig,ax = plt.subplots()
    # cax = ax.imshow(np.log(np.abs(matrix_block_approx)), cmap="inferno")
    # fig.colorbar(cax)
    # plt.show()

"""
    Test the ClusterTree on a dragon mesh
"""
if Test == "Dragon":
    import collada
    import trimesh
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    dae_name = "mesh/model.dae"

    ## From Collada file to Trimesh object
    dae = collada.Collada(dae_name)
    # Extract geometry data
    vertices = []
    faces = []
    for geometry in dae.geometries:
        for primitive in geometry.primitives:
            vertices.append(primitive.vertex)
            faces.append(primitive.vertex_index.reshape(-1, 3))
    mesh = trimesh.Trimesh(vertices=np.vstack(vertices), faces=np.vstack(faces))

    for vertex in mesh.vertices:
        # check if the vertex has 3 coordinates
        assert len(vertex) == 3
    print("Successfully loaded mesh")

    ctree = ct.ClusterTree(mesh.vertices)
    clusters = ctree.run_clustering(min_cluster_size)
    minimal_depth = ctree.get_minimal_depth(0)
    if minimal_depth != ctree.depth:
        ctree.mergeup_clusters(minimal_depth)
        clusters = ctree.clusters
    ctree.build_hierarchical_matrix(eta=Eta)
    ctree.plot_hierarchical_matrix(FileName)

    SaveMultiLevel = False
    if SaveMultiLevel:
        for level in range(len(clusters)):
            print("Level ", level)
            print("Number of clusters: ", len(clusters[level]))
            p = pv.Plotter()
            for cl in clusters[level]:
                coords = cl.coords
                cloud = pv.PolyData(coords)
                random_color = np.random.rand(3)
                cloud['z'] = np.zeros((len(coords),3))
                cloud['z'][:,2] = random_color[0]

                # surf = cloud.delaunay_2d(alpha=.0)  # Use delaunay_3d instead of delaunay_2d             
                p.add_mesh(cloud, color=random_color)

            p.add_mesh(mesh, show_edges=True, opacity=0.2, edge_color='black')
            p.view_vector((.0, 0.25, -1))
            # p.add_floor('-y', lighting=True, color='#a3a3a3', pad=3)
            # p.enable_shadows()
            p.show()
            p.screenshot("clusters_level_"+str(level)+".png")

