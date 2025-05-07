import math
import numpy as np
from abc import ABC, abstractmethod
import meshio


def quat2mat(q):
    """
    Convert a quaternion q = (qw, qx, qy, qz) to a 3x3 rotation matrix.
    Assumes q is normalized or near-normalized.
    """
    qw, qx, qy, qz = q
    qw2 = qw * qw
    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz

    R = np.array(
        [
            [qw2 + qx2 - qy2 - qz2, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), qw2 - qx2 + qy2 - qz2, 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw2 - qx2 - qy2 + qz2],
        ],
        dtype=float,
    )

    return R


class RigidBodyBase(ABC):
    def __init__(self, mesh, q0, name="Rigid Body"):
        if len(q0) != 7:
            raise ValueError("q0 must have length 7: (x, y, z, qw, qx, qy, qz).")

        self.name = name
        self.position = q0[:3]  # (x, y, z)
        self.orientation = q0[3:]  # (qw, qx, qy, qz)
        self.mesh = None  # Will be a meshio.Mesh after building and transforming

        # Build the local mesh (child class must implement build_mesh_local)
        local_mesh = self.build_mesh_local()

        # Transform local mesh to global using self.position & self.orientation
        transformed_points = self.apply_transform(local_mesh.points)

        # Store the transformed mesh
        self.mesh = meshio.Mesh(
            points=transformed_points,
            cells=local_mesh.cells,
            point_data=local_mesh.point_data,
            cell_data=local_mesh.cell_data,
            field_data=local_mesh.field_data,
        )

    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        Apply the rigid transform from self.position and self.orientation
        to an array of local points. Returns the transformed (global) points.
        """
        R = quaternion_to_rotation_matrix(self.orientation)
        return (points @ R.T) + self.position

    def save_mesh(self, filename: str):
        """
        Save current mesh to file using meshio.
        """
        if self.mesh is None:
            raise RuntimeError("No mesh available. Did you build the rigid body?")
        meshio.write(filename, self.mesh)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name}, "
            f"pos={self.position}, quat={self.orientation})"
        )


class Sphere(RigidBody):
    """
    Sphere shape, defined by radius and mesh resolution.
    """

    def __init__(self, name: str, radius: float, q0: np.ndarray, resolution: int = 20):
        self.radius = radius
        self.resolution = resolution
        super().__init__(name, q0)

    def build_mesh_local(self) -> meshio.Mesh:
        """
        Build a sphere in local coords with radius=1, then scale by self.radius.
        """
        vertices = []
        faces = []
        for i in range(self.resolution + 1):
            theta = math.pi * i / self.resolution
            for j in range(self.resolution + 1):
                phi = 2.0 * math.pi * j / self.resolution
                x = math.sin(theta) * math.cos(phi)
                y = math.sin(theta) * math.sin(phi)
                z = math.cos(theta)
                vertices.append([x, y, z])

        def idx(a, b):
            return a * (self.resolution + 1) + b

        for i in range(self.resolution):
            for j in range(self.resolution):
                p1 = idx(i, j)
                p2 = idx(i, (j + 1) % (self.resolution + 1))
                p3 = idx(i + 1, j)
                p4 = idx(i + 1, (j + 1) % (self.resolution + 1))
                # two triangles per quad
                faces.append([p1, p2, p3])
                faces.append([p2, p4, p3])

        vertices = np.array(vertices) * self.radius
        faces = np.array(faces, dtype=np.int32)
        return meshio.Mesh(points=vertices, cells=[("triangle", faces)])


class Cube(RigidBody):
    """
    Cube shape, defined by side length.
    """

    def __init__(self, name: str, side: float, q0: np.ndarray):
        self.side = side
        super().__init__(name, q0)

    def build_mesh_local(self) -> meshio.Mesh:
        """
        Build a cube of side length=1 centered at origin, then scale by self.side.
        """
        s = self.side / 2.0
        vertices = np.array(
            [
                [-s, -s, -s],
                [s, -s, -s],
                [s, s, -s],
                [-s, s, -s],
                [-s, -s, s],
                [s, -s, s],
                [s, s, s],
                [-s, s, s],
            ],
            dtype=float,
        )

        # 12 triangles (2 per face * 6 faces)
        faces = np.array(
            [
                # bottom (z=-s)
                [0, 1, 2],
                [0, 2, 3],
                # top    (z= s)
                [4, 6, 5],
                [4, 7, 6],
                # front  (y= s)
                [3, 2, 6],
                [3, 6, 7],
                # back   (y=-s)
                [1, 0, 4],
                [1, 4, 5],
                # right  (x= s)
                [1, 5, 6],
                [1, 6, 2],
                # left   (x=-s)
                [0, 3, 7],
                [0, 7, 4],
            ],
            dtype=np.int32,
        )

        return meshio.Mesh(points=vertices, cells=[("triangle", faces)])


class SemiSphere(RigidBody):
    """
    SemiSphere (upper hemisphere).
    """

    def __init__(self, name: str, radius: float, q0: np.ndarray, resolution: int = 20):
        self.radius = radius
        self.resolution = resolution
        super().__init__(name, q0)

    def build_mesh_local(self) -> meshio.Mesh:
        """
        Build a hemisphere from theta=0 to pi/2, radius=1 in local coords, then scale.
        """
        vertices = []
        faces = []
        for i in range(self.resolution + 1):
            theta = (math.pi / 2.0) * i / self.resolution
            for j in range(self.resolution + 1):
                phi = 2.0 * math.pi * j / self.resolution
                x = math.sin(theta) * math.cos(phi)
                y = math.sin(theta) * math.sin(phi)
                z = math.cos(theta)
                vertices.append([x, y, z])

        def idx(a, b):
            return a * (self.resolution + 1) + b

        for i in range(self.resolution):
            for j in range(self.resolution):
                p1 = idx(i, j)
                p2 = idx(i, (j + 1) % (self.resolution + 1))
                p3 = idx(i + 1, j)
                p4 = idx(i + 1, (j + 1) % (self.resolution + 1))
                faces.append([p1, p2, p3])
                faces.append([p2, p4, p3])

        vertices = np.array(vertices) * self.radius
        faces = np.array(faces, dtype=np.int32)
        return meshio.Mesh(points=vertices, cells=[("triangle", faces)])


class Cylinder(RigidBody):
    """
    Cylinder shape, defined by radius, height, and resolution.
    """

    def __init__(
        self,
        name: str,
        radius: float,
        height: float,
        q0: np.ndarray,
        resolution: int = 20,
    ):
        self.radius = radius
        self.height = height
        self.resolution = resolution
        super().__init__(name, q0)

    def build_mesh_local(self) -> meshio.Mesh:
        """
        Build a cylinder of given radius, height, oriented along z-axis,
        from z=-h/2 to z=+h/2, plus top/bottom caps.
        """
        top_z = self.height / 2.0
        bot_z = -self.height / 2.0

        vertices = []
        side_faces = []
        top_faces = []
        bottom_faces = []

        # Build top and bottom rings
        for i in range(self.resolution):
            angle = 2.0 * math.pi * i / self.resolution
            x = self.radius * math.cos(angle)
            y = self.radius * math.sin(angle)
            # bottom
            vertices.append([x, y, bot_z])
            # top
            vertices.append([x, y, top_z])

        bottom_center_idx = len(vertices)
        vertices.append([0.0, 0.0, bot_z])

        top_center_idx = len(vertices)
        vertices.append([0.0, 0.0, top_z])

        def ring_idx(i, top=False):
            return 2 * i + (1 if top else 0)

        # side triangles
        for i in range(self.resolution):
            i_next = (i + 1) % self.resolution
            b1 = ring_idx(i, top=False)
            b2 = ring_idx(i_next, top=False)
            t1 = ring_idx(i, top=True)
            t2 = ring_idx(i_next, top=True)

            side_faces.append([b1, b2, t1])
            side_faces.append([b2, t2, t1])

        # top faces (fan)
        for i in range(self.resolution):
            i_next = (i + 1) % self.resolution
            t1 = ring_idx(i, top=True)
            t2 = ring_idx(i_next, top=True)
            top_faces.append([top_center_idx, t1, t2])

        # bottom faces (fan)
        for i in range(self.resolution):
            i_next = (i + 1) % self.resolution
            b1 = ring_idx(i, top=False)
            b2 = ring_idx(i_next, top=False)
            bottom_faces.append([bottom_center_idx, b2, b1])

        vertices = np.array(vertices, dtype=float)
        side_faces = np.array(side_faces, dtype=np.int32)
        top_faces = np.array(top_faces, dtype=np.int32)
        bottom_faces = np.array(bottom_faces, dtype=np.int32)

        # Combine all faces
        all_faces = np.concatenate([side_faces, top_faces, bottom_faces], axis=0)
        return meshio.Mesh(points=vertices, cells=[("triangle", all_faces)])


class Parallelepiped(RigidBody):
    """
    A parallelepiped defined by three edge vectors v1, v2, v3.
    If v1, v2, v3 are linearly independent, this forms a "skewed box".
    """

    def __init__(
        self, name: str, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, q0: np.ndarray
    ):
        """
        :param name: Rigid body name.
        :param v1:   A 3D vector (numpy array).
        :param v2:   A 3D vector (numpy array).
        :param v3:   A 3D vector (numpy array).
        :param q0:   7-element array, position + quaternion orientation.
        """
        # Basic checks
        if not (v1.shape == (3,) and v2.shape == (3,) and v3.shape == (3,)):
            raise ValueError("v1, v2, v3 must be 3-element numpy arrays.")
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        super().__init__(name, q0)

    def build_mesh_local(self) -> meshio.Mesh:
        """
        Build the parallelepiped in local coords, spanned by v1, v2, v3.
        The 8 corners are formed by all combinations of 0 or 1 times the vectors:
            0: (0,0,0)
            1: v1
            2: v2
            3: v3
            4: v1 + v2
            5: v2 + v3
            6: v3 + v1
            7: v1 + v2 + v3
        Then each face is two triangles.
        """
        # List corners
        c0 = np.array([0, 0, 0], dtype=float)
        c1 = self.v1
        c2 = self.v2
        c3 = self.v3
        c4 = self.v1 + self.v2
        c5 = self.v2 + self.v3
        c6 = self.v3 + self.v1
        c7 = self.v1 + self.v2 + self.v3

        vertices = np.array([c0, c1, c2, c3, c4, c5, c6, c7], dtype=float)

        # We'll define 6 faces, each is a quadrilateral => 2 triangles
        # Face1: bottom (0,1,4,2)
        # Face2: face around (0,2,5,3)
        # Face3: face around (0,3,6,1)
        # Face4: top (7,4,1,6)
        # Face5: face around (7,5,2,4)
        # Face6: face around (7,6,3,5)

        faces = [
            [0, 1, 4],
            [0, 4, 2],  # bottom
            [0, 2, 5],
            [0, 5, 3],  # next side
            [0, 3, 6],
            [0, 6, 1],  # next side
            [7, 4, 1],
            [7, 1, 6],  # top
            [7, 5, 2],
            [7, 2, 4],  # next side
            [7, 6, 3],
            [7, 3, 5],  # next side
        ]
        faces = np.array(faces, dtype=np.int32)

        return meshio.Mesh(points=vertices, cells=[("triangle", faces)])


class GeneralRigidBody(RigidBody):
    """
    A general rigid body that loads its geometry from a given mesh file,
    then applies the transform from q0 (position+orientation).
    """

    def __init__(self, name: str, q0: np.ndarray, mesh_file: str):
        self.mesh_file = mesh_file
        super().__init__(name, q0)

    def build_mesh_local(self) -> meshio.Mesh:
        """
        Load the mesh from self.mesh_file (assuming it's in a 'local' coordinate system).
        If you want to recenter or rescale it, do so here.
        """
        local_mesh = meshio.read(self.mesh_file)
        # Optionally modify local_mesh.points if desired
        return local_mesh


# --------------------------------------------------------------------------
# USAGE EXAMPLE
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Suppose we have an initial configuration q0 = position(1,2,3) + quaternion(1,0,0,0)
    q0 = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])

    # 1) Sphere
    sphere_body = Sphere("MySphere", radius=1.0, q0=q0, resolution=8)
    print(sphere_body)
    sphere_body.save_mesh("sphere.vtk")

    # 2) Cube
    cube_body = Cube("MyCube", side=2.0, q0=q0)
    print(cube_body)
    cube_body.save_mesh("cube.vtk")

    # 3) SemiSphere
    semisphere_body = SemiSphere("MySemiSphere", radius=1.5, q0=q0)
    print(semisphere_body)
    semisphere_body.save_mesh("semisphere.vtk")

    # 4) Cylinder
    cylinder_body = Cylinder("MyCylinder", radius=1.0, height=2.5, q0=q0, resolution=12)
    print(cylinder_body)
    cylinder_body.save_mesh("cylinder.vtk")

    # 5) Parallelepiped with custom vectors
    #    For instance, define v1, v2, v3 as edges:
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.5, 1.0, 0.0])
    v3 = np.array([0.0, 0.2, 1.0])
    pp_body = Parallelepiped("MyParallelepiped", v1, v2, v3, q0=q0)
    print(pp_body)
    pp_body.save_mesh("parallelepiped.vtk")

    # 6) General RigidBody from an existing mesh file, e.g. "my_mesh.stl"
    # general_body = GeneralRigidBody("MyGeneralBody", q0, mesh_file="my_mesh.stl")
    # print(general_body)
    # general_body.save_mesh("transformed_output.vtk")
