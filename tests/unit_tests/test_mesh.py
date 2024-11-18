import unittest
import numpy as np
from cofebem.mesh.mesh import Mesh


class TestMesh(unittest.TestCase):
    def setUp(self):
        """
        Set up a Mesh instance for testing.
        """
        self.mesh_file = "test_mesh.msh"  # Replace with the path to a valid mesh file
        self.mesh = Mesh(self.mesh_file)

    def test_mesh_initialization(self):
        """
        Test if the Mesh object initializes correctly from a file.
        """
        self.assertIsInstance(self.mesh, Mesh)
        self.assertIsNotNone(self.mesh.nodes)
        self.assertIsInstance(self.mesh.nodes, np.ndarray)
        self.assertGreater(
            len(self.mesh.nodes), 0, "Mesh should have at least one node."
        )

    def test_elements_loading(self):
        """
        Test if elements are loaded correctly from the mesh file.
        """
        self.assertIsInstance(self.mesh.elements, dict)
        self.assertGreater(
            len(self.mesh.elements), 0, "Mesh should have at least one element type."
        )
        for element_type, elements in self.mesh.elements.items():
            self.assertIsInstance(elements, np.ndarray)
            self.assertGreater(
                len(elements), 0, f"Element type {element_type} should have elements."
            )

    def test_boundary_entities(self):
        """
        Test if boundary entities are identified correctly.
        """
        self.mesh.get_boundary_entities()
        self.assertIsInstance(self.mesh.boundary_elements, dict)
        for element_type, boundary_elements in self.mesh.boundary_elements.items():
            self.assertIsInstance(boundary_elements, np.ndarray)

    def test_build_topology(self):
        """
        Test if the topology is built correctly.
        """
        self.mesh.build_topology()
        self.assertIsInstance(self.mesh.edges, list)
        self.assertGreater(len(self.mesh.edges), 0, "Edges should be populated.")
        self.assertIsInstance(self.mesh.element_neighbors, dict)

    def test_compute_element_areas(self):
        """
        Test area computation for 2D meshes.
        """
        if self.mesh.dimension == 2:
            self.mesh.compute_element_areas()
            self.assertIsInstance(self.mesh.element_areas, np.ndarray)
            self.assertGreater(
                len(self.mesh.element_areas), 0, "Element areas should be populated."
            )

    def test_compute_element_volumes(self):
        """
        Test volume computation for 3D meshes.
        """
        if self.mesh.dimension == 3:
            self.mesh.compute_element_volumes()
            self.assertIsInstance(self.mesh.element_volumes, np.ndarray)
            self.assertGreater(
                len(self.mesh.element_volumes),
                0,
                "Element volumes should be populated.",
            )

    def test_write_mesh(self):
        """
        Test if the mesh can be written to a file.
        """
        output_file = "output_mesh.msh"
        self.mesh.write_mesh(output_file)
        # Validate that the output file exists and has content
        try:
            with open(output_file, "r") as f:
                content = f.read()
                self.assertGreater(
                    len(content), 0, "Output mesh file should not be empty."
                )
        finally:
            import os

            if os.path.exists(output_file):
                os.remove(output_file)  # Clean up the test output

    def test_translate_mesh(self):
        """
        Test if the mesh translation works correctly.
        """
        original_nodes = self.mesh.nodes.copy()
        translation_vector = np.array([1.0, 1.0, 0.0])
        self.mesh.translate(translation_vector)
        self.assertTrue(
            np.allclose(self.mesh.nodes, original_nodes + translation_vector),
            "Mesh nodes should be translated correctly.",
        )

    def test_scale_mesh(self):
        """
        Test if the mesh scaling works correctly.
        """
        original_nodes = self.mesh.nodes.copy()
        scale_factor = 2.0
        self.mesh.scale(scale_factor)
        self.assertTrue(
            np.allclose(self.mesh.nodes, original_nodes * scale_factor),
            "Mesh nodes should be scaled correctly.",
        )

    def test_visualization(self):
        """
        Test if the visualization function works without error.
        """
        try:
            self.mesh.visualize()
        except ImportError:
            self.fail("Visualization requires additional libraries (e.g., PyVista).")

    def test_metadata(self):
        """
        Test if metadata is correctly populated.
        """
        self.assertIsInstance(self.mesh.metadata, dict)
        self.assertIn(
            "meshio", self.mesh.metadata, "Metadata should contain 'meshio' key."
        )


if __name__ == "__main__":
    unittest.main()
