import unittest
import numpy as np
from kdtree.build_kdtree import build_kdtree
from kdtree.nearest_neighbour_search import nearest_neighbour_search
from kdtree.kd_node import KDNode
from examples.hypercube_points import hypercube_points

class TestKDTree(unittest.TestCase):
    def setUp(self):
        """
        Set up test data.
        """
        self.cube_size = 10.0
        self.num_dimensions = 2

    def test_build_kdtree(self):
        """
        Test that KD-Tree is built correctly for different cases.

        Cases:
        - Empty points list.
        - Positive depth value.
        - Negative depth value.
        """
        test_cases = [
            (0, self.cube_size, self.num_dimensions, 0, None),  # Empty points list
            (10, self.cube_size, self.num_dimensions, 2, KDNode),  # Depth = 2, 2D points
            (10, self.cube_size, 3, -2, KDNode),  # Depth = -2, 3D points
        ]

        for num_points, cube_size, num_dimensions, depth, expected_result in test_cases:
            with self.subTest(num_points = num_points, cube_size = cube_size, num_dimensions = num_dimensions, depth = depth, expected_result = expected_result):
                points = (
                    hypercube_points(num_points, cube_size, num_dimensions).tolist()
                    if num_points > 0
                    else []
                )

                kdtree = build_kdtree(points, depth = depth)

                if expected_result is None:
                    # Empty points list case
                    self.assertIsNone(kdtree, f"Expected None for empty points list, got {kdtree}")
                else:
                    # Check if root node is not None
                    self.assertIsNotNone(kdtree, "Expected a KDNode, got None")

                    # Check if root has correct dimensions
                    self.assertEqual(len(kdtree.point), num_dimensions, f"Expected point dimension {num_dimensions}, got {len(kdtree.point)}")

                    # Check that the tree is balanced to some extent (simplistic check)
                    self.assertIsInstance(kdtree, KDNode, f"Expected KDNode instance, got {type(kdtree)}")

    def test_nearest_neighbour_search(self):
        """
        Test the nearest neighbor search function.
        """
        num_points = 20
        cube_size = 15.0
        num_dimensions = 5
        points = hypercube_points(num_points, cube_size, num_dimensions)
        kdtree = build_kdtree(points.tolist())

        rng = np.random.default_rng()
        query_point = rng.random(self.num_dimensions).tolist()

        nearest_point, nearest_dist, nodes_visited = nearest_neighbour_search(kdtree, query_point)

        # Check that nearest point is not None
        self.assertIsNotNone(nearest_point)

        # Check that distance is a non-negative number
        self.assertGreaterEqual(nearest_dist, 0)

        # Check that nodes visited is a non-negative integer
        self.assertGreaterEqual(nodes_visited, 0)

    def test_edge_cases(self):
        """
        Test edge cases such as an empty KD-Tree.
        """
        empty_kdtree = build_kdtree([])
        query_point = [0.0] * self.num_dimensions

        nearest_point, nearest_dist, nodes_visited = nearest_neighbour_search(empty_kdtree, query_point)

        # With an empty KD-Tree, nearest_point should be None
        self.assertIsNone(nearest_point)
        self.assertEqual(nearest_dist, float("inf"))
        self.assertEqual(nodes_visited, 0)


if __name__ == "__main__":
    unittest.main()
