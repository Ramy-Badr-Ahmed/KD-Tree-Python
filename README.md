![Python](https://img.shields.io/badge/Python-3670A0?style=plastic&logo=python&logoColor=ffdd54) ![NumPy](https://img.shields.io/badge/Numpy-777BB4.svg?style=plastic&logo=numpy&logoColor=white) ![GitHub](https://img.shields.io/github/license/Ramy-Badr-Ahmed/KD-Tree-Python?style=plastic&cached)

# KD-Tree Implementation in Python

This repository contains Python implementation of the kd-tree data structure and performing k-nearest neighbour search.

The Matlab lab implementation is located here: [KD-Tree-Matlab](https://github.com/Ramy-Badr-Ahmed/KD-Tree-Matlab)

### About
The kd-tree is a space-partitioning data structure for organizing points in a k-dimensional space.

> [Mathematica Link](https://reference.wolfram.com/language/ref/datastructure/KDTree.html)

### Scripts

1. `build_kdtree.py`

   > Builds a kd-tree from a set of points.

2. `nearest_neighbour_search.py`

   > Performs nearest neighbour search using the built kd-tree.

3. `hypercube_points.py`

   > Generates n-Dimensional Points Uniformly in an n-Dimensional Hypercube.

### Example Usage

```python
from kdtree.build_kdtree import build_kdtree
from kdtree.nearest_neighbor_search import nearest_neighbor_search
from examples.hypercube_points import hypercube_points

num_points = 5000
cube_size = 10
num_dimensions = 10

points = hypercube_points(num_points, cube_size, num_dimensions)
hypercube_kdtree = build_kdtree(points.tolist())

query_point = np.random.rand(num_dimensions).tolist()

nearest_point, nearest_dist, nodes_visited = nearest_neighbor_search(hypercube_kdtree, query_point)

print(f"Query point: {query_point}")
print(f"Nearest point: {nearest_point}")
print(f"Distance: {nearest_dist:.4f}")
print(f"Nodes visited: {nodes_visited}")
