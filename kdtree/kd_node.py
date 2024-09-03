from __future__ import annotations

class KDNode:
    """
    Represents a node in a KD-Tree.

    Attributes:
        point: The point stored in this node.
        left: The left child node.
        right: The right child node.
    """
    def __init__(self, point: list[float], left: KDNode | None = None, right: KDNode | None = None) -> None:
        self.point = point
        self.left = left
        self.right = right
