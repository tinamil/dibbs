from __future__ import annotations
from numba import njit
import numpy as np
from collections import defaultdict
from typing import Optional, Callable, Tuple, List, Dict, Set
import math
import rubiks_optimized as ro


class Node:
    def __init__(self, parent: Optional[Node], state: np.ndarray, face: Optional[int], rotation: Optional[int], cost: int, heuristic_func, other_heuristic):
        self.parent = parent
        self.state = state
        self.face = face
        self.rotation = rotation
        self.cost = cost
        self.heuristic = heuristic_func(self.state)
        self.combined = self.cost + self.heuristic
        self.reverse_heuristic = other_heuristic(self.state)
        self.pr = max(self.combined, 2 * self.cost)
        if self.reverse_heuristic > self.cost:
            print(self.reverse_heuristic, self.cost)
            for x in self:
                print(x.face, x.rotation)
                exit(-1)
        self.f_bar = self.combined + self.cost - self.reverse_heuristic
        self.reverse_parent = None

    def get_path(self) -> List[Tuple[int, int]]:
        """Iterates backwards through the parent nodes to produce a path from start to this position"""
        path = [(x.face, x.rotation) for x in self]
        path.reverse()
        return path

    def __eq__(self, other: Node) -> bool:
        for idx, val in enumerate(other.state):
            if val != self.state[idx]:
                return False
        return True

    def __hash__(self):
        return hash(self.state.tobytes())

    def __iter__(self):
        """Define an iterator (for _ in Node) will iterate back through to the parent"""
        return NodeIterator(self)


class NodeIterator:
    """Iterates through the provided node until the starting node (parent == None) is reached.  Used as a helper
    function to generate full path backwards."""
    def __init__(self, node: Node) -> None:
        self.node = node

    def __next__(self) -> Node:
        if self.node is None:
            raise StopIteration()
        else:
            node = self.node
            self.node = self.node.parent
            return node

    def __iter__(self) -> NodeIterator:
        return self
