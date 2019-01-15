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


def expand(frontier: List[Node], other_frontier: List[Node], closed: Dict[Node, Node], f_heuristic: Callable, r_heuristic: Callable,
           upper_bound: int, best_node: Node, count: int, fmin: int, gmin: int):

    parent_node = frontier.pop()
    closed[parent_node] = parent_node

    for face in range(6):
        if parent_node.face is not None and ro.skip_rotations(parent_node.face, face):
            continue
        for rotation in range(3):
            new_state = ro.rotate(parent_node.state, face, rotation)
            child_node = Node(parent_node, new_state, face, rotation, parent_node.cost + 1, f_heuristic, r_heuristic)
            count += 1

            if child_node in frontier and frontier[frontier.index(child_node)].cost <= child_node.cost:
                continue
            elif child_node in frontier:
                frontier.remove(child_node)

            if child_node in closed and closed[child_node].cost <= child_node.cost:
                continue
            elif child_node in closed:
                closed.pop(child_node)

            gmin = min(child_node.cost, gmin)
            fmin = min(child_node.combined, fmin)

            frontier.append(child_node)
            if child_node in other_frontier:
                reverse_found = other_frontier[other_frontier.index(child_node)]
                if upper_bound > child_node.cost + reverse_found.cost:
                    upper_bound = min(upper_bound, child_node.cost + reverse_found.cost)
                    best_node = child_node
                    best_node.reverse_parent = reverse_found
                    print("New upper bound: ", upper_bound)
    return upper_bound, best_node, count, fmin, gmin


def mm(start: np.ndarray, goal: np.ndarray, forward_heuristic, reverse_heuristic):

    start_node = Node(None, start, None, None, 0, forward_heuristic, reverse_heuristic)
    goal_node = Node(None, goal, None, None, 0, reverse_heuristic, forward_heuristic)

    f_frontier = [start_node]
    b_frontier = [goal_node]
    f_closed = dict()
    b_closed = dict()

    upper_bound = math.inf
    prminf = math.inf
    prminb = math.inf
    fminf = math.inf
    fminb = math.inf
    gminf = math.inf
    gminb = math.inf

    best_node = None
    count = 0
    while len(f_frontier) > 0 and len(b_frontier) > 0:
        c = min(prminf, prminb)
        if upper_bound <= max(c, fminf, fminb, gminf + gminb):
            return upper_bound

        if c == prminf:
            upper_bound, best_node, count, fminf, gminf = expand(f_frontier, b_frontier, f_closed, forward_heuristic, reverse_heuristic, upper_bound, best_node, count, fminf, gminf)
            f_frontier.sort(key=lambda x: x.pr, reverse=True)
            prminf = f_frontier[-1].pr

        else:
            upper_bound, best_node, count, fminb, gminb = expand(b_frontier, f_frontier, b_closed, reverse_heuristic, forward_heuristic, upper_bound, best_node, count, fminb, gminb)
            b_frontier.sort(key=lambda x: x.pr, reverse=True)
            prminb = b_frontier[-1].pr
    return math.inf

    path = best_node.get_path()
    reverse_path = best_node.reverse_parent.get_path()

    best_start_state = None
    for x in best_node:
        best_start_state = x.state

    if np.array_equal(best_start_state, start):
        path, reverse_path = reverse_path, path

    path = [(face, ro.inverse_rotation(rotation)) for face, rotation in path]
    path.extend(reversed(reverse_path))

    faces = []
    rotations = []

    for face, rotation in path:
        if face is not None:
            faces.append(face)
            rotations.append(rotation)

    return faces, rotations, count

