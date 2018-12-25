from __future__ import annotations
from numba import njit
import numpy as np
from collections import defaultdict
from typing import Optional, Callable, Tuple, List, Dict, Set
import math
import rubiks_optimized as ro


class Node:
    def __init__(self, parent: Optional[Node], state: np.ndarray, face: int, rotation: int, cost: int, heuristic_func, other_heuristic):
        self.parent = parent
        self.state = state
        self.face = face
        self.rotation = rotation
        self.cost = cost
        self.heuristic = heuristic_func(self)
        self.combined = self.cost + self.heuristic
        self.f_bar = self.combined + self.cost - other_heuristic(self)
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


def heuristic(node: Node, corner_db, edge_10):
    new_corner_index = ro.get_corner_index(node.state)
    # new_edge_index_6a = get_edge_index(state, edge_pos_indices_6a, edge_rot_indices_6a)
    # new_edge_index_6b = get_edge_index(state, edge_pos_indices_6b, edge_rot_indices_6b)
    new_edge_index_10 = ro.get_edge_index(node.state, ro.edge_pos_indices_10, ro.edge_rot_indices_10)
    return max(corner_db[new_corner_index], edge_10[new_edge_index_10])


def backward_heuristic(node: Node, target_state: np.ndarray, corner_db, edge_10):
    goal = target_state
    for x in node:
        goal = ro.rotate(goal, x.face, ro.inverse_rotation(x.rotation))
    return heuristic(goal, corner_db, edge_10)


def expand_forward(frontier: List[Node], other_frontier: List[Node], closed: Dict[Node, Node], other_closed: Dict[Node, Node], upper_bound: int):
    next_value = frontier.pop()
    if next_value not in closed:
        for face in range(6):
            for rotation in range(3):
                new_state = ro.rotate(next_value.state, face, rotation)
                node = Node(next_value, new_state, face, rotation, next_value.cost + 1, heuristic, backward_heuristic)
                if node in closed and node.f_bar < closed[node].f_bar:
                    closed.pop(node)

                if node not in closed:
                    frontier.append(node)
                    reverse_found = None
                    if node in other_closed:
                        reverse_found = other_closed[node]
                    elif node in other_frontier:
                        reverse_found = other_frontier[other_frontier.index(node)]
                    if reverse_found is not None and node.cost + reverse_found.cost < upper_bound:
                        upper_bound = node.cost + reverse_found.cost
                        best_node = node
                        best_node.reverse_parent = reverse_found
        closed[next_value] = next_value
    return upper_bound


def dibbs(start: np.ndarray, goal: np.ndarray, corner_db, edge_10):
    forward_costs = defaultdict(lambda: math.inf)
    backward_costs = defaultdict(lambda: math.inf)
    forward_fbar = defaultdict(lambda: math.inf)
    backward_fbar = defaultdict(lambda: math.inf)

    start_node = Node(None, start, None, None, 0, heuristic, backward_heuristic)
    goal_node = Node(None, goal, None, None, 0, backward_heuristic, heuristic)

    forward_costs[start.tobytes()] = 0
    forward_fbar[start.tobytes()] = start_node.f_bar
    forward_fbar_min = 0

    backward_costs[goal.tobytes()] = 0
    backward_fbar[goal.tobytes()] = goal_node.f_bar
    backward_fbar_min = 0

    forward_frontier = [start]
    backward_frontier = [goal]
    forward_closed = set()
    backward_closed = set()

    upper_bound = math.inf

    while upper_bound > (forward_fbar_min + backward_fbar_min) / 2:
        if explore_forward:
            expand_forward()
        else:
            expand_backward()

