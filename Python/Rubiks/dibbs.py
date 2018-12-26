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


def heuristic(node: np.ndarray, corner_db, edge_6a, edge_6b, edge_10):
    new_corner_index = ro.get_corner_index(node)
    new_edge_index_6a = ro.get_edge_index(node, ro.edge_pos_indices_6a, ro.edge_rot_indices_6a)
    new_edge_index_6b = ro.get_edge_index(node, ro.edge_pos_indices_6b, ro.edge_rot_indices_6b)
    new_edge_index_10 = ro.get_edge_index(node, ro.edge_pos_indices_10, ro.edge_rot_indices_10)
    return max(corner_db[new_corner_index], edge_6a[new_edge_index_6a], edge_6b[new_edge_index_6b])


def backward_heuristic(node: Node, target_state: np.ndarray, corner_db, edge_6a, edge_6b, edge_10):
    goal = target_state
    for x in node:
        if x.parent is not None:
            goal = ro.rotate(goal, x.face, ro.inverse_rotation(x.rotation))
    return heuristic(goal, corner_db, edge_6a, edge_6b, edge_10)


def expand_forward(frontier: List[Node], other_frontier: List[Node], closed: Dict[Node, Node],
                   other_closed: Dict[Node, Node], f_heuristic: Callable, r_heuristic: Callable,
                   upper_bound: int):
    next_value = frontier.pop()
    if next_value not in closed:
        for face in range(6):
            for rotation in range(3):
                new_state = ro.rotate(next_value.state, face, rotation)
                node = Node(next_value, new_state, face, rotation, next_value.cost + 1, f_heuristic, r_heuristic)
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


def dibbs(start: np.ndarray, goal: np.ndarray, corner_db, edge_6a, edge_6b, edge_10):
    forward_costs = defaultdict(lambda: math.inf)
    backward_costs = defaultdict(lambda: math.inf)
    forward_fbar = defaultdict(lambda: math.inf)
    backward_fbar = defaultdict(lambda: math.inf)

    forward_heuristic = lambda node: heuristic(node.state, corner_db, edge_6a, edge_6b, edge_10)
    reverse_heuristic = lambda node: backward_heuristic(node, goal, corner_db, edge_6a, edge_6b, edge_10)

    start_node = Node(None, start, None, None, 0, forward_heuristic, reverse_heuristic)
    goal_node = Node(None, goal, None, None, 0, reverse_heuristic, forward_heuristic)

    forward_costs[start_node] = 0
    forward_fbar[start_node] = start_node.f_bar
    forward_fbar_min = 0

    backward_costs[goal_node] = 0
    backward_fbar[goal_node] = goal_node.f_bar
    backward_fbar_min = 0

    forward_frontier = [start_node]
    backward_frontier = [goal_node]
    forward_closed = dict()
    backward_closed = dict()

    upper_bound = math.inf

    explore_forward = True
    while upper_bound > (forward_fbar_min + backward_fbar_min) / 2:
        if explore_forward:
            upper_bound = expand_forward(forward_frontier, backward_frontier, forward_closed, backward_closed, forward_heuristic, reverse_heuristic, upper_bound)
            forward_frontier.sort(key=lambda x: x.f_bar, reverse=True)
            forward_fbar_min = forward_frontier[len(forward_frontier) - 1].f_bar
            explore_forward = False
        else:
            upper_bound = expand_forward(backward_frontier, forward_frontier, backward_closed, forward_closed, reverse_heuristic, forward_heuristic, upper_bound)
            backward_frontier.sort(key=lambda x: x.f_bar, reverse=True)
            backward_fbar_min = backward_frontier[len(backward_frontier) - 1].f_bar
            explore_forward = True

