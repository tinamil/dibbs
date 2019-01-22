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
        self.pr = max(self.combined, 2 * self.cost + 1)
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


def expand(frontier_pr: List[Node], frontier_f: List[Node], frontier_g: List[Node], other_frontier: List[Node], closed: Dict[Node, Node], f_heuristic: Callable, r_heuristic: Callable,
           upper_bound: int, best_node: Node, count: int):

    parent_node = frontier_pr.pop()
    frontier_f.remove(parent_node)
    frontier_g.remove(parent_node)
    closed[parent_node] = parent_node

    for face in range(6):
        if parent_node.face is not None and ro.skip_rotations(parent_node.face, face):
            continue
        for rotation in range(3):
            new_state = ro.rotate(parent_node.state, face, rotation)
            child_node = Node(parent_node, new_state, face, rotation, parent_node.cost + 1, f_heuristic, r_heuristic)
            count += 1

            if child_node in frontier_pr and frontier_pr[frontier_pr.index(child_node)].cost <= child_node.cost:
                continue
            elif child_node in frontier_pr:
                frontier_pr.remove(child_node)
                frontier_g.remove(child_node)
                frontier_f.remove(child_node)

            if child_node in closed and closed[child_node].cost <= child_node.cost:
                continue
            elif child_node in closed:
                closed.pop(child_node)

            frontier_pr.append(child_node)
            frontier_g.append(child_node)
            frontier_f.append(child_node)
            if child_node in other_frontier:
                reverse_found = other_frontier[other_frontier.index(child_node)]
                print("Reverse found")
                if upper_bound > child_node.cost + reverse_found.cost:
                    upper_bound = min(upper_bound, child_node.cost + reverse_found.cost)
                    print(upper_bound)
                    best_node = child_node
                    best_node.reverse_parent = reverse_found
                    print("New upper bound: ", upper_bound)
    return upper_bound, best_node, count


def mm(start: np.ndarray, goal: np.ndarray, forward_heuristic, reverse_heuristic):

    start_node = Node(None, start, None, None, 0, forward_heuristic, reverse_heuristic)
    goal_node = Node(None, goal, None, None, 0, reverse_heuristic, forward_heuristic)

    f_frontier_pr = [start_node]
    f_frontier_f = [start_node]
    f_frontier_g = [start_node]
    b_frontier_pr = [goal_node]
    b_frontier_f = [goal_node]
    b_frontier_g = [goal_node]
    f_closed = dict()
    b_closed = dict()

    upper_bound = math.inf
    prminf = f_frontier_pr[-1].pr
    prminb = b_frontier_pr[-1].pr
    fminf = f_frontier_f[-1].combined
    fminb = b_frontier_f[-1].combined
    gminf = f_frontier_g[-1].cost
    gminb = b_frontier_g[-1].cost

    best_node = None
    count = 0
    while len(f_frontier_pr) > 0 and len(b_frontier_pr) > 0:
        c = min(prminf, prminb)
        if upper_bound < math.inf and upper_bound <= max(c, fminf, fminb, gminf + gminb):
            # Return a solution
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

        if c == prminf:
            upper_bound, best_node, count = expand(f_frontier_pr, f_frontier_f, f_frontier_g, b_frontier_pr, f_closed, forward_heuristic, reverse_heuristic, upper_bound, best_node, count)
            f_frontier_g.sort(key=lambda x: x.cost, reverse=True)
            gminf = f_frontier_g[-1].cost
            f_frontier_f.sort(key=lambda x: x.combined, reverse=True)
            fminf = f_frontier_f[-1].combined
            f_frontier_pr.sort(key=lambda x: x.pr, reverse=True)
            prminf = f_frontier_pr[-1].pr
        else:
            upper_bound, best_node, count = expand(b_frontier_pr, b_frontier_f, b_frontier_g, f_frontier_g, b_closed, reverse_heuristic, forward_heuristic, upper_bound, best_node, count)
            b_frontier_g.sort(key=lambda x: x.cost, reverse=True)
            gminb = b_frontier_g[-1].cost
            b_frontier_f.sort(key=lambda x: x.combined, reverse=True)
            fminb = b_frontier_f[-1].combined
            b_frontier_pr.sort(key=lambda x: x.pr, reverse=True)
            prminb = b_frontier_pr[-1].pr
    raise ValueError("Unable to find a solution")


