from __future__ import annotations
from numba import njit
import numpy as np
from collections import defaultdict
from typing import Optional, Callable, Tuple, List, Dict, Set
import math
import rubiks_optimized as ro
from node import Node


def expand(frontier: List[Node], other_frontier: List[Node], closed: Dict[Node, Node], f_heuristic: Callable, r_heuristic: Callable, upper_bound: int, best_node: Node, count: int):
    next_value = frontier.pop()
    if next_value not in closed:
        for face in range(6):

            if next_value.face is not None and ro.skip_rotations(next_value.face, face):
                continue

            for rotation in range(3):
                new_state = ro.rotate(next_value.state, face, rotation)
                node = Node(next_value, new_state, face, rotation, next_value.cost + 1, f_heuristic, r_heuristic)
                count += 1
                if node in closed and node.f_bar < closed[node].f_bar:
                    closed.pop(node)

                if node not in closed:
                    frontier.append(node)

                    if node in other_frontier:
                        upper_bound = min(upper_bound, node.cost)

        closed[next_value] = next_value
    return upper_bound, best_node, count


def mm(start: np.ndarray, goal: np.ndarray, forward_heuristic, reverse_heuristic):
    start_node = Node(None, start, None, None, 0, forward_heuristic, reverse_heuristic)
    goal_node = Node(None, goal, None, None, 0, reverse_heuristic, forward_heuristic)

    forward_frontier = [start_node]
    backward_frontier = [goal_node]
    forward_closed = dict()
    backward_closed = dict()

    upper_bound = math.inf
    prminf = math.inf
    prminb = math.inf

    best_node = None
    count = 0
    while len(forward_frontier) > 0 and len(backward_frontier) > 0:
        c = min(prminf, prminb)
        if upper_bound <= max(c, fminf, fminb, gminf + gminb):
            return upper_bound

        if c == prminf:
            node = forward_frontier.pop()
            forward_closed[node] = node

            for face in range(6):
                if node.face is not None and ro.skip_rotations(node.face, face):
                    continue
                for rotation in range(3):
                    new_state = ro.rotate(node.state, face, rotation)
                    node = Node(node, new_state, face, rotation, node.cost + 1, f_heuristic, r_heuristic)
                    count += 1

                    if node in forward_closed and node.cost < closed[node].f_bar:
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
                            print("New upper bound: ", upper_bound)

        else:
            upper_bound, best_node, count = expand(backward_frontier, forward_frontier, backward_closed, forward_closed, reverse_heuristic, forward_heuristic, upper_bound, best_node, count)
            backward_frontier.sort(key=lambda x: x.f_bar, reverse=True)
            backward_fbar_min = backward_frontier[-1].f_bar
            if backward_fbar_min > best_fbar:
                best_fbar = backward_fbar_min
                print(best_fbar)
        explore_forward = forward_fbar_min < backward_fbar_min
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

