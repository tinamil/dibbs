from __future__ import annotations
from numba import njit
import numpy as np
from collections import defaultdict
from typing import Optional, Callable, Tuple, List, Dict, Set
import math
import rubiks_optimized as ro
from node import Node
import heapq
import pprofile


# def backward_heuristic(node: Node, target_state: np.ndarray, corner_db, edge_6a, edge_6b, edge_10):
#     goal = target_state
#     for x in node:
#         if x.parent is not None:
#             goal = ro.rotate(goal, x.face, ro.inverse_rotation(x.rotation))
#     back_h = heuristic(goal, corner_db, edge_6a, edge_6b, edge_10)
#     return back_h


def expand(frontier: List[Node], frontier_set: Dict[Node, Tuple[Node, int]], other_set: Dict[Node, Tuple[Node, int]], f_heuristic: Callable, r_heuristic: Callable,
           upper_bound: int, best_node: Node, count: int):
    next_value = heapq.heappop(frontier)
    tmp_next_val, node_count = frontier_set.pop(next_value)
    if node_count > 1:
        frontier_set[tmp_next_val] = tmp_next_val, node_count - 1
    for face in range(6):

        if next_value.face is not None and ro.skip_rotations(next_value.face, face):
            continue

        for rotation in range(3):
            new_state = ro.rotate(next_value.state, face, rotation)
            node = Node(next_value, new_state, face, rotation, next_value.cost + 1, f_heuristic, r_heuristic)
            count += 1

            heapq.heappush(frontier, node)
            if node in frontier_set:
                tmp_next_val, node_count = frontier_set[node]
                if node.f_bar < tmp_next_val.f_bar:
                    frontier_set[node] = node, node_count + 1
                else:
                    frontier_set[tmp_next_val] = tmp_next_val, node_count + 1
            else:
                frontier_set[node] = node, 1
            reverse_found = None
            if node in other_set:
                reverse_found, node_count = other_set[node]
            if reverse_found is not None and node.cost + reverse_found.cost < upper_bound:
                upper_bound = node.cost + reverse_found.cost
                best_node = node
                best_node.reverse_parent = reverse_found
                print("New upper bound: ", upper_bound)

    return upper_bound, best_node, count


def dibbs(start: np.ndarray, goal: np.ndarray, forward_heuristic, reverse_heuristic):
    forward_fbar = defaultdict(lambda: math.inf)
    backward_fbar = defaultdict(lambda: math.inf)

    start_node = Node(None, start, None, None, 0, forward_heuristic, reverse_heuristic)
    goal_node = Node(None, goal, None, None, 0, reverse_heuristic, forward_heuristic)

    forward_fbar[start_node] = start_node.f_bar
    forward_fbar_min = 0

    backward_fbar[goal_node] = goal_node.f_bar
    backward_fbar_min = 0

    forward_frontier = [start_node]
    forward_set = {start_node: (start_node, 1)}
    backward_frontier = [goal_node]
    backward_set = {goal_node: (goal_node, 1)}

    upper_bound = math.inf

    explore_forward = False
    best_node = None
    count = 0
    f_combined = -math.inf
    b_combined = -math.inf
    best_f_fbar = -math.inf
    best_b_fbar = -math.inf
    f_cost = -math.inf
    b_cost = -math.inf
    while upper_bound > (forward_fbar_min + backward_fbar_min) / 2:
        if explore_forward:
            upper_bound, best_node, count = expand(forward_frontier, forward_set, backward_set, forward_heuristic, reverse_heuristic, upper_bound, best_node, count)
            forward_fbar_min = forward_frontier[0].f_bar
            if forward_frontier[0].cost > f_cost:
                f_cost = forward_frontier[0].cost
                print("Forward cost: ", f_cost)
            if forward_fbar_min > best_f_fbar:
                best_f_fbar = forward_fbar_min
                print("Forward fbar:", best_f_fbar)
            if forward_frontier[0].combined > f_combined:
                f_combined = forward_frontier[0].combined
                print("Forward combined: ", f_combined)
        else:
            upper_bound, best_node, count = expand(backward_frontier, backward_set, forward_set, reverse_heuristic, forward_heuristic, upper_bound, best_node, count)
            backward_fbar_min = backward_frontier[0].f_bar
            if backward_frontier[0].cost > b_cost:
                b_cost = backward_frontier[0].cost
                print("Backward cost: ", b_cost)
            if backward_fbar_min > best_b_fbar:
                best_b_fbar = backward_fbar_min
                print("Backward fbar:", best_b_fbar)
            if backward_frontier[0].combined > b_combined:
                b_combined = backward_frontier[0].combined
                print("Backward combined: ", b_combined)
        explore_forward = forward_fbar_min < backward_fbar_min
        #explore_forward = len(forward_frontier) < len(backward_frontier)
        #explore_forward = forward_frontier[-1].cost < backward_frontier[-1].cost

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

