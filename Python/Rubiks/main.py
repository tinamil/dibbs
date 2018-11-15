from rubiks_optimized import *
import time
import argparse
from typing import Tuple, List
from queue import PriorityQueue
import numpy as np


def a_star(state, forward_db) -> Tuple[List[Tuple[Face, Rotation]], int]:
    queue = list()
    queue.append((state, 0))

    # 8 corners for 8 positions, 7 of which can have 3 unique rotations, 88179840 possibilities
    found_index = np.zeros(10000, dtype=np.uint8)
    found_index_stack = np.full(all_corners, 100, dtype=np.uint8)
    found_index[get_corner_index(state)] = 1
    id_depth = 1
    count = 1
    while True:

        if len(queue) == 0:
            id_depth += 1
            found_index_stack = np.full(all_corners, 100, dtype=np.uint8)
            queue.append((state, 0))
            print("Incrementing id-depth to", id_depth)

        next_state, depth = queue.pop()
        if is_solved(next_state):
            pass
            #return solution

        for face in range(6):
            for rotation in range(3):
                new_state = rotate(np.copy(next_state), face, rotation)
                new_state_index = get_corner_index(new_state)
                new_state_depth = depth + 1
                new_state_heuristic = forward_db[new_state_index]
                new_state_cost = new_state_depth + new_state_heuristic
                if new_state_depth == id_depth and found_index[new_state_index] == 0:
                    pattern_lookup[new_state_index] = new_state_depth
                    found_index[new_state_index] = 1
                    count += 1
                    if count % 100000 == 0 or count > 88100000:
                        print(count, new_state_depth, len(queue))
                elif new_state_depth < id_depth and new_state_depth < found_index_stack[new_state_index]:
                    found_index_stack[new_state_index] = new_state_depth
                    queue.append((new_state, new_state_depth))

    return pattern_lookup

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Rubik\'s Cube Solver')

    #parser.add_argument('--method', dest="search", type=str, default="astar",
    #                    choices=["astar", "dibbs"],
    #                    help='search method - default astar')

    #args = parser.parse_args()
    #app.execute(args.filename, args.search, args.save)

    with open('official_scramble.txt') as f:
        output = f.read()
        print(rubiks.scramble(output))

    pass
