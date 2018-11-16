from rubiks_optimized import *
import time
import argparse
from typing import Tuple, List
from queue import PriorityQueue
import numpy as np
import _khash_ffi

from numba import cffi_support

cffi_support.register_module(_khash_ffi)

khash_init = _khash_ffi.lib.khash_int2int_init
khash_get = _khash_ffi.lib.khash_int2int_get
khash_set = _khash_ffi.lib.khash_int2int_set
khash_destroy = _khash_ffi.lib.khash_int2int_destroy


#@njit
def a_star(state, forward_db):
    queue = list()
    starting_state = state
    queue.append((starting_state, 0, np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8)))

    if is_solved(state):
        return np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8), 0

    new_state_index = get_corner_index(state)
    min_moves = forward_db[new_state_index]
    print("Minimum number of moves to solve: ", min_moves)
    id_depth = min_moves
    count = 0
    while True:

        if len(queue) == 0:
            id_depth += 1
            queue.append((starting_state, 0, np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8)))
            print("Incrementing id-depth to", id_depth)

        next_state, depth, prev_faces, prev_rots = queue.pop()

        for face in range(6):

            new_faces = np.empty(len(prev_faces) + 1, dtype=np.uint8)
            for idx, val in enumerate(prev_faces):
                new_faces[idx] = val
            new_faces[len(prev_faces)] = face

            for rotation in range(3):
                new_state_base = rotate(next_state, face, rotation)
                new_state_depth = depth + 1
                new_state_index = get_corner_index(new_state_base)
                new_state_heuristic = forward_db[new_state_index]
                new_state_cost = new_state_depth + new_state_heuristic

                new_rots = np.empty(len(prev_rots) + 1, dtype=np.uint8)
                for idx, val in enumerate(prev_rots):
                    new_rots[idx] = val
                new_rots[len(prev_rots)] = rotation

                count += 1

                if is_solved(new_state_base):
                    return new_faces, new_rots, count

                if new_state_cost > id_depth:
                    continue

                queue.append((new_state_base, new_state_depth, new_faces, new_rots))


if __name__ == "__main__":
    start = time.perf_counter()
    #parser = argparse.ArgumentParser(description='Rubik\'s Cube Solver')

    #parser.add_argument('--method', dest="search", type=str, default="astar",
    #                    choices=["astar", "dibbs"],
    #                    help='search method - default astar')

    #args = parser.parse_args()
    #app.execute(args.filename, args.search, args.save)
    corner_db = generate_pattern_database(get_cube())
    save_pattern_database('database', corner_db)


    #generate_pattern_database.inspect_types()

    #corner_db = load_pattern_database('database')

    with open('official_scramble.txt') as f:
        output = f.read()
        print(output)
        cube = scramble(output)
        print(cube)

    faces, rotations, searched = a_star(cube, corner_db)
    size = len(faces)
    print("Moves required to solve:")
    for i in range(size - 1, -1, -1):
        print(Face(faces[i]).name, Rotation(rotations[i]).name)

    print("Finished", time.perf_counter() - start)
    pass
