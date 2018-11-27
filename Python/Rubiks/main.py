from rubiks_optimized import *
import time
import numpy as np

import _khash_ffi

from numba import cffi_support

cffi_support.register_module(_khash_ffi)

khash_init = _khash_ffi.lib.khash_int2int_init
khash_get = _khash_ffi.lib.khash_int2int_get
khash_set = _khash_ffi.lib.khash_int2int_set
khash_destroy = _khash_ffi.lib.khash_int2int_destroy


@njit
def a_star(state, corner_db, edge_list, edge_max_depth):
    edge_db = khash_init()
    for key, val in edge_list:
        khash_set(edge_db, key, val)

    queue = list()
    starting_state = state
    queue.append((starting_state, 0, np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8)))

    if is_solved(state):
        return np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8), 0

    new_corner_index = get_corner_index(state)
    new_edge_index = get_edge_index(state)
    min_moves = max(khash_get(edge_db, new_edge_index, edge_max_depth), corner_db[new_corner_index])
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
                new_corner_index = get_corner_index(new_state_base)
                new_edge_index = get_edge_index(new_state_base)
                new_state_heuristic = max(khash_get(edge_db, new_edge_index, edge_max_depth), corner_db[new_corner_index])
                new_state_cost = new_state_depth + new_state_heuristic

                if new_state_cost > id_depth:
                    if is_solved(new_state_base):
                        edge_h = khash_get(edge_db, new_edge_index, edge_max_depth)
                        corner_h = corner_db[new_corner_index]
                        print(new_edge_index, edge_h, corner_h, new_state_depth, new_state_base, "Failure to issolve")
                        raise ValueError("Failed")
                    continue

                new_rots = np.empty(len(prev_rots) + 1, dtype=np.uint8)
                for idx, val in enumerate(prev_rots):
                    new_rots[idx] = val
                new_rots[len(prev_rots)] = rotation

                count += 1

                if is_solved(new_state_base):
                    return new_faces, new_rots, count

                queue.append((new_state_base, new_state_depth, new_faces, new_rots))


if __name__ == "__main__":
    mode = 0

    edge_max_depth = 20
    corner_max_depth = 10

    start = time.perf_counter()
    if mode == 0:
        khash_db, indices = generate_edges_pattern_database(get_cube(), edge_max_depth)
        khash_db = cffi_support.ffi.cast('void *', khash_db)
        edge_db = convert_khash_to_dict(khash_db, indices, edge_max_depth)
        save_pattern_database('edge_db.pkl', edge_db)

    elif mode == 1:
        corner_db = generate_corners_pattern_database(get_cube(), corner_max_depth)
        save_pattern_database('corner_db.npy', corner_db)

    else:
        corner_db = load_pattern_database('corner_db.npy')
        dict_edge_db = load_pattern_database('edge_db.pkl')
        edge_khash_db = convert_dict_to_list(dict_edge_db)

        with open('test_file.txt') as f:
            output = f.read()
            print(output)
            cube = scramble(output)
            print(cube)

        faces, rotations, searched = a_star(cube, corner_db, edge_khash_db, edge_max_depth)

        size = len(faces)
        print("Moves required to solve:")
        for i in range(size - 1, -1, -1):
            print(Face(faces[i]).name, Rotation(rotations[i]).name)

    print("Finished", time.perf_counter() - start)
    pass
