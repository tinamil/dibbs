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
def a_star(state, corner_db, edge_6a, edge_6b, edge_10):
    queue = list()
    starting_state = state
    queue.append((starting_state, 0, np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8)))

    if is_solved(state):
        return np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8), 0

    min_moves = heuristic(state, corner_db, edge_10)
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
                new_state_heuristic = heuristic(new_state_base, corner_db, edge_10)
                new_state_cost = new_state_depth + new_state_heuristic

                if new_state_cost > id_depth:
                    continue

                new_rots = np.empty(len(prev_rots) + 1, dtype=np.uint8)
                for idx, val in enumerate(prev_rots):
                    new_rots[idx] = val
                new_rots[len(prev_rots)] = rotation

                count += 1

                if is_solved(new_state_base):
                    return new_faces, new_rots, count

                queue.append((new_state_base, new_state_depth, new_faces, new_rots))


@njit
def heuristic(state, corner_db, edge_10):
    new_corner_index = get_corner_index(state)
    # new_edge_index_6a = get_edge_index(state, edge_pos_indices_6a, edge_rot_indices_6a)
    # new_edge_index_6b = get_edge_index(state, edge_pos_indices_6b, edge_rot_indices_6b)
    new_edge_index_10 = get_edge_index(state, edge_pos_indices_10, edge_rot_indices_10)
    return max(corner_db[new_corner_index], edge_10[new_edge_index_10])


if __name__ == "__main__":
    mode = 3

    edge_max_depth = 20
    corner_max_depth = 20

    start = time.perf_counter()
    print("Starting at ", time.ctime())

    if mode == 0:
        edge_db_6a = generate_edges_pattern_database(get_cube(), edge_max_depth, edge_pos_indices_6a, edge_rot_indices_6a)
        save_pattern_database('edge_db_6a.npy', edge_db_6a)
        del edge_db_6a

        edge_db_6b = generate_edges_pattern_database(get_cube(), edge_max_depth, edge_pos_indices_6b, edge_rot_indices_6b)
        save_pattern_database('edge_db_6b.npy', edge_db_6b)
        del edge_db_6b

        edge_db_10 = generate_edges_pattern_database(get_cube(), edge_max_depth, edge_pos_indices_10, edge_rot_indices_10)
        save_pattern_database('edge_db_10.npy', edge_db_10)
        del edge_db_10

    elif mode == 2:
        load_corner_db = generate_corners_pattern_database(get_cube(), corner_max_depth)
        save_pattern_database('corner_db.npy', load_corner_db)

    else:
        load_corner_db = load_pattern_database('corner_db.npy')
        dict_edge_db_a = load_pattern_database('edge_db_6a.npy')
        dict_edge_db_b = load_pattern_database('edge_db_6b.npy')
        dict_edge_db_10 = load_pattern_database('edge_db_10.npy')

        with open('test_file.txt') as f:
            output = f.read()
            print(output)
            cube = scramble(output)
            print(cube)

        faces, rotations, searched = a_star(cube, load_corner_db, dict_edge_db_a, dict_edge_db_b, dict_edge_db_10)

        size = len(faces)
        print("Moves required to solve:")
        for i in range(size - 1, -1, -1):
            print(Face(faces[i]).name, Rotation(rotations[i]).name)

    print("Finished", time.perf_counter() - start)
    pass
