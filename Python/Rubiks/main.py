from rubiks_optimized import is_solved, manhattan_heuristic, __goal, rotate, get_corner_index, get_edge_index, edge_pos_indices_6a, edge_pos_indices_6b, edge_rot_indices_6a, edge_rot_indices_6b, \
    Face, Rotation, generate_edges_memmap, get_cube, generate_corners_pattern_database, load_pattern_database, scramble, save_pattern_database, npr, edge_pos_indices_10, edge_rot_indices_10
import time
import numpy as np
import dibbs
from numba import njit
import enum

import _khash_ffi

from numba import cffi_support

cffi_support.register_module(_khash_ffi)

khash_init = _khash_ffi.lib.khash_int2int_init
khash_get = _khash_ffi.lib.khash_int2int_get
khash_set = _khash_ffi.lib.khash_int2int_set
khash_destroy = _khash_ffi.lib.khash_int2int_destroy


#TODO, implement bidirectional unguided search


@njit
def a_star(state, heuristic_func):
    queue = list()
    starting_state = state
    queue.append((starting_state, 0, np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8)))

    if is_solved(state):
        return np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8), 0

    min_moves = heuristic_func(state)
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
                new_state_heuristic = heuristic_func(new_state_base)
                new_state_cost = new_state_depth + new_state_heuristic

                if new_state_cost > id_depth:
                    continue

                new_rots = np.empty(len(prev_rots) + 1, dtype=np.uint8)
                for idx, val in enumerate(prev_rots):
                    new_rots[idx] = val
                new_rots[len(prev_rots)] = rotation

                count += 1

                if is_solved(new_state_base):
                    flip(new_faces)
                    flip(new_rots)
                    return new_faces, new_rots, count

                queue.append((new_state_base, new_state_depth, new_faces, new_rots))


@njit
def flip(array):
    count = len(array) - 1
    for idx in range((count+1) // 2):
        array[idx], array[count - idx] = array[count - idx], array[idx]


@njit
def pattern_database_lookup(state, corner_db, edge_6a, edge_6b, edge_10):
    new_corner_index = get_corner_index(state)
    new_edge_index_6a = get_edge_index(state, edge_pos_indices_6a, edge_rot_indices_6a)
    new_edge_index_6b = get_edge_index(state, edge_pos_indices_6b, edge_rot_indices_6b)
    #new_edge_index_10 = get_edge_index(state, edge_pos_indices_10, edge_rot_indices_10)
    #return max(corner_db[new_corner_index], edge_10[new_edge_index_10])
    return max(corner_db[new_corner_index], edge_6a[new_edge_index_6a], edge_6b[new_edge_index_6b])


load_corner_db = None
dict_edge_db_6a = None
dict_edge_db_6b = None
dict_edge_db_10 = None


@njit
def forward_pattern_database_heuristic(state):
    return pattern_database_lookup(state, load_corner_db, dict_edge_db_6a, dict_edge_db_6b, dict_edge_db_10)


@njit
def forward_manhattan_heuristic(state):
    return manhattan_heuristic(state, __goal)


start_state = np.empty(0)


@njit
def backward_manhattan_heuristic(state):
    return manhattan_heuristic(state, start_state)

@njit
def zero_heuristic(state):
    return 0


@enum.unique
class HeuristicType(enum.Enum):
    man = enum.auto()
    pattern = enum.auto()
    zero = enum.auto()


@enum.unique
class AlgorithmType(enum.Enum):
    astar = enum.auto()
    dibbs = enum.auto()


@enum.unique
class Mode(enum.Enum):
    generate_edges = enum.auto()
    generate_corners = enum.auto()
    search = enum.auto()


if __name__ == "__main__":
    mode = Mode.search
    heuristic_choice = HeuristicType.zero
    algorithm_choice = AlgorithmType.dibbs

    edge_max_depth = 20
    corner_max_depth = 20

    start = time.perf_counter()
    print("Starting at ", time.ctime())

    if mode == Mode.generate_edges:
        #edge_db_6a = generate_edges_pattern_database(get_cube(), edge_max_depth, edge_pos_indices_6a, edge_rot_indices_6a)
        #save_pattern_database('edge_db_6a.npy', edge_db_6a)
        #del edge_db_6a

        #edge_db_6b = generate_edges_pattern_database(get_cube(), edge_max_depth, edge_pos_indices_6b, edge_rot_indices_6b)
        #save_pattern_database('edge_db_6b.npy', edge_db_6b)
        #del edge_db_6b
        mem_map = np.memmap('edge_db_10.npy', dtype=np.int8, mode='w+', shape=np.uint64(npr(12, len(edge_pos_indices_10)) * 2**(len(edge_pos_indices_10))))
        #TODO: Eats 245 GB of RAM, needs intermittent flushed
        mem_map[:] = edge_max_depth
        edge_db_10 = generate_edges_memmap(get_cube(), edge_max_depth, edge_pos_indices_10, edge_rot_indices_10, mem_map)
        #save_pattern_database('edge_db_10.npy', edge_db_10)
        del edge_db_10

    elif mode == Mode.generate_corners:
        load_corner_db = generate_corners_pattern_database(get_cube(), corner_max_depth)
        save_pattern_database('corner_db.npy', load_corner_db)

    elif mode == Mode.search:
        load_corner_db = load_pattern_database('corner_db.npy')
        dict_edge_db_6a = load_pattern_database('edge_db_6a.npy')
        dict_edge_db_6b = load_pattern_database('edge_db_6b.npy')
        #dict_edge_db_10 = load_pattern_database('edge_db_10.npy')
        dict_edge_db_10 = None

        with open('test_file.txt') as f:
            output = f.read()
            print(output)
            start_state = scramble(output)
            print(start_state)

        if algorithm_choice == AlgorithmType.astar:
            print("A*")
            if heuristic_choice == HeuristicType.man:
                faces, rotations, searched = a_star(start_state, forward_manhattan_heuristic)
            elif heuristic_choice == HeuristicType.pattern:
                faces, rotations, searched = a_star(start_state, forward_pattern_database_heuristic)
            elif heuristic_choice == HeuristicType.zero:
                faces, rotations, searched = a_star(start_state, zero_heuristic)
            else:
                raise Exception("Failed to identify type of heuristic")
        elif algorithm_choice == AlgorithmType.dibbs:
            print("DIBBS")
            if heuristic_choice == HeuristicType.man:
                faces, rotations, searched = dibbs.dibbs(start_state, get_cube(), forward_manhattan_heuristic, backward_manhattan_heuristic)
            elif heuristic_choice == HeuristicType.pattern:
                faces, rotations, searched = dibbs.dibbs(start_state, get_cube(), forward_pattern_database_heuristic, backward_pattern_database_heuristic)
            elif heuristic_choice == HeuristicType.zero:
                faces, rotations, searched = dibbs.dibbs(start_state, get_cube(), zero_heuristic, zero_heuristic)
            else:
                raise Exception("Failed to identify type of heuristic")
        else:
            raise Exception("Failed to identify type of algorithm")

        size = len(faces)
        print(f"Moves required to solve ({size}):")
        for i in range(size - 1, -1, -1):
            print(Face(faces[i]).__repr__() + Rotation(rotations[i]).__repr__())
        print(f"Explored {searched} nodes")

    print("Finished", time.perf_counter() - start)
    pass
