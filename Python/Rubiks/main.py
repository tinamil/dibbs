from rubiks_optimized import get_corner_index, get_edge_index, edge_pos_indices_6a, edge_pos_indices_6b, edge_rot_indices_6a, edge_rot_indices_6b, __goal, manhattan_heuristic, get_cube, \
    generate_edges_pattern_database, npr, save_pattern_database

import rubiks_optimized as ro
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


@njit
def pattern_database_lookup(state, corner_db, edge_6a, edge_6b, edge_10):
    new_corner_index = get_corner_index(state)
    new_edge_index_6a = get_edge_index(state, edge_pos_indices_6a, edge_rot_indices_6a)
    new_edge_index_6b = get_edge_index(state, edge_pos_indices_6b, edge_rot_indices_6b)
    #new_edge_index_10 = get_edge_index(state, edge_pos_indices_10, edge_rot_indices_10)
    #return max(corner_db[new_corner_index], edge_10[new_edge_index_10])
    return max(corner_db[new_corner_index], edge_6a[new_edge_index_6a], edge_6b[new_edge_index_6b])


goal_corner_db = None
goal_edge_db_6a = None
goal_edge_db_6b = None
goal_edge_db_10 = None


@njit
def forward_pattern_database_heuristic(state):
    return pattern_database_lookup(state, goal_corner_db, goal_edge_db_6a, goal_edge_db_6b, goal_edge_db_10)


start_corner_db = None
start_edge_db_6a = None
start_edge_db_6b = None
start_edge_db_10 = None


@njit
def reverse_pattern_database_heuristic(state):
    return pattern_database_lookup(state, start_corner_db, start_edge_db_6a, start_edge_db_6b, start_edge_db_10)


@njit
def forward_manhattan_heuristic(state):
    return manhattan_heuristic(state, __goal)

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


def search(mode, heuristic_choice, algorithm_choice, start_state):
    edge_max_depth = 10
    corner_max_depth = 10

    start = time.perf_counter()
    print("Starting at ", time.ctime())

    if mode == Mode.generate_edges:
        edge_db_6a = generate_edges_pattern_database(get_cube(), edge_max_depth, edge_pos_indices_6a, edge_rot_indices_6a)
        save_pattern_database('edge_db_6a.npy', edge_db_6a)
        del edge_db_6a

        edge_db_6b = generate_edges_pattern_database(get_cube(), edge_max_depth, edge_pos_indices_6b, edge_rot_indices_6b)
        ro.save_pattern_database('edge_db_6b.npy', edge_db_6b)
        del edge_db_6b
        mem_map = np.memmap('edge_db_10.npy', dtype=np.int8, mode='w+', shape=np.uint64(npr(12, len(ro.edge_pos_indices_10)) * 2 ** (len(ro.edge_pos_indices_10))))
        # TODO: Eats 245 GB of RAM, needs intermittent flushed
        mem_map[:] = edge_max_depth
        edge_db_10 = ro.generate_edges_memmap(ro.get_cube(), edge_max_depth, ro.edge_pos_indices_10, ro.edge_rot_indices_10, mem_map)
        # save_pattern_database('edge_db_10.npy', edge_db_10)
        del edge_db_10

    elif mode == Mode.generate_corners:
        load_corner_db = ro.generate_corners_pattern_database(ro.get_cube(), corner_max_depth)
        ro.save_pattern_database('corner_db.npy', load_corner_db)

    elif mode == Mode.search:
        goal_corner_db = ro.load_pattern_database('corner_db.npy')
        goal_edge_db_6a = ro.load_pattern_database('edge_db_6a.npy')
        goal_edge_db_6b = ro.load_pattern_database('edge_db_6b.npy')
        # goal_edge_db_10 = load_pattern_database('edge_db_10.npy')
        goal_edge_db_10 = None

        if heuristic_choice == HeuristicType.pattern:
            try:
                start_corner_db = ro.load_pattern_database('_start_corner_db.npy')
            except IOError:
                start_corner_db = ro.generate_corners_pattern_database(start_state, corner_max_depth)
                ro.save_pattern_database('_start_corner_db.npy', start_corner_db)

            try:
                start_edge_db_6a = ro.load_pattern_database('_start_edge_db_6a.npy')
            except IOError:
                start_edge_db_6a = ro.generate_edges_pattern_database(start_state, edge_max_depth, ro.edge_pos_indices_6a, ro.edge_rot_indices_6a)
                ro.save_pattern_database('_start_edge_db_6a.npy', start_edge_db_6a)

            try:
                start_edge_db_6b = ro.load_pattern_database('_start_edge_db_6b.npy')
            except IOError:
                start_edge_db_6b = ro.generate_edges_pattern_database(start_state, edge_max_depth, ro.edge_pos_indices_6b, ro.edge_rot_indices_6b)
                ro.save_pattern_database('_start_edge_db_6b.npy', start_edge_db_6b)

            start_edge_db_10 = None

        if algorithm_choice == AlgorithmType.astar:
            print("A*")
            if heuristic_choice == HeuristicType.man:
                faces, rotations, searched = ro.a_star(start_state, forward_manhattan_heuristic)
            elif heuristic_choice == HeuristicType.pattern:
                faces, rotations, searched = ro.a_star(start_state, forward_pattern_database_heuristic)
            elif heuristic_choice == HeuristicType.zero:
                faces, rotations, searched = ro.a_star(start_state, zero_heuristic)
            else:
                raise Exception("Failed to identify type of heuristic")

        elif algorithm_choice == AlgorithmType.dibbs:
            print("DIBBS")
            if heuristic_choice == HeuristicType.man:
                backward_manhattan_heuristic = lambda state: manhattan_heuristic(state, start_state)
                faces, rotations, searched = dibbs.dibbs(start_state, ro.get_cube(), forward_manhattan_heuristic, backward_manhattan_heuristic)
            elif heuristic_choice == HeuristicType.pattern:
                faces, rotations, searched = dibbs.dibbs(start_state, ro.get_cube(), forward_pattern_database_heuristic, reverse_pattern_database_heuristic)
            elif heuristic_choice == HeuristicType.zero:
                faces, rotations, searched = dibbs.dibbs(start_state, ro.get_cube(), zero_heuristic, zero_heuristic)
            else:
                raise Exception("Failed to identify type of heuristic")

        else:
            raise Exception("Failed to identify type of algorithm")

        size = len(faces)
        print(f"Moves required to solve ({size}):")
        for i in range(size - 1, -1, -1):
            print(ro.Face(faces[i]).__repr__() + ro.Rotation(rotations[i]).__repr__())
        print(f"Explored {searched} nodes")

    print("Finished", time.perf_counter() - start)
    return size, searched


def load_cube(file: str):
    with open(file) as f:
        output = f.read()
        print(output)
        start_state = ro.scramble(output)
        return start_state


if __name__ == "__main__":
    dibbs_results = []
    astar_results = []

    mode = Mode.search
    heuristic_choice = HeuristicType.man
    algorithm_choice = AlgorithmType.dibbs
    solution_length = 7

    while len(dibbs_results) < 100:
        start_state = ro.random_scramble(solution_length)
        size, searched = search(mode, heuristic_choice, AlgorithmType.dibbs, start_state)
        if size == solution_length:
            dibbs_results.append(searched)
            size, searched = search(mode, heuristic_choice, AlgorithmType.astar, start_state)
            astar_results.append(searched)

    print("DIBBS:", dibbs_results)
    print("A*:", astar_results)
    print("DIBBS:", np.mean(dibbs_results), "A*:", np.mean(astar_results))
