from rubiks_optimized import get_corner_index, get_edge_index, edge_pos_indices_6a, edge_pos_indices_6b, edge_rot_indices_6a, edge_rot_indices_6b, __goal, manhattan_heuristic, get_cube, \
    generate_edges_pattern_database, npr, save_pattern_database

import rubiks_optimized as ro
import time
import numpy as np
import dibbs
from numba import njit
import enum
import mm
import os

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
    #new_edge_index_6a = get_edge_index(state, edge_pos_indices_6a, edge_rot_indices_6a)
    #new_edge_index_6b = get_edge_index(state, edge_pos_indices_6b, edge_rot_indices_6b)
    #new_edge_index_10 = get_edge_index(state, edge_pos_indices_10, edge_rot_indices_10)
    #return max(corner_db[new_corner_index], edge_10[new_edge_index_10])
    #return max(corner_db[new_corner_index], edge_6a[new_edge_index_6a], edge_6b[new_edge_index_6b])
    return corner_db[new_corner_index]



@njit
def forward_manhattan_heuristic(state):
    return manhattan_heuristic(state, __goal)


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
    mm = enum.auto()


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
    try:
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

    finally:
        print("Finished", time.perf_counter() - start)


def asymmetric_search(mode, forward_heuristic_choice, backward_heuristic_choice, algorithm_choice, start_state):
    edge_max_depth = 10
    corner_max_depth = 10

    start = time.perf_counter()
    print("Starting at ", time.ctime())
    try:
            goal_corner_db = ro.load_pattern_database('corner_db.npy')
            goal_edge_db_6a = ro.load_pattern_database('edge_db_6a.npy')
            goal_edge_db_6b = ro.load_pattern_database('edge_db_6b.npy')
            # goal_edge_db_10 = load_pattern_database('edge_db_10.npy')
            goal_edge_db_10 = None


            if algorithm_choice == AlgorithmType.astar:
                print("A*")
                algorithm = ro.a_star_with_backward_args
            elif algorithm_choice == AlgorithmType.dibbs:
                print("DIBBS")
                algorithm = dibbs.dibbs
            elif algorithm_choice == AlgorithmType.mm:
                print("MM")
                algorithm = mm.mm
            else:
                raise Exception("Failed to identify type of heuristic")

            if backward_heuristic_choice == HeuristicType.pattern:
                try:
                    start_corner_db = ro.load_pattern_database('_start_corner_db.npy')
                except IOError:
                    start_corner_db = ro.generate_corners_pattern_database(start_state, corner_max_depth)
                    ro.save_pattern_database('_start_corner_db.npy', start_corner_db)

                #try:
                #    start_edge_db_6a = ro.load_pattern_database('_start_edge_db_6a.npy')
                #except IOError:
                #    start_edge_db_6a = ro.generate_edges_pattern_database(start_state, edge_max_depth, ro.edge_pos_indices_6a, ro.edge_rot_indices_6a)
                #    ro.save_pattern_database('_start_edge_db_6a.npy', start_edge_db_6a)

                #try:
                #    start_edge_db_6b = ro.load_pattern_database('_start_edge_db_6b.npy')
                #except IOError:
                #    start_edge_db_6b = ro.generate_edges_pattern_database(start_state, edge_max_depth, ro.edge_pos_indices_6b, ro.edge_rot_indices_6b)
                #    ro.save_pattern_database('_start_edge_db_6b.npy', start_edge_db_6b)

                start_edge_db_6a = None
                start_edge_db_6b = None
                start_edge_db_10 = None

            if forward_heuristic_choice == HeuristicType.man:
                forward_heuristic = forward_manhattan_heuristic
            elif forward_heuristic_choice == HeuristicType.pattern:
                forward_heuristic = lambda state: pattern_database_lookup(state, goal_corner_db, goal_edge_db_6a, goal_edge_db_6b, goal_edge_db_10)
            elif forward_heuristic_choice == HeuristicType.zero:
                forward_heuristic = zero_heuristic
            else:
                raise Exception("Failed to identify type of forward heuristic")

            if backward_heuristic_choice == HeuristicType.man:
                backward_heuristic = lambda state: manhattan_heuristic(state, start_state)
            elif backward_heuristic_choice == HeuristicType.pattern:
                backward_heuristic = lambda state: pattern_database_lookup(state, start_corner_db, start_edge_db_6a, start_edge_db_6b, start_edge_db_10)
            elif backward_heuristic_choice == HeuristicType.zero:
                backward_heuristic = zero_heuristic
            else:
                raise Exception("Failed to identify type of backward heuristic")

            faces, rotations, searched = algorithm(start_state, ro.get_cube(), forward_heuristic, backward_heuristic)

            size = len(faces)
            print(f"Moves required to solve ({size}):")
            for i in range(size - 1, -1, -1):
                print(ro.Face(faces[i]).__repr__() + ro.Rotation(rotations[i]).__repr__())
            print(f"Explored {searched} nodes")
            return size, searched, time.perf_counter() - start
    finally:
        print("Finished", time.perf_counter() - start)


def load_cube(file: str):
    with open(file) as f:
        output = f.read()
        print(output)
        state = ro.scramble(output)
        return state


def explore_search(heuristic_choice, reverse_heuristic, solution_length, iterations=100):
    dibbs_results = []
    astar_results = []
    mm_results = []
    dibbs_time = []
    astar_time = []
    mm_time = []

    while len(dibbs_results) < iterations:
        try:
            os.remove("_start_corner_db.npy")
        except FileNotFoundError:
            pass
        start_state = ro.random_scramble(solution_length)
        size, searched, time_taken = asymmetric_search(Mode.search, heuristic_choice, reverse_heuristic, AlgorithmType.astar, start_state)
        if size == solution_length:
            astar_results.append(searched)
            astar_time.append(time_taken)
            size, searched, time_taken = asymmetric_search(Mode.search, heuristic_choice, reverse_heuristic, AlgorithmType.dibbs, start_state)
            assert (size == solution_length)
            dibbs_results.append(searched)
            dibbs_time.append(time_taken)
            size, searched, time_taken = asymmetric_search(Mode.search, heuristic_choice, reverse_heuristic, AlgorithmType.mm, start_state)
            assert(size == solution_length)
            mm_results.append(searched)
            mm_time.append(time_taken)

    print("DIBBS:", dibbs_results)
    print(dibbs_time)
    print("A*:", astar_results)
    print(astar_time)
    print("MM:", mm_results)
    print(mm_time)
    print("DIBBS:", np.mean(dibbs_results), "A*:", np.mean(astar_results), "MM:", np.mean(mm_results))
    print("TIME: DIBBS:", np.mean(dibbs_time), "A*:", np.mean(astar_time), "MM:", np.mean(mm_time))


if __name__ == "__main__":
    file = 'test_file.txt'
    mode = Mode.search
    forward_heuristic_choice = HeuristicType.pattern
    reverse_heuristic_choice = HeuristicType.man
    algorithm_choice = AlgorithmType.astar
    solution_length = 9
    iterations = 100

    asymmetric_search(mode, forward_heuristic_choice, reverse_heuristic_choice, algorithm_choice, load_cube(file))
    explore_search(forward_heuristic_choice, reverse_heuristic_choice, solution_length, iterations)