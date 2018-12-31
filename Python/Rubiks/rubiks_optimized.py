import copy
import enum
import numpy as np
import pickle
import time
from numba import jit, njit
import random
from typing import List, Dict, Tuple


# 6 center cubies are fixed position (rotation doesn't change) and define the solution color for that face
# 8 corner cubies are defined by an integer defining their rotation: 0-2186 (3^7-1, 8th cube is defined by the other 7)
#                     and an integer defining their positions: 0-40319 (8! - 1)
# 12 edge cubies have rotation integer 0-2047 (2^11-1) and position integer 0-12!-1

# A cube state is an array of 20 sets of (position, rotation), values ranging from 0-19 and 0-2 respectively.  Rotations are 0-1 for edge pieces.  Center face pieces are fixed and not included.
# Cubies are numbered from 0 to 19 throughout the code, as shown here ( credit to https://github.com/brownan/Rubiks-Cube-Solver/blob/master/cube.h )
#     5----6----7
#     |         |\
#     3    Y    4 \
#     |         |  \
#     0----1----2   \
#      \             \
#       \   10---R---11
#        \  |         |\
#         \ B    X    G \
#          \|         |  \
#           8----O----9   \
#            \             \
#             \   17--18---19
#              \  |         |
#               \ 15   W   16
#                \|         |
#                 12--13---14


# import _khash_ffi
# from numba import cffi_support
#
# cffi_support.register_module(_khash_ffi)
#
# khash_init = _khash_ffi.lib.khash_int2int_init
# khash_get = _khash_ffi.lib.khash_int2int_get
# khash_set = _khash_ffi.lib.khash_int2int_set
# khash_destroy = _khash_ffi.lib.khash_int2int_destroy


__corner_cubies = np.array([0, 2, 5, 7, 12, 14, 17, 19], dtype=np.uint8)
__edge_cubies = np.array([1, 3, 4, 6, 8, 9, 10, 11, 13, 15, 16, 18], dtype=np.uint8)


'''Given a cubie [0-19] and a rotation [0-17] being performed, returns the new position of that cubie'''
__turn_position_lookup = np.array([[0, 0, 5, 2, 12, 0, 0, 0, 12, 5, 2, 0, 0, 0, 17, 7, 14, 0],
                                   [1, 1, 1, 4, 8, 1, 1, 1, 1, 3, 9, 1, 1, 1, 1, 6, 13, 1],
                                   [2, 2, 2, 7, 0, 14, 2, 2, 2, 0, 14, 7, 2, 2, 2, 5, 12, 19],
                                   [3, 3, 10, 1, 3, 3, 3, 3, 8, 6, 3, 3, 3, 3, 15, 4, 3, 3],
                                   [4, 4, 4, 6, 4, 9, 4, 4, 4, 1, 4, 11, 4, 4, 4, 3, 4, 16],
                                   [5, 7, 17, 0, 5, 5, 5, 17, 0, 7, 5, 5, 5, 19, 12, 2, 5, 5],
                                   [6, 11, 6, 3, 6, 6, 6, 10, 6, 4, 6, 6, 6, 18, 6, 1, 6, 6],
                                   [7, 19, 7, 5, 7, 2, 7, 5, 7, 2, 7, 19, 7, 17, 7, 0, 7, 14],
                                   [8, 8, 3, 8, 13, 8, 8, 8, 15, 8, 1, 8, 8, 8, 10, 8, 9, 8],
                                   [9, 9, 9, 9, 1, 16, 9, 9, 9, 9, 13, 4, 9, 9, 9, 9, 8, 11],
                                   [10, 6, 15, 10, 10, 10, 10, 18, 3, 10, 10, 10, 10, 11, 8, 10, 10, 10],
                                   [11, 18, 11, 11, 11, 4, 11, 6, 11, 11, 11, 16, 11, 10, 11, 11, 11, 9],
                                   [17, 12, 0, 12, 14, 12, 14, 12, 17, 12, 0, 12, 19, 12, 5, 12, 2, 12],
                                   [15, 13, 13, 13, 9, 13, 16, 13, 13, 13, 8, 13, 18, 13, 13, 13, 1, 13],
                                   [12, 14, 14, 14, 2, 19, 19, 14, 14, 14, 12, 2, 17, 14, 14, 14, 0, 7],
                                   [18, 15, 8, 15, 15, 15, 13, 15, 10, 15, 15, 15, 16, 15, 3, 15, 15, 15],
                                   [13, 16, 16, 16, 16, 11, 18, 16, 16, 16, 16, 9, 15, 16, 16, 16, 16, 4],
                                   [19, 5, 12, 17, 17, 17, 12, 19, 5, 17, 17, 17, 14, 7, 0, 17, 17, 17],
                                   [16, 10, 18, 18, 18, 18, 15, 11, 18, 18, 18, 18, 13, 6, 18, 18, 18, 18],
                                   [14, 17, 19, 19, 19, 7, 17, 7, 19, 19, 19, 14, 12, 5, 19, 19, 19, 2]], dtype=np.uint8)

'''Given a cubie [0-19] and a face [0-5], returns true if the cubie is being turned and 0 otherwise'''
__turn_lookup = np.array([[False, False, True, True, True, False],
                          [False, False, False, True, True, False],
                          [False, False, False, True, True, True],
                          [False, False, True, True, False, False],
                          [False, False, False, True, False, True],
                          [False, True, True, True, False, False],
                          [False, True, False, True, False, False],
                          [False, True, False, True, False, True],
                          [False, False, True, False, True, False],
                          [False, False, False, False, True, True],
                          [False, True, True, False, False, False],
                          [False, True, False, False, False, True],
                          [True, False, True, False, True, False],
                          [True, False, False, False, True, False],
                          [True, False, False, False, True, True],
                          [True, False, True, False, False, False],
                          [True, False, False, False, False, True],
                          [True, True, True, False, False, False],
                          [True, True, False, False, False, False],
                          [True, True, False, False, False, True]])

__corner_rotation = np.array([[0, 2, 1],  # Front
                              [1, 0, 2],  # Top
                              [2, 1, 0],  # Left
                              [0, 2, 1],  # Back
                              [1, 0, 2],  # Down
                              [2, 1, 0]  # Right
                              ], dtype=np.uint8)

__corner_booleans = np.array([True, True, False, False, True, True, False, False,
                              False, False, True, True, False, False, True, True,
                              False, False, False, False, False, False, False, False,
                              True, True, False, False, True, True, False, False,
                              False, False, True, True, False, False, True, True])

__corner_pos_indices = np.array([0, 4, 10, 14, 24, 28, 34, 38], dtype=np.uint8)
__corner_rot_indices = __corner_pos_indices + 1

edge_pos_indices_6a = np.array([2, 6, 8, 12, 16, 18], dtype=np.uint8)
edge_rot_indices_6a = edge_pos_indices_6a + 1
edge_pos_indices_6b = np.array([20, 22, 26, 30, 32, 36], dtype=np.uint8)
edge_rot_indices_6b = edge_pos_indices_6b + 1
edge_pos_indices_10 = np.array([2, 6, 8, 12, 16, 18, 20, 22, 26, 30], dtype=np.uint8)
edge_rot_indices_10 = edge_pos_indices_10 + 1
__edge_translations = np.array([0, 0, 0, 1, 2, 0, 3, 0, 4, 5, 6, 7, 0, 8, 0, 9, 10, 0, 11, 0], dtype=np.uint8)

__factorial_lookup = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype=np.uint64)


__goal = np.array([0, 0, 1, 0, 2, 0, 3, 0,
                   4, 0, 5, 0, 6, 0, 7, 0,
                   8, 0, 9, 0, 10, 0, 11, 0,
                   12, 0, 13, 0, 14, 0, 15, 0,
                   16, 0, 17, 0, 18, 0, 19, 0], dtype=np.uint8)


@njit
def fast_factorial(n):
    if n > 20:
        raise ValueError
    return __factorial_lookup[n]


@enum.unique
class Face(enum.IntEnum):
    front = 0
    up = 1
    left = 2
    back = 3
    down = 4
    right = 5

    def __repr__(self):
        return self.name.upper()[0]


@enum.unique
class Rotation(enum.IntEnum):
    clockwise = 0
    counterclockwise = 1
    half = 2

    def __repr__(self):
        if Rotation.clockwise == self:
            return ""
        elif Rotation.counterclockwise == self:
            return "'"
        else:
            return "2"


@enum.unique
class Color(enum.IntEnum):
    orange = Face.up
    red = Face.down
    white = Face.front
    blue = Face.right
    yellow = Face.back
    green = Face.left
    blank = -1


@njit()
def rotate(old_state: np.ndarray, face: int, rotation: int):
    state = np.copy(old_state)
    rotation_index = 6 * rotation + face
    # If half rotation then no change in rotations
    if rotation == Rotation.half:
        for i in range(0, 40, 2):
            state[i] = __turn_position_lookup[state[i], rotation_index]
    else:
        for i in range(0, 40, 2):
            if __turn_lookup[state[i], face]:
                if __corner_booleans[i]:
                    state[i + 1] = __corner_rotation[face, state[i + 1]]
                elif face == Face.left or face == Face.right:
                    state[i + 1] = 1 - state[i + 1]

                state[i] = __turn_position_lookup[state[i], rotation_index]
    return state


@jit(nopython=True)
def get_corner_index(state):
    '''
    Gets the unique index of the corners of this cube.
    Finds the permutation of 8 corner cubies using a factorial number system by counting the number of inversions per corner. https://en.wikipedia.org/wiki/Factorial_number_system
    Each given permutation of corners has 2187 possible rotations of corners (3^7), so multiply by 2187 and then calculate the rotation by assuming each rotation is a digit in base 3.

    :return:
    '''
    '''Select all of the even (position) corner indices'''

    corners = state[__corner_pos_indices]

    '''Count the number of inversions in the corner table per element'''
    inversions = np.zeros(7, dtype=np.uint8)
    for i in range(7):
        for j in range(i + 1, 8):
            if corners[i] > corners[j]:
                inversions[i] += 1

    corner_index = inversions[0] * 5040  # 7 * 6 * 5 * 4 * 3 * 2 * 1
    corner_index += inversions[1] * 720  # 6 * 5 * 4 * 3 * 2 * 1
    corner_index += inversions[2] * 120  # 5 * 4 * 3 * 2 * 1
    corner_index += inversions[3] * 24  # 4 * 3 * 2 * 1
    corner_index += inversions[4] * 6  # 3 * 2 * 1
    corner_index += inversions[5] * 2  # 2 * 1
    corner_index += inversions[6]  # 1
    # corner_index += inversions[7]       # The last inversion is always 0

    '''Index into the specific corner rotation that we're in'''
    corner_index *= 2187

    '''View the odd (rotation) corner indices then convert them from a base 3 to base 10 number'''
    corners = state[__corner_rot_indices]
    corner_index += corners[0] * 729  # 3^6
    corner_index += corners[1] * 243  # 3^5
    corner_index += corners[2] * 81  # 3^4
    corner_index += corners[3] * 27  # 3^3
    corner_index += corners[4] * 9  # 3^2
    corner_index += corners[5] * 3  # 3^1
    corner_index += corners[6]  # 3^
    return corner_index


@njit
def generate_corners_pattern_database(state, max_depth):
    print("Generating corners db")
    queue = list()
    queue.append((state, np.uint8(0)))

    # 8 corners for 8 positions, 7 of which can have 3 unique rotations, 88179840 possibilities
    all_corners = np.uint32(88179840)
    pattern_lookup = np.full(all_corners, max_depth, dtype=np.int8)
    pattern_lookup[get_corner_index(state)] = np.int8(0)
    found_index_stack = np.full(all_corners, max_depth, dtype=np.uint8)
    id_depth = np.uint8(1)
    count = np.uint32(1)
    new_state = state
    new_state_index = np.uint32(0)
    new_state_depth = np.uint8(0)
    next_state = state
    depth = np.uint8(0)
    while count < all_corners and id_depth < max_depth:

        if len(queue) == 0:
            id_depth += np.uint8(1)
            found_index_stack = np.full(all_corners, max_depth, dtype=np.uint8)
            queue.append((state, np.uint8(0)))
            print("Incrementing id-depth to", id_depth)

        next_state, depth = queue.pop()
        for face in range(6):
            for rotation in range(3):
                new_state = rotate(np.copy(next_state), face, rotation)
                new_state_index = get_corner_index(new_state)
                new_state_depth = depth + np.uint8(1)
                if new_state_depth == id_depth and pattern_lookup[new_state_index] == max_depth:
                    pattern_lookup[new_state_index] = new_state_depth
                    count += np.uint64(1)
                    if count % 100000 == 0 or count > 88100000:
                        print(count, new_state_depth, len(queue))
                elif new_state_depth < id_depth and new_state_depth < found_index_stack[new_state_index]:
                    found_index_stack[new_state_index] = new_state_depth
                    queue.append((new_state, new_state_depth))

    return pattern_lookup


@njit
def get_edge_index(state, edge_pos_indices: np.ndarray, edge_rot_indices: np.ndarray) -> np.uint64:
    """
    #Edge index = perm_number * 2^x + orientation number
    perm_number = (n-(i+1))!/(n-r)! * #unused_digits
    Where i is the index of the number in the smaller set, n is the size of the set to choose from and r is the size
    of the smaller set and number of preceding digits are the number of digits preceding the current digit in the
    original set that have yet to be used. -https://www.doc.ic.ac.uk/teaching/distinguished-projects/2015/l.hoang.pdf
    """
    full_size = 12  # Total number of edges, always 12 in a 3x3x3 Rubik's Cube
    edge_pos = __edge_translations[state[edge_pos_indices]]
    size = len(edge_pos)
    permute_number = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        permute_number[i] = edge_pos[i]
        for j in range(0, i):
            if edge_pos[j] < edge_pos[i]:
                permute_number[i] -= 1

    edge_index = np.uint64(0)
    small_size = __factorial_lookup[full_size - size]
    for i in range(size):
        edge_index += permute_number[i] * (__factorial_lookup[full_size - i - 1] // small_size)

    '''Index into the specific edge rotation that we're in'''
    edge_index *= np.uint64(2**size)

    '''View the odd (rotation) edge indices then convert them from a base 2 to base 10 number'''
    edge_rots = state[edge_rot_indices]
    for i in range(size):
        edge_index += np.uint64(edge_rots[i] * 1 << (size - i - 1))

    return edge_index


@njit
def npr(n, r):
    return __factorial_lookup[n] / __factorial_lookup[n-r]


def inverse_rotation(rotation):
    if rotation is None:
        return None
    if rotation == 2:
        return 2
    else:
        return 1 - rotation


@njit
def generate_edges_pattern_database(state, max_depth, edge_pos_indices, edge_rot_indices):
    print("Generating edges db")
    queue = list()
    queue.append((state, np.uint8(0), -1))

    # 12 permute x positions * 2^x rotations
    all_edges = np.uint64(npr(12, len(edge_pos_indices)) * 2**(len(edge_pos_indices)))
    print(all_edges)
    pattern_lookup = np.full(shape=all_edges, fill_value=max_depth, dtype=np.int8)
    new_state_index = get_edge_index(state, edge_pos_indices, edge_rot_indices)
    pattern_lookup[new_state_index] = 0
    found_index_stack = np.full(shape=all_edges, fill_value=max_depth, dtype=np.uint8)
    id_depth = np.uint8(1)
    count = np.uint64(1)
    new_state = state
    new_state_depth = np.uint8(0)
    next_state = state
    depth = np.uint8(0)
    while count < all_edges and id_depth < max_depth:

        if len(queue) == 0:
            id_depth += np.uint8(1)
            found_index_stack = np.full(shape=all_edges, fill_value=max_depth, dtype=np.uint8)
            queue.append((state, np.uint8(0), -1))
            print("Incrementing id-depth to", id_depth)

        next_state, depth, last_face = queue.pop()
        for face in range(6):

            if last_face == face:
                continue
            if last_face == Face.back and face == Face.front:
                continue
            if last_face == Face.right and face == Face.left:
                continue
            if last_face == Face.down and face == Face.up:
                continue

            for rotation in range(3):

                new_state = rotate(np.copy(next_state), face, rotation)
                new_state_index = get_edge_index(new_state, edge_pos_indices, edge_rot_indices)
                new_state_depth = depth + np.uint8(1)
                if new_state_depth == id_depth and pattern_lookup[new_state_index] == max_depth:
                    pattern_lookup[new_state_index] = new_state_depth
                    count += np.uint64(1)
                    if count % 100000 == 0:
                        print(count, new_state_depth, len(queue))
                elif new_state_depth < id_depth and new_state_depth < found_index_stack[new_state_index]:
                    found_index_stack[new_state_index] = new_state_depth
                    queue.append((new_state, new_state_depth, face))

    return pattern_lookup


@njit
def generate_edges_memmap(state, max_depth, edge_pos_indices, edge_rot_indices, pattern_lookup):
    print("Generating edges db")
    queue = list()
    queue.append((state, np.uint8(0), -1))

    # 12 permute x positions * 2^x rotations
    all_edges = np.uint64(npr(12, len(edge_pos_indices)) * 2**(len(edge_pos_indices)))
    print(all_edges)

    new_state_index = get_edge_index(state, edge_pos_indices, edge_rot_indices)
    pattern_lookup[new_state_index] = 0
    #found_index_stack = np.full(shape=all_edges, fill_value=max_depth, dtype=np.uint8)
    id_depth = np.uint8(1)
    count = np.uint64(1)
    new_state = state
    new_state_depth = np.uint8(0)
    next_state = state
    depth = np.uint8(0)
    while count < all_edges and id_depth < max_depth:

        if len(queue) == 0:
            id_depth += np.uint8(1)
            #found_index_stack = np.full(shape=all_edges, fill_value=max_depth, dtype=np.uint8)
            queue.append((state, np.uint8(0), -1))
            print("Incrementing id-depth to", id_depth)

        next_state, depth, last_face = queue.pop()
        for face in range(6):

            if last_face == face:
                continue
            if last_face == Face.back and face == Face.front:
                continue
            if last_face == Face.right and face == Face.left:
                continue
            if last_face == Face.down and face == Face.up:
                continue

            for rotation in range(3):

                new_state = rotate(np.copy(next_state), face, rotation)
                new_state_index = get_edge_index(new_state, edge_pos_indices, edge_rot_indices)
                new_state_depth = depth + np.uint8(1)
                if new_state_depth == id_depth and pattern_lookup[new_state_index] == max_depth:
                    pattern_lookup[new_state_index] = new_state_depth
                    count += np.uint64(1)
                    if count % 10000 == 0:
                        print(count, new_state_depth, len(queue))
                elif new_state_depth < id_depth:# and new_state_depth < found_index_stack[new_state_index]:
                    #found_index_stack[new_state_index] = new_state_depth
                    queue.append((new_state, new_state_depth, face))

    return pattern_lookup



def convert_khash_to_dict(db, indices, max_depth):
    new_edge_db = dict()
    for idx in indices:
        new_edge_db[idx] = khash_get(db, idx, max_depth)
    return new_edge_db


def convert_dict_to_list(db):
    new_db = []
    for key, val in db.items():
        new_db.append((key, val))
    return new_db


def load_pattern_database(file: str):
    if file.endswith(".npy"):
        return np.load(file)
    else:
        with open(f'{file}', 'rb') as f:
            return pickle.load(f)


def save_pattern_database(file: str, db):

    if file.endswith(".npy"):
        np.save(f'{file}', db)
    else:
        with open(f'{file}', 'wb') as f:
            pickle.dump(db, f, pickle.DEFAULT_PROTOCOL)


@njit
def get_cube():
    return np.copy(__goal)


@njit
def is_solved(cube):
    for idx, val in enumerate(__goal):
        if val != cube[idx]:
            return False
    return True


@njit
def is_solved_cubie(cube, cubie, solution):
    pos_index = cubie * 2
    rot_index = pos_index + 1
    if solution[pos_index] != cube[pos_index] or solution[rot_index] != cube[rot_index]:
        return False
    return True


@njit
def generate_random_cube():
    cube = get_cube()
    for i in range(1000):
        face = random.randrange(6)
        rot = random.randrange(3)
        cube = rotate(cube, face, rot)
    return cube


def scramble(notation: str) -> np.ndarray:
    start = get_cube()
    moves = notation.split()
    for move in moves:
        face, rotation = convert(move)
        start = rotate(start, int(face), int(rotation))
    return start


def translate_face(input_face: str) -> Face:
    face = input_face.upper()
    if face == 'U':
        return Face.up
    if face == 'D':
        return Face.down
    if face == 'F':
        return Face.front
    if face == 'B':
        return Face.back
    if face == 'R':
        return Face.right
    if face == 'L':
        return Face.left

    raise ValueError("No face match for " + str(input_face))


def convert(notation_move: str) -> Tuple[Face, Rotation]:
    if len(notation_move) == 1:
        return translate_face(notation_move), Rotation.clockwise
    elif len(notation_move) == 2:
        face = translate_face(notation_move[0])
        if notation_move[1] == "'":
            rotation = Rotation.counterclockwise
        elif notation_move[1] == '2':
            rotation = Rotation.half
        else:
            raise ValueError("Unable to identify rotation type: " + str(notation_move))
        return face, rotation
    else:
        raise ValueError("Length is not 1 or 2 characters to convert notation: " + str(notation_move))


@njit
def manhattan_heuristic(state: np.ndarray, solution: np.ndarray):
    count = 0
    for idx in __edge_cubies:
        count += solve(state, idx, solution)
    count /= 4  # 4 edge cubies can be moved in each rotation
    return int(np.ceil(count))


@njit
def solve(state: np.ndarray, cubie: int, solution: np.ndarray):
    queue = list()
    starting_state = state
    queue.append((starting_state, 0))

    if is_solved_cubie(state, cubie, solution):
        return 0

    id_depth = 1
    while True:

        if len(queue) == 0:
            id_depth += 1
            queue.append((starting_state, 0))

        next_state, depth = queue.pop()

        for face in range(6):
            for rotation in range(3):
                new_state_base = rotate(next_state, face, rotation)
                new_state_depth = depth + 1
                new_state_cost = new_state_depth

                if new_state_cost > id_depth:
                    continue

                if is_solved_cubie(new_state_base, cubie, solution):
                    return new_state_depth

                queue.append((new_state_base, new_state_depth))
