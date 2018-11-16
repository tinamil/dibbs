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


import _khash_ffi
from numba import cffi_support

cffi_support.register_module(_khash_ffi)

khash_init = _khash_ffi.lib.khash_int2int_init
khash_get = _khash_ffi.lib.khash_int2int_get
khash_set = _khash_ffi.lib.khash_int2int_set
khash_destroy = _khash_ffi.lib.khash_int2int_destroy

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
__corner_rot_indices = np.array([1, 5, 11, 15, 25, 29, 35, 39], dtype=np.uint8)

__factorial_lookup = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')


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


@enum.unique
class Rotation(enum.IntEnum):
    clockwise = 0
    counterclockwise = 1
    half = 2


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
            state[i] = __turn_position_lookup[state[i]][rotation_index]
    else:
        for i in range(0, 40, 2):
            if __turn_lookup[state[i]][face]:
                if __corner_booleans[i]:
                    state[i + 1] = __corner_rotation[face][state[i + 1]]
                elif face == Face.left or face == Face.right:
                    state[i + 1] = 1 - state[i + 1]

                state[i] = __turn_position_lookup[state[i]][rotation_index]
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
    inversions = np.zeros(8, dtype=np.uint8)
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


# @jit(nopython=True)
# def get_cube_index(state):
#     '''Count the number of inversions'''
#     size = 19
#     inversions = np.zeros(size, dtype=np.uint8)
#     for i in range(size):
#         if i % 2 == 1: continue
#         for j in range(i + 2, size+1):
#             if j % 2 == 1: continue
#             if state[i] > state[j]:
#                 inversions[i//2] += 1
#
#     cube_index = 0
#     for i in range(size):
#         cube_index += inversions[i] * fast_factorial(size-i)
#
#     '''Index into the specific rotation that we're in'''
#     cube_index *= 4478976
#
#     '''View the odd (rotation) corner indices then convert them from a base 3 to base 10 number'''
#     corners = 6
#     edges = 10
#     for i in range(size):
#         if i % 2 == 0: continue
#         if np_isin(i, __corner_rot_indices):
#             cube_index += state[i] * 3**corners
#             corners -= 1
#         else:
#             cube_index += state[i] * 2**edges
#             edges -= 1
#
#     return cube_index
#
#
# @njit
# def np_isin(element, array):
#     for x in array:
#         if element == x:
#             return True
#     return False


@jit(nopython=True)
def generate_pattern_database(state):
    queue = list()
    queue.append((state, np.uint8(0)))

    # 8 corners for 8 positions, 7 of which can have 3 unique rotations, 88179840 possibilities
    all_corners = np.uint32(88179840+1)
    pattern_lookup = np.full(all_corners, -1, dtype=np.int8)
    pattern_lookup[get_corner_index(state)] = np.int8(0)
    found_index_stack = np.full(all_corners, 100, dtype=np.uint8)
    id_depth = np.uint8(1)
    count = np.uint32(1)
    new_state = state
    new_state_index = np.uint32(0)
    new_state_depth = np.uint8(0)
    next_state = state
    depth = np.uint8(0)
    while count < all_corners:

        if len(queue) == 0:
            id_depth += np.uint8(1)
            found_index_stack = np.full(all_corners, 100, dtype=np.uint8)
            queue.append((state, np.uint8(0)))
            print("Incrementing id-depth to", id_depth)

        next_state, depth = queue.pop()
        for face in range(6):
            for rotation in range(3):
                new_state = rotate(np.copy(next_state), face, rotation)
                new_state_index = get_corner_index(new_state)
                new_state_depth = depth + np.uint8(1)
                if new_state_depth == id_depth and pattern_lookup[new_state_index] == np.int8(-1):
                    pattern_lookup[new_state_index] = new_state_depth
                    count += np.uint64(1)
                    if count % 100000 == 0 or count > 88100000:
                        print(count, new_state_depth, len(queue))
                elif new_state_depth < id_depth and new_state_depth < found_index_stack[new_state_index]:
                    found_index_stack[new_state_index] = new_state_depth
                    queue.append((new_state, new_state_depth))

    test_count = 0
    for x in range(all_corners):
        if pattern_lookup[x] == -1:
            test_count += 1
    assert (test_count == all_corners - count)

    return pattern_lookup


def load_pattern_database(file: str):
    with open(f'{file}.pkl', 'rb') as db_file:
        return pickle.load(db_file)


def save_pattern_database(file: str, db):
    with open(f'{file}.pkl', 'wb') as db_file:
        pickle.dump(db, db_file, pickle.HIGHEST_PROTOCOL)


__goal = np.array([0, 0, 1, 0, 2, 0, 3, 0,
                   4, 0, 5, 0, 6, 0, 7, 0,
                   8, 0, 9, 0, 10, 0, 11, 0,
                   12, 0, 13, 0, 14, 0, 15, 0,
                   16, 0, 17, 0, 18, 0, 19, 0], dtype=np.uint8)


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

