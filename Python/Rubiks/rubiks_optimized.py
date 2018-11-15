import copy
import enum
import numpy as np
import pickle
import time
from numba import jit, njit

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


@jit(nopython=True)
def rotate(state, face, rotation):
    rotation_index = face * rotation
    # If half rotation then no change in rotations
    if rotation == Rotation.half:
        for i in range(40):
            if i % 2 == 0:
                state[i] = __turn_position_lookup[state[i]][rotation_index]
    else:
        for i in range(40):
            if i % 2 == 0:
                if __turn_lookup[state[i]][rotation_index]:
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


@jit(nopython=True)
def generate_pattern_database(state):
    queue = list()
    queue.append((state, 0))

    # 8 corners for 8 positions, 7 of which can have 3 unique rotations, 88179840 possibilities
    all_corners = 88179840
    pattern_lookup = np.empty(all_corners, dtype=np.uint8)
    pattern_lookup[get_corner_index(state)] = 0
    found_index = np.zeros(all_corners, dtype=np.uint8)
    found_index_stack = np.full(all_corners, 100, dtype=np.uint8)
    found_index[get_corner_index(state)] = 1
    id_depth = 1
    count = 1
    while count < all_corners:

        if len(queue) == 0:
            id_depth += 1
            found_index_stack = np.full(all_corners, 100, dtype=np.uint8)
            queue.append((state, 0))
            print("Incrementing id-depth to", id_depth)

        next_state, depth = queue.pop()
        for face in range(6):
            for rotation in range(3):
                new_state = rotate(np.copy(next_state), face, rotation)
                new_state_index = get_corner_index(new_state)
                new_state_depth = depth + 1
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
    return np.array_equal(cube, __goal)


@enum.unique
class Face(enum.IntEnum):
    front = 0
    up = 1
    left = 2
    back = 3
    down = 4
    right = 5

    def __repr__(self):
        return self.name


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

    def __repr__(self):
        return self.name


if __name__ == "__main__":
    start = time.perf_counter()
    db = generate_pattern_database(get_cube())
    print("Finished: ", time.perf_counter() - start)
    # pattern_db = Rubiks.load_pattern_database('database')
    pass
