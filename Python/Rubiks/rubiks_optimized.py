import copy
import enum
import numpy as np
import pickle
from math import factorial
import time

# Uses facelet numbering defined at http://kociemba.org/cube.htm
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

class Rubiks:
    __goal = np.array([ 0, 0,  1, 0,  2, 0,  3, 0,
                        4, 0,  5, 0,  6, 0,  7, 0,
                        8, 0,  9, 0, 10, 0, 11, 0,
                       12, 0, 13, 0, 14, 0, 15, 0,
                       16, 0, 17, 0, 18, 0, 19, 0 ], dtype=np.uint8)

    __corner_booleans = np.array([True, True, False, False, True, True, False, False,
                                 False, False, True, True, False, False, True, True,
                                 False, False, False, False, False, False, False, False,
                                 True, True, False, False, True, True, False, False,
                                 False, False, True, True, False, False, True, True])

    corner_pos_indices = np.array([0, 4, 10, 14, 24, 28, 34, 38], dtype=np.uint8)
    corner_rot_indices = np.array([1, 5, 11, 15, 25, 29, 35, 39], dtype=np.uint8)

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
                                  [2, 1, 0]   # Right
                                  ], dtype=np.uint8)

    def __init__(self):
        self.__state = copy.copy(Rubiks.__goal)

    def __eq__(self, other):
        if other is Rubiks:
            return np.array_equal(self.__state, other.__state)
        else:
            return False

    def __hash__(self):
        return hash(self.__state.tobytes())

    def is_objective(self):
        return np.array_equal(self.__state, self.__goal)

    @classmethod
    def get_corner_index(cls, state):
        '''
        Gets the unique index of the corners of this cube.
        Finds the permutation of 8 corner cubies using a factorial number system by counting the number of inversions per corner. https://en.wikipedia.org/wiki/Factorial_number_system
        Each given permutation of corners has 2187 possible rotations of corners (3^7), so multiply by 2187 and then calculate the rotation by assuming each rotation is a digit in base 3.

        :return:
        '''
        '''Select all of the even (position) corner indices'''

        corners = state[Rubiks.corner_pos_indices]

        '''Count the number of inversions in the corner table per element'''
        inversions = np.zeros(8, dtype=np.uint8)
        for i in range(7):
            for j in range(i+1, 8):
                if corners[i] > corners[j]:
                    inversions[i] += 1

        corner_index = inversions[0] * 5040  # 7 * 6 * 5 * 4 * 3 * 2 * 1
        corner_index += inversions[1] * 720  # 6 * 5 * 4 * 3 * 2 * 1
        corner_index += inversions[2] * 120  # 5 * 4 * 3 * 2 * 1
        corner_index += inversions[3] * 24   # 4 * 3 * 2 * 1
        corner_index += inversions[4] * 6    # 3 * 2 * 1
        corner_index += inversions[5] * 2    # 2 * 1
        corner_index += inversions[6]        # 1
        #corner_index += inversions[7]       # The last inversion is always 0

        '''Index into the specific corner rotation that we're in'''
        corner_index *= 2187

        '''View the odd (rotation) corner indices then convert them from a base 3 to base 10 number'''
        corners = state[Rubiks.corner_rot_indices]
        corner_index += corners[0] * 729  # 3^6
        corner_index += corners[1] * 243  # 3^5
        corner_index += corners[2] * 81   # 3^4
        corner_index += corners[3] * 27   # 3^3
        corner_index += corners[4] * 9    # 3^2
        corner_index += corners[5] * 3    # 3^1
        corner_index += corners[6]        # 3^
        return corner_index

    @staticmethod
    def rotate(state, face, rotation):
        rotation_index = face * rotation
        # If half rotation then no change in rotations
        if rotation == Rotation.half:
            for i in np.arange(0, 40, 2):
                state[i] = Rubiks.__turn_position_lookup[state[i]][rotation_index]
        else:
            for i in np.arange(0, 40, 2):
                if Rubiks.__turn_lookup[state[i]][rotation_index]:
                    if Rubiks.__corner_booleans[i]:
                        state[i+1] = Rubiks.__corner_rotation[face][state[i+1]]
                    elif face == Face.left or face == Face.right:
                        state[i+1] = 1 - state[i+1]

                    state[i] = Rubiks.__turn_position_lookup[state[i]][rotation_index]
        return state

    @classmethod
    def generate_pattern_database(cls, file: str):
        state = Rubiks().__state

        queue = list()
        queue.append((state, 0, -1, -1))

        # 8 corners for 8 positions, 7 of which can have 3 unique rotations, 88179840 possibilities
        all_corners = factorial(8) * 3**7
        pattern_lookup = np.empty(all_corners, dtype=np.uint8)
        pattern_lookup[cls.get_corner_index(state)] = 0
        found_index = np.zeros(all_corners, dtype=np.bool)
        found_index_stack = np.full(all_corners, 100, dtype=np.uint8)
        found_index[cls.get_corner_index(state)] = True
        id_depth = 1
        count = 1
        while count < all_corners and count < 50000:

            if len(queue) == 0:
                id_depth += 1
                found_index_stack = np.full(all_corners, 100, dtype=np.uint8)
                queue.append((state, 0, -1, -1))
                print(f"Incrementing id-depth to {id_depth}")

            next_state, depth, last_face, last_rotation = queue.pop()

            for face in range(6):
                for rotation in range(3):
                    if last_face == face:
                        # Avoid rotating back to the previous state
                        if (last_rotation == 2 and rotation == 2) \
                                or (last_rotation == 0 and rotation == 1) \
                                or (last_rotation == 1 and rotation == 0):
                            continue
                    new_state = cls.rotate(next_state, face, rotation)
                    new_state_index = cls.get_corner_index(new_state)
                    new_state_depth = depth + 1
                    if new_state_depth < found_index_stack[new_state_index]:
                        found_index_stack[new_state_index] = depth
                        if new_state_depth == id_depth:
                            if not found_index[new_state_index]:
                                pattern_lookup[new_state_index] = new_state_depth
                                found_index[new_state_index] = True
                                count += 1
                                if count % 10000 == 0:
                                    print(count, depth + 1, len(queue))
                        else:
                            queue.append((new_state, depth + 1, face, rotation))

        while len(queue) > 0:
            next_state, depth, last_face, last_rotation = queue.pop()
            new_state_index = cls.get_corner_index(next_state)
            if new_state_index not in pattern_lookup or pattern_lookup[new_state_index] > depth:
                if new_state_index in pattern_lookup:
                    print("Found more values in the stack with lower depth, BUG!!!")
                    pattern_lookup[new_state_index] = depth

        with open(f'{file}.pkl', 'wb') as f:
            pickle.dump(pattern_lookup, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pattern_database(file: str):
        with open(f'{file}.pkl', 'rb') as f:
            return pickle.load(f)


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
    Rubiks.generate_pattern_database('database')
    print("Finished: ", time.perf_counter() - start)
    #pattern_db = Rubiks.load_pattern_database('database')
    pass
