import copy
import enum
import numpy as np
from collections import deque
from typing import Deque, Tuple
import pickle
from math import factorial
import psutil
import random
import pprofile

# Uses facelet numbering defined at http://kociemba.org/cube.htm
# 6 center cubies are fixed position (rotation doesn't change) and define the solution color for that face
# 8 corner cubies are defined by an integer defining their rotation: 0-2186 (3^7-1, 8th cube is defined by the other 7)
#                     and an integer defining their positions: 0-40319 (8! - 1)
# 12 edge cubies have rotation integer 0-2047 (2^11-1) and position integer 0-12!-1
# state is (corner position, edge position, corner rotation, edge rotation): (uint16, uint16, uint16, uint32)


@enum.unique
class Face(enum.IntEnum):
    up = 0
    right = 1
    front = 2
    down = 3
    left = 4
    back = 5

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


def translate_face(face):
    if face is 'U':
        return Face.up
    if face is 'D':
        return Face.down
    if face is 'F':
        return Face.front
    if face is 'B':
        return Face.back
    if face is 'R':
        return Face.right
    if face is 'L':
        return Face.left


def build_transform_lookups():
    move_rotations = [
        # Rotate U
        [
            # 90 CW
            [['U7', 'U4', 'U1', 'U8', 'U5', 'U2', 'U9', 'U6', 'U3'],
             ['B1', 'B2', 'B3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9'],
             ['R1', 'R2', 'R3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9'],
             ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9'],
             ['F1', 'F2', 'F3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9'],
             ['L1', 'L2', 'L3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9']],

            # -90 CCW
            [['U3', 'U6', 'U9', 'U2', 'U5', 'U8', 'U1', 'U4', 'U7'],
             ['F1', 'F2', 'F3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9'],
             ['L1', 'L2', 'L3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9'],
             ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9'],
             ['B1', 'B2', 'B3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9'],
             ['R1', 'R2', 'R3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9']]
        ],

        # Rotate R
        [
            # 90
            [['U1', 'U2', 'F3', 'U4', 'U5', 'F6', 'U7', 'U8', 'F9'],
             ['R7', 'R4', 'R1', 'R8', 'R5', 'R2', 'R9', 'R6', 'R3'],
             ['F1', 'F2', 'D3', 'F4', 'F5', 'D6', 'F7', 'F8', 'D9'],
             ['D1', 'D2', 'B7', 'D4', 'D5', 'B4', 'D7', 'D8', 'B1'],
             ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9'],
             ['U9', 'B2', 'B3', 'U6', 'B5', 'B6', 'U3', 'B8', 'B9']],

            # -90
            [['U1', 'U2', 'B7', 'U4', 'U5', 'B4', 'U7', 'U8', 'B1'],
             ['R3', 'R6', 'R9', 'R2', 'R5', 'R8', 'R1', 'R4', 'R7'],
             ['F1', 'F2', 'U3', 'F4', 'F5', 'U6', 'F7', 'F8', 'U9'],
             ['D1', 'D2', 'F3', 'D4', 'D5', 'F6', 'D7', 'D8', 'F9'],
             ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9'],
             ['D9', 'B2', 'B3', 'D6', 'B5', 'B6', 'D3', 'B8', 'B9']]
        ],

        # Rotate F
        [
            # 90
            [['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'L9', 'L6', 'L3'],
             ['U7', 'R2', 'R3', 'U8', 'R5', 'R6', 'U9', 'R8', 'R9'],
             ['F7', 'F4', 'F1', 'F8', 'F5', 'F2', 'F9', 'F6', 'F3'],
             ['R7', 'R4', 'R1', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9'],
             ['L1', 'L2', 'D1', 'L4', 'L5', 'D2', 'L7', 'L8', 'D3'],
             ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9']],

            # -90
            [['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'R1', 'R4', 'R7'],
             ['D3', 'R2', 'R3', 'D2', 'R5', 'R6', 'D1', 'R8', 'R9'],
             ['F3', 'F6', 'F9', 'F2', 'F5', 'F8', 'F1', 'F4', 'F7'],
             ['L3', 'L6', 'L9', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9'],
             ['L1', 'L2', 'U9', 'L4', 'L5', 'U8', 'L7', 'L8', 'U7'],
             ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9']]
        ],

        # Rotate D
        [
            # 90
            [['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9'],
             ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'F7', 'F8', 'F9'],
             ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'L7', 'L8', 'L9'],
             ['D7', 'D4', 'D1', 'D8', 'D5', 'D2', 'D9', 'D6', 'D3'],
             ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'B7', 'B8', 'B9'],
             ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'R7', 'R8', 'R9']],

            # -90
            [['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9'],
             ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'B7', 'B8', 'B9'],
             ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'R7', 'R8', 'R9'],
             ['D3', 'D6', 'D9', 'D2', 'D5', 'D8', 'D1', 'D4', 'D7'],
             ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'F7', 'F8', 'F9'],
             ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'L7', 'L8', 'L9']]
        ],

        # Rotate L
        [
            # 90
            [['B9', 'U2', 'U3', 'B6', 'U5', 'U6', 'B3', 'U8', 'U9'],
             ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9'],
             ['U1', 'F2', 'F3', 'U4', 'F5', 'F6', 'U7', 'F8', 'F9'],
             ['F1', 'D2', 'D3', 'F4', 'D5', 'D6', 'F7', 'D8', 'D9'],
             ['L7', 'L4', 'L1', 'L8', 'L5', 'L2', 'L9', 'L6', 'L3'],
             ['B1', 'B2', 'D7', 'B4', 'B5', 'D4', 'B7', 'B8', 'D1']],

            # -90
            [['F1', 'U2', 'U3', 'F4', 'U5', 'U6', 'F7', 'U8', 'U9'],
             ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9'],
             ['D1', 'F2', 'F3', 'D4', 'F5', 'F6', 'D7', 'F8', 'F9'],
             ['B9', 'D2', 'D3', 'B6', 'D5', 'D6', 'B3', 'D8', 'D9'],
             ['L3', 'L6', 'L9', 'L2', 'L5', 'L8', 'L1', 'L4', 'L7'],
             ['B1', 'B2', 'U7', 'B4', 'B5', 'U4', 'B7', 'B8', 'U1']]
        ],

        # Rotate B
        [
            # 90
            [['R3', 'R6', 'R9', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9'],
             ['R1', 'R2', 'D9', 'R4', 'R5', 'D8', 'R7', 'R8', 'D7'],
             ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9'],
             ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'L1', 'L4', 'L7'],
             ['U3', 'L2', 'L3', 'U2', 'L5', 'L6', 'U1', 'L8', 'L9'],
             ['B7', 'B4', 'B1', 'B8', 'B5', 'B2', 'B9', 'B6', 'B3']],

            # -90
            [['L7', 'L4', 'L1', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9'],
             ['R1', 'R2', 'U1', 'R4', 'R5', 'U2', 'R7', 'R8', 'U3'],
             ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9'],
             ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'R9', 'R6', 'R3'],
             ['D7', 'L2', 'L3', 'D8', 'L5', 'L6', 'D9', 'L8', 'L9'],
             ['B3', 'B6', 'B9', 'B2', 'B5', 'B8', 'B1', 'B4', 'B7']]
        ]
     ]

    lookups = []
    for plane_index, plane in enumerate(move_rotations):
        directions = []
        for direction_index, direction in enumerate(plane):
            index_face = []
            index_item = []
            assert (len(np.unique(move_rotations[plane_index][direction_index])) == 9*6)
            for face in direction:
                for item in face:
                    index_face.append(translate_face(item[0]))
                    index_item.append(int(item[1]) - 1)
            directions.append((np.array(index_face), np.array(index_item)))
        lookups.append(directions)
    return lookups


class Rubiks:
    __goal = np.empty([6, 9], dtype=np.uint8)
    for face in range(6):
        for row in range(9):
            __goal[face][row] = np.uint8(face)
    __transforms = build_transform_lookups()

    # All of the faces of each corner, except for up/right/front
    # because it's determined by the the other 7 corners faces
    __corner_indices = np.array([[True, False,  True, False, False, False,  True, False,  False],
                                 [False, False,  True, False, False, False,  True, False,  True],
                                 [True, False,  False, False, False, False,  True, False,  True],
                                 [True, False,  True, False, False, False,  True, False,  True],
                                 [True, False,  True, False, False, False,  True, False,  True],
                                 [True, False,  True, False, False, False,  True, False,  True]]).reshape(6, 9)

    def __init__(self, initial=None):
        if initial is None:
            self.__state = copy.copy(self.__goal)
        else:
            self.__state = initial.__state.copy()

    def __repr__(self):
        return str(np.array([list(map(Color, x)) for x in self.__state.tolist()], dtype=Color).reshape(6, 3, 3))

    def __eq__(self, other):
        if other is Rubiks:
            return np.array_equal(self.__state, other.__state)
        else:
            return False

    def __hash__(self):
        return hash(self.__state.tobytes())

    def random_generation(self):
        pass

    def is_objective(self):
        return np.array_equal(self.__state, self.__goal)

    def rotate(self, plane, direction):
        '''
        :param plane:
        :param direction: 0 for 90 degrees, 1 for -90 degrees
        :return:
        '''
        self.__state = self.__state[Rubiks.__transforms[plane][direction]].reshape(6, 9)
        return self

    def get_corners(self):
        return self.__state[Rubiks.__corner_indices].tobytes()

    @staticmethod
    def generate_pattern_database(file: str):
        # Work directly from an ndarray state, not the parent Rubiks
        state = Rubiks().__state

        queue = list()
        queue.append((state, 0))

        # 8 corners for 8 positions, 7 of which can have 3 unique rotations, 88179840 possibilities
        all_corners = factorial(8) * 3**7
        hash_lookup = dict()
        hash_lookup[state[Rubiks.__corner_indices].tobytes()] = 0
        in_stack = dict()
        id_depth = 1
        while len(hash_lookup) < all_corners:
            next_state, depth = queue.pop()

            for plane in range(12):
                if plane < 6 or plane > 8:
                    for direction in range(2):
                        new_state = next_state[Rubiks.__transforms[plane][direction]].reshape(6, 9)
                        new_state_bytes = new_state[Rubiks.__corner_indices].tobytes()
                        if new_state_bytes not in in_stack or in_stack[new_state_bytes] > depth + 1:
                            in_stack[new_state_bytes] = depth + 1
                            if depth + 1 == id_depth:
                                if new_state_bytes not in hash_lookup:
                                    hash_lookup[new_state_bytes] = depth + 1
                                    if len(hash_lookup) % 10000 == 0:
                                        print(len(hash_lookup), depth + 1, len(queue))
                            else:
                                queue.append((new_state, depth + 1))

            if len(queue) == 0:
                id_depth += 1
                in_stack = dict()
                queue.append((state, 0))
                print(f"Incrementing id-depth to {id_depth}")

        while len(queue) > 0:
            next_state, depth, corners = queue.pop()
            if corners not in hash_lookup or hash_lookup[corners] > depth:
                if corners in hash_lookup:
                    print("Found more values in the stack with lower depth, BUG!!!")
                hash_lookup[corners] = depth

        with open(f'{file}.pkl', 'wb') as f:
            pickle.dump(hash_lookup, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pattern_database(file: str):
        with open(f'{file}.pkl', 'rb') as f:
            return pickle.load(f)


if __name__ == "__main__":
    Rubiks.generate_pattern_database('database')
    #pattern_db = Rubiks.load_pattern_database('database')
    pass
