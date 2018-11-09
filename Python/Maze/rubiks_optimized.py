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


# 6 center cubies are fixed position (rotation doesn't change) and define the solution color for that face
# 8 corner cubies are defined by an integer defining their rotation: 0-2186 (3^7-1, 8th cube is defined by the other 7)
#                     and an integer defining their positions: 0-40319 (8! - 1)
# 12 edge cubies have rotation integer 0-2047 (2^11-1) and position integer 0-12!-1
# state is (corner position, edge position, corner rotation, edge rotation): (uint16, uint16, uint16, uint32)





@enum.unique
class Transformation(enum.IntEnum):
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
    orange = Transformation.up
    red = Transformation.down
    white = Transformation.front
    blue = Transformation.right
    yellow = Transformation.back
    green = Transformation.left
    blank = -1

    def __repr__(self):
        return self.name


class Rubiks:

    def __init__(self):
        pass

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
