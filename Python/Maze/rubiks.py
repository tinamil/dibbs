import re
import copy
import enum
import numpy as np


@enum.unique
class Face(enum.IntEnum):
    top = 0
    bottom = 1
    left = 2
    right = 3
    front = 4
    back = 5


class Rubiks:
    __goal = np.empty([6, 3, 3], dtype=np.uint8)
    for face in range(6):
        for row in range(3):
            for col in range(3):
                __goal[face][row][col] = np.uint8(face)

    def __init__(self):
        self.__state = copy.deepcopy(self.__goal)

    def random_generation(self):
        pass

    def is_objective(self):
        return np.array_equal(self.__state, self.__goal)

    def rotate(self, face, angle):
       #6 Faces + 3 slices clockwise and counter-clockwise
       # Create a new index array from ndarray that specifies the new positions (https://stackoverflow.com/questions/26194389/numpy-rearrange-array-based-upon-index-array)
       #(U1, U2, U3, U4, U5, U6, U7, U8, U9, R1, R2, R3, R4, R5, R6, R7, R8, R9, F1, F2, F3, F4, F5, F6, F7, F8, F9, D1, D2, D3, D4, D5, D6, D7, D8, D9, L1, L2, L3, L4, L5, L6, L7, L8, L9, B1, B2, B3, B4, B5, B6, B7, B8, B9)
       #(U3, U6, U9, U2, U5, U8, U1, U4, U7, F1, F2, F3, R4, R5, R6, R7, R8, R9, L1, L2, L3, F4, F5, F6, F7, F8, F9, D1, D2, D3, D4, D5, D6, D7, D8, D9, B1, B2, B3, L4, L5, L6, L7, L8, L9, R1, R2, R3, B4, B5, B6, B7, B8, B9)
       print(self.__state)
       print(np.rot90(self.__state, k=1))

    # Returns the start position as a tuple of (row, column)
    def getStart(self):
        return self.__start

    def getNeighbors(self):
        neighbors = []
        return neighbors