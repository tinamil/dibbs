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
    __goal = np.empty([6, 9], dtype=np.uint8)
    for face in range(6):
        for row in range(9):
            __goal[face][row] = np.uint8(face)

    def __init__(self):
        self.__state = copy.deepcopy(self.__goal)

    def random_generation(self):
        pass

    def is_objective(self):
        return np.array_equal(self.__state, self.__goal)

    def rotate(self, face, angle):
        #6 Faces + 4 slices clockwise and counter-clockwise
        # Create a new index array from ndarray that specifies the new positions (https://stackoverflow.com/questions/26194389/numpy-rearrange-array-based-upon-index-array)
        #(U1, U2, U3, U4, U5, U6, U7, U8, U9, R1, R2, R3, R4, R5, R6, R7, R8, R9, F1, F2, F3, F4, F5, F6, F7, F8, F9, D1, D2, D3, D4, D5, D6, D7, D8, D9, L1, L2, L3, L4, L5, L6, L7, L8, L9, B1, B2, B3, B4, B5, B6, B7, B8, B9)
        #(U3, U6, U9, U2, U5, U8, U1, U4, U7, F1, F2, F3, R4, R5, R6, R7, R8, R9, L1, L2, L3, F4, F5, F6, F7, F8, F9, D1, D2, D3, D4, D5, D6, D7, D8, D9, B1, B2, B3, L4, L5, L6, L7, L8, L9, R1, R2, R3, B4, B5, B6, B7, B8, B9)
        #print(self.__state)
        #print(self.__state)
        print(self.__state)
        print(self.__state[self.translate()].reshape(6, 9))
        #print(self.__state[np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])].reshape(1, 3, 3))
        #print(np.rot90(self.__state, k=1))

    def translate(self):
        reference = [['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9'],
                          ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9'],
                          ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9'],
                          ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9'],
                          ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9'],
                          ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9']]
        flip_up = [['U7', 'U4', 'U1', 'U8', 'U5', 'U2', 'U9', 'U6', 'U3'],
                   ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9'],
                   ['F1', 'F2', 'F3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9'],
                   ['B1', 'B2', 'B3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9'],
                   ['R1', 'R2', 'R3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9'],
                   ['L1', 'L2', 'L3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9']]

        index_face = []
        index_item = []
        for face in flip_up:
            for item in face:
                index_face.append(self.translate_face(item[0]))
                index_item.append(int(item[1]) - 1)
        return np.array(index_face), np.array(index_item)
        #np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]), np.array(
        #    [0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

    @staticmethod
    def translate_face(face):
        if face is 'U':
            return Face.top
        if face is 'D':
            return Face.bottom
        if face is 'F':
            return Face.front
        if face is 'B':
            return Face.back
        if face is 'R':
            return Face.right
        if face is 'L':
            return Face.left

    # Returns the start position as a tuple of (row, column)
    def getStart(self):
        return self.__start

    def getNeighbors(self):
        neighbors = []
        return neighbors