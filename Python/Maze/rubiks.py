import copy
import enum
import numpy as np


@enum.unique
class Face(enum.IntEnum):
    up = 0
    right = 1
    front = 2
    down = 3
    left = 4
    back = 5
    middle = 6
    equatorial = 7
    standing = 8


@enum.unique
class Color(enum.IntEnum):
    orange = Face.up
    red = Face.down
    white = Face.front
    blue = Face.right
    yellow = Face.back
    green = Face.left

    def __repr__(self):
        return self.name


class Rubiks:
    __goal = np.empty([6, 9], dtype=np.uint8)
    for face in range(6):
        for row in range(9):
            __goal[face][row] = np.uint8(face)
    __transforms = None

    def __init__(self):
        if self.__transforms is None:
            self.__transforms = self.build_transform_lookups()
        self.__state = copy.deepcopy(self.__goal)

    def __repr__(self):
        return str(np.array([list(map(Color, x)) for x in self.__state.tolist()], dtype=Color).reshape(6, 3, 3))

    def random_generation(self):
        pass

    def is_objective(self):
        return np.array_equal(self.__state, self.__goal)

    def rotate(self, face, angle):
        '''

        :param face:
        :param angle: 0 for 90 degrees, 1 for -90 degrees
        :return:
        '''
        self.__state = self.__state[self.__transforms[face][angle]].reshape(6, 9)
        return self

    @staticmethod
    def build_transform_lookups():
        move_rotations = [
            # Rotate U
            [
                # 90
                [['U7', 'U4', 'U1', 'U8', 'U5', 'U2', 'U9', 'U6', 'U3'],
                 ['B1', 'B2', 'B3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9'],
                 ['R1', 'R2', 'R3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9'],
                 ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9'],
                 ['F1', 'F2', 'F3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9'],
                 ['L1', 'L2', 'L3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9']],

                # -90
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
                 ['D1', 'D2', 'B1', 'D4', 'D5', 'B4', 'D7', 'D8', 'B7'],
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
            ],

            # Rotate M
            [
                # 90
                [['U1', 'B2', 'U3', 'U4', 'B5', 'U6', 'U7', 'B8', 'U9'],
                 ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9'],
                 ['F1', 'U2', 'F3', 'F4', 'U5', 'F6', 'F7', 'U8', 'F9'],
                 ['D1', 'F2', 'D3', 'D4', 'F5', 'D6', 'D7', 'F8', 'D9'],
                 ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9'],
                 ['B1', 'D2', 'B3', 'B4', 'D5', 'B6', 'B7', 'D8', 'B9']],

                # -90
                [['U1', 'F2', 'U3', 'U4', 'F5', 'U6', 'U7', 'F8', 'U9'],
                 ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9'],
                 ['F1', 'D2', 'F3', 'F4', 'D5', 'F6', 'F7', 'D8', 'F9'],
                 ['D1', 'B2', 'D3', 'D4', 'B5', 'D6', 'D7', 'B8', 'D9'],
                 ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9'],
                 ['B1', 'U2', 'B3', 'B4', 'U5', 'B6', 'B7', 'U8', 'B9']],
            ],

            # Rotate E
            [
                # 90
                [['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9'],
                 ['R1', 'R2', 'R3', 'B4', 'B5', 'B6', 'R7', 'R8', 'R9'],
                 ['F1', 'F2', 'F3', 'R4', 'R5', 'R6', 'F7', 'F8', 'F9'],
                 ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9'],
                 ['L1', 'L2', 'L3', 'F4', 'F5', 'F6', 'L7', 'L8', 'L9'],
                 ['B1', 'B2', 'B3', 'L4', 'L5', 'L6', 'B7', 'B8', 'B9']],

                # -90
                [['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9'],
                 ['R1', 'R2', 'R3', 'F4', 'F5', 'F6', 'R7', 'R8', 'R9'],
                 ['F1', 'F2', 'F3', 'L4', 'L5', 'L6', 'F7', 'F8', 'F9'],
                 ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9'],
                 ['L1', 'L2', 'L3', 'B4', 'B5', 'B6', 'L7', 'L8', 'L9'],
                 ['B1', 'B2', 'B3', 'R4', 'R5', 'R6', 'B7', 'B8', 'B9']],
            ],

            # Rotate S
            [
                # 90
                [['U1', 'U2', 'U3', 'L2', 'L5', 'L8', 'U7', 'U8', 'U9'],
                 ['R1', 'U4', 'R3', 'R4', 'U5', 'R6', 'R7', 'U6', 'R9'],
                 ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9'],
                 ['D1', 'D2', 'D3', 'R2', 'R5', 'R8', 'D7', 'D8', 'D9'],
                 ['L1', 'D4', 'L3', 'L4', 'D5', 'L6', 'L7', 'D6', 'L9'],
                 ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9']],

                # -90
                [['U1', 'U2', 'U3', 'R2', 'R5', 'R8', 'U7', 'U8', 'U9'],
                 ['R1', 'D4', 'R3', 'R4', 'D5', 'R6', 'R7', 'D6', 'R9'],
                 ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9'],
                 ['D1', 'D2', 'D3', 'L2', 'L5', 'L8', 'D7', 'D8', 'D9'],
                 ['L1', 'U6', 'L3', 'L4', 'U5', 'L6', 'L7', 'U4', 'L9'],
                 ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9']],
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
                        index_face.append(Rubiks.translate_face(item[0]))
                        index_item.append(int(item[1]) - 1)
                directions.append((np.array(index_face), np.array(index_item)))
            lookups.append(directions)
        return lookups

    @staticmethod
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


if __name__ == "__main__":
    cube = Rubiks()
    print(cube.rotate(Face.middle, 0).rotate(Face.equatorial, 0).rotate(Face.standing, 0))
