from numba import jit
import copy
import enum
import numpy as np
import pickle
import time

# Uses facelet numbering defined at http://kociemba.org/cube.htm


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
    halfway = 2


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
            [['U7', 'U4', 'U1', 'U8', 'U2', 'U9', 'U6', 'U3'],
             ['B1', 'B2', 'B3', 'R4', 'R6', 'R7', 'R8', 'R9'],
             ['R1', 'R2', 'R3', 'F4', 'F6', 'F7', 'F8', 'F9'],
             ['D1', 'D2', 'D3', 'D4', 'D6', 'D7', 'D8', 'D9'],
             ['F1', 'F2', 'F3', 'L4', 'L6', 'L7', 'L8', 'L9'],
             ['L1', 'L2', 'L3', 'B4', 'B6', 'B7', 'B8', 'B9']],

            # -90 CCW
            [['U3', 'U6', 'U9', 'U2', 'U8', 'U1', 'U4', 'U7'],
             ['F1', 'F2', 'F3', 'R4', 'R6', 'R7', 'R8', 'R9'],
             ['L1', 'L2', 'L3', 'F4', 'F6', 'F7', 'F8', 'F9'],
             ['D1', 'D2', 'D3', 'D4', 'D6', 'D7', 'D8', 'D9'],
             ['B1', 'B2', 'B3', 'L4', 'L6', 'L7', 'L8', 'L9'],
             ['R1', 'R2', 'R3', 'B4', 'B6', 'B7', 'B8', 'B9']],

            # 180
            [['U9', 'U8', 'U7', 'U6', 'U4', 'U3', 'U2', 'U1'],
             ['L1', 'L2', 'L3', 'R4', 'R6', 'R7', 'R8', 'R9'],
             ['B1', 'B2', 'B3', 'F4', 'F6', 'F7', 'F8', 'F9'],
             ['D1', 'D2', 'D3', 'D4', 'D6', 'D7', 'D8', 'D9'],
             ['R1', 'R2', 'R3', 'L4', 'L6', 'L7', 'L8', 'L9'],
             ['F1', 'F2', 'F3', 'B4', 'B6', 'B7', 'B8', 'B9']]
        ],

        # Rotate R
        [
            # 90
            [['U1', 'U2', 'F3', 'U4', 'F6', 'U7', 'U8', 'F9'],
             ['R7', 'R4', 'R1', 'R8', 'R2', 'R9', 'R6', 'R3'],
             ['F1', 'F2', 'D3', 'F4', 'D6', 'F7', 'F8', 'D9'],
             ['D1', 'D2', 'B7', 'D4', 'B4', 'D7', 'D8', 'B1'],
             ['L1', 'L2', 'L3', 'L4', 'L6', 'L7', 'L8', 'L9'],
             ['U9', 'B2', 'B3', 'U6', 'B6', 'U3', 'B8', 'B9']],

            # -90
            [['U1', 'U2', 'B7', 'U4', 'B4', 'U7', 'U8', 'B1'],
             ['R3', 'R6', 'R9', 'R2', 'R8', 'R1', 'R4', 'R7'],
             ['F1', 'F2', 'U3', 'F4', 'U6', 'F7', 'F8', 'U9'],
             ['D1', 'D2', 'F3', 'D4', 'F6', 'D7', 'D8', 'F9'],
             ['L1', 'L2', 'L3', 'L4', 'L6', 'L7', 'L8', 'L9'],
             ['D9', 'B2', 'B3', 'D6', 'B6', 'D3', 'B8', 'B9']],

            # 180
            [['U1', 'U2', 'D3', 'U4', 'D6', 'U7', 'U8', 'D9'],
             ['R9', 'R8', 'R7', 'R6', 'R4', 'R3', 'R2', 'R1'],
             ['F1', 'F2', 'B7', 'F4', 'B4', 'F7', 'F8', 'B1'],
             ['D1', 'D2', 'U3', 'D4', 'U6', 'D7', 'D8', 'U9'],
             ['L1', 'L2', 'L3', 'L4', 'L6', 'L7', 'L8', 'L9'],
             ['F9', 'B2', 'B3', 'F6', 'B6', 'F3', 'B8', 'B9']]
        ],

        # Rotate F
        [
            # 90
            [['U1', 'U2', 'U3', 'U4', 'U6', 'L9', 'L6', 'L3'],
             ['U7', 'R2', 'R3', 'U8', 'R6', 'U9', 'R8', 'R9'],
             ['F7', 'F4', 'F1', 'F8', 'F2', 'F9', 'F6', 'F3'],
             ['R7', 'R4', 'R1', 'D4', 'D6', 'D7', 'D8', 'D9'],
             ['L1', 'L2', 'D1', 'L4', 'D2', 'L7', 'L8', 'D3'],
             ['B1', 'B2', 'B3', 'B4', 'B6', 'B7', 'B8', 'B9']],

            # -90
            [['U1', 'U2', 'U3', 'U4', 'U6', 'R1', 'R4', 'R7'],
             ['D3', 'R2', 'R3', 'D2', 'R6', 'D1', 'R8', 'R9'],
             ['F3', 'F6', 'F9', 'F2', 'F8', 'F1', 'F4', 'F7'],
             ['L3', 'L6', 'L9', 'D4', 'D6', 'D7', 'D8', 'D9'],
             ['L1', 'L2', 'U9', 'L4', 'U8', 'L7', 'L8', 'U7'],
             ['B1', 'B2', 'B3', 'B4', 'B6', 'B7', 'B8', 'B9']],

            # 180
            [['U1', 'U2', 'U3', 'U4', 'U6', 'D3', 'D2', 'D1'],
             ['L9', 'R2', 'R3', 'L6', 'R6', 'L3', 'R8', 'R9'],
             ['F9', 'F8', 'F7', 'F6', 'F4', 'F3', 'F2', 'F1'],
             ['U9', 'U8', 'U7', 'D4', 'D6', 'D7', 'D8', 'D9'],
             ['L1', 'L2', 'R7', 'L4', 'R4', 'L7', 'L8', 'R1'],
             ['B1', 'B2', 'B3', 'B4', 'B6', 'B7', 'B8', 'B9']]
        ],

        # Rotate D
        [
            # 90
            [['U1', 'U2', 'U3', 'U4', 'U6', 'U7', 'U8', 'U9'],
             ['R1', 'R2', 'R3', 'R4', 'R6', 'F7', 'F8', 'F9'],
             ['F1', 'F2', 'F3', 'F4', 'F6', 'L7', 'L8', 'L9'],
             ['D7', 'D4', 'D1', 'D8', 'D2', 'D9', 'D6', 'D3'],
             ['L1', 'L2', 'L3', 'L4', 'L6', 'B7', 'B8', 'B9'],
             ['B1', 'B2', 'B3', 'B4', 'B6', 'R7', 'R8', 'R9']],

            # -90
            [['U1', 'U2', 'U3', 'U4', 'U6', 'U7', 'U8', 'U9'],
             ['R1', 'R2', 'R3', 'R4', 'R6', 'B7', 'B8', 'B9'],
             ['F1', 'F2', 'F3', 'F4', 'F6', 'R7', 'R8', 'R9'],
             ['D3', 'D6', 'D9', 'D2', 'D8', 'D1', 'D4', 'D7'],
             ['L1', 'L2', 'L3', 'L4', 'L6', 'F7', 'F8', 'F9'],
             ['B1', 'B2', 'B3', 'B4', 'B6', 'L7', 'L8', 'L9']],

            # 180
            [['U1', 'U2', 'U3', 'U4', 'U6', 'U7', 'U8', 'U9'],
             ['R1', 'R2', 'R3', 'R4', 'R6', 'L7', 'L8', 'L9'],
             ['F1', 'F2', 'F3', 'F4', 'F6', 'B7', 'B8', 'B9'],
             ['D9', 'D8', 'D7', 'D6', 'D4', 'D3', 'D2', 'D1'],
             ['L1', 'L2', 'L3', 'L4', 'L6', 'R7', 'R8', 'R9'],
             ['B1', 'B2', 'B3', 'B4', 'B6', 'F7', 'F8', 'F9']]
        ],

        # Rotate L
        [
            # 90
            [['B9', 'U2', 'U3', 'B6', 'U6', 'B3', 'U8', 'U9'],
             ['R1', 'R2', 'R3', 'R4', 'R6', 'R7', 'R8', 'R9'],
             ['U1', 'F2', 'F3', 'U4', 'F6', 'U7', 'F8', 'F9'],
             ['F1', 'D2', 'D3', 'F4', 'D6', 'F7', 'D8', 'D9'],
             ['L7', 'L4', 'L1', 'L8', 'L2', 'L9', 'L6', 'L3'],
             ['B1', 'B2', 'D7', 'B4', 'D4', 'B7', 'B8', 'D1']],

            # -90
            [['F1', 'U2', 'U3', 'F4', 'U6', 'F7', 'U8', 'U9'],
             ['R1', 'R2', 'R3', 'R4', 'R6', 'R7', 'R8', 'R9'],
             ['D1', 'F2', 'F3', 'D4', 'F6', 'D7', 'F8', 'F9'],
             ['B9', 'D2', 'D3', 'B6', 'D6', 'B3', 'D8', 'D9'],
             ['L3', 'L6', 'L9', 'L2', 'L8', 'L1', 'L4', 'L7'],
             ['B1', 'B2', 'U7', 'B4', 'U4', 'B7', 'B8', 'U1']],

            # 180
            [['D1', 'U2', 'U3', 'D4', 'U6', 'D7', 'U8', 'U9'],
             ['R1', 'R2', 'R3', 'R4', 'R6', 'R7', 'R8', 'R9'],
             ['B9', 'F2', 'F3', 'B6', 'F6', 'B3', 'F8', 'F9'],
             ['U1', 'D2', 'D3', 'U4', 'D6', 'U7', 'D8', 'D9'],
             ['L9', 'L8', 'L7', 'L6', 'L4', 'L3', 'L2', 'L1'],
             ['B1', 'B2', 'F7', 'B4', 'F4', 'B7', 'B8', 'F1']]
        ],

        # Rotate B
        [
            # 90
            [['R3', 'R6', 'R9', 'U4', 'U6', 'U7', 'U8', 'U9'],
             ['R1', 'R2', 'D9', 'R4', 'D8', 'R7', 'R8', 'D7'],
             ['F1', 'F2', 'F3', 'F4', 'F6', 'F7', 'F8', 'F9'],
             ['D1', 'D2', 'D3', 'D4', 'D6', 'L1', 'L4', 'L7'],
             ['U3', 'L2', 'L3', 'U2', 'L6', 'U1', 'L8', 'L9'],
             ['B7', 'B4', 'B1', 'B8', 'B2', 'B9', 'B6', 'B3']],

            # -90
            [['L7', 'L4', 'L1', 'U4', 'U6', 'U7', 'U8', 'U9'],
             ['R1', 'R2', 'U1', 'R4', 'U2', 'R7', 'R8', 'U3'],
             ['F1', 'F2', 'F3', 'F4', 'F6', 'F7', 'F8', 'F9'],
             ['D1', 'D2', 'D3', 'D4', 'D6', 'R9', 'R6', 'R3'],
             ['D7', 'L2', 'L3', 'D8', 'L6', 'D9', 'L8', 'L9'],
             ['B3', 'B6', 'B9', 'B2', 'B8', 'B1', 'B4', 'B7']],

            # 180
            [['D9', 'D8', 'D7', 'U4', 'U6', 'U7', 'U8', 'U9'],
             ['R1', 'R2', 'L7', 'R4', 'L4', 'R7', 'R8', 'L1'],
             ['F1', 'F2', 'F3', 'F4', 'F6', 'F7', 'F8', 'F9'],
             ['D1', 'D2', 'D3', 'D4', 'D6', 'U3', 'U2', 'U1'],
             ['R9', 'L2', 'L3', 'R6', 'L6', 'R3', 'L8', 'L9'],
             ['B9', 'B8', 'B7', 'B6', 'B4', 'B3', 'B2', 'B1']]
        ]
     ]

    lookups = []
    for plane_index, plane in enumerate(move_rotations):
        directions = []
        for direction_index, direction in enumerate(plane):
            index_face = []
            index_item = []
            if len(np.unique(move_rotations[plane_index][direction_index])) != 8 * 6:
                print(move_rotations[plane_index][direction_index])
            assert (len(np.unique(move_rotations[plane_index][direction_index])) == 8*6)
            for face in direction:
                for item in face:
                    index_face.append(translate_face(item[0]))
                    position = int(item[1]) - 1
                    if position > 4:
                        position -= 1
                    index_item.append(position)
            directions.append((np.array(index_face), np.array(index_item)))
        lookups.append(directions)
    return lookups


class Rubiks:
    __goal = np.empty([6, 8], dtype=np.uint8)
    for face in range(6):
        for row in range(8):
            __goal[face][row] = np.uint8(face)
    __transforms = build_transform_lookups()

    # All of the faces of each corner, except for up/right/front
    # because it's determined by the the other 7 corners faces
    __corner_indices = np.array([[True, False,  True, False, False,  True, False,  False],
                                 [False, False,  True, False, False,  True, False,  True],
                                 [True, False,  False, False, False,  True, False,  True],
                                 [True, False,  True, False, False,  True, False,  True],
                                 [True, False,  True, False, False,  True, False,  True],
                                 [True, False,  True, False, False,  True, False,  True]]).reshape(6, 8)

    def __init__(self, initial=None):
        if initial is None:
            self.state = copy.copy(self.__goal)
        else:
            self.state = initial.__state.copy()

    def __repr__(self):
        return str(np.array([list(map(Color, x)) for x in self.state.tolist()], dtype=Color).reshape(6, 8))

    def __eq__(self, other):
        if other is Rubiks:
            return np.array_equal(self.state, other.__state)
        else:
            return False

    def __hash__(self):
        return hash(self.state.tobytes())

    def random_generation(self):
        pass

    def is_objective(self):
        return np.array_equal(self.state, self.__goal)

    def rotate(self, plane, direction):
        '''
        :param plane:
        :param direction: 0 for 90 degrees, 1 for -90 degrees
        :return:
        '''
        self.state = self.state[Rubiks.__transforms[plane][direction]].reshape(6, 8)
        return self

    def get_corners(self):
        return self.state[Rubiks.__corner_indices].tobytes()

    @staticmethod
    @jit(nopython=True)
    def generate_pattern_database(initial_state):
        queue = list()
        queue.append((initial_state, 0, -1, 0))

        # 8 corners for 8 positions, 7 of which can have 3 unique rotations, 88179840 possibilities
        all_corners = 88179840
        hash_lookup = dict()
        hash_lookup[initial_state[Rubiks.__corner_indices].tobytes()] = 0

        in_stack = dict()
        count = 1
        id_depth = 1
        while count < all_corners:
            if len(queue) == 0:
                id_depth += 1
                in_stack = dict()
                queue.append((initial_state, 0, -1, 0))
                print("Incrementing id-depth to ", id_depth)

            next_state, depth, last_rotation, last_face = queue.pop()

            new_state_depth = depth + 1
            for face in range(6):
                for rotation in range(3):
                    if last_face == face:
                        # Avoid rotating back to the previous state
                        if (last_rotation == 2 and rotation == 2) or (last_rotation == 0 and rotation == 1) or (last_rotation == 1 and rotation == 0):
                            continue
                    new_state = next_state[Rubiks.__transforms[face][rotation]].reshape(6, 8)
                    new_state_bytes = new_state[Rubiks.__corner_indices].tobytes()
                    if new_state_depth == id_depth and new_state_bytes not in hash_lookup:
                        hash_lookup[new_state_bytes] = new_state_depth
                        count += 1
                        if count % 10000 == 0:
                            print(count, new_state_depth, len(queue))
                    elif new_state_depth < id_depth and (new_state_bytes not in in_stack or in_stack[new_state_bytes] > new_state_depth):
                        in_stack[new_state_bytes] = new_state_depth
                        queue.append((new_state, new_state_depth, rotation, face))

        while len(queue) > 0:
            next_state, depth, _, _ = queue.pop()
            corners = next_state[Rubiks.__corner_indices].tobytes()
            if corners not in hash_lookup or hash_lookup[corners] > depth:
                if corners in hash_lookup:
                    print("Found more values in the stack with lower depth, BUG!!!")
                hash_lookup[corners] = depth

        return hash_lookup

    @staticmethod
    def load_pattern_database(file: str):
        with open(f'{file}.pkl', 'rb') as f:
            return pickle.load(f)


if __name__ == "__main__":
    start = time.perf_counter()
    db = Rubiks.generate_pattern_database(Rubiks().state)
    with open(f'database.pkl', 'wb') as f:
        pickle.dump(db, f, pickle.HIGHEST_PROTOCOL)

    print("Finished: ", time.perf_counter() - start)
    #pattern_db = Rubiks.load_pattern_database('database')
    pass
