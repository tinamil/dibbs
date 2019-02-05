#include <iostream>
#include <vector>
#include "rubiks.h"

using namespace std;

int main()
{
    cout << "Hello world!" << endl;
    cout << Rubiks::fast_factorial(10);
    return 0;
}

void a_star(uint8_t state[]){
    std::vector<uint8_t*> state_queue;
    std::vector<uint8_t> depth_queue;

//
//    queue.append((starting_state, 0, np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8)))
//
//    if is_solved(state):
//        return np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8), 0
//
//    min_moves = heuristic_func(state)
//    print("Minimum number of moves to solve: ", min_moves)
//    id_depth = min_moves
//    count = 0
//    while True:
//
//        if len(queue) == 0:
//            id_depth += 1
//            queue.append((starting_state, 0, np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8)))
//            print("Incrementing id-depth to", id_depth)
//
//        next_state, depth, prev_faces, prev_rots = queue.pop()
//        for face in range(6):
//
//            if len(prev_faces) > 0 and skip_rotations(prev_faces[-1], face):
//                continue
//
//            new_faces = np.empty(len(prev_faces) + 1, dtype=np.uint8)
//            for idx, val in enumerate(prev_faces):
//                new_faces[idx] = val
//            new_faces[-1] = face
//
//            for rotation in range(3):
//                new_state_base = rotate(next_state, face, rotation)
//                new_state_depth = depth + 1
//                new_state_heuristic = heuristic_func(new_state_base)
//                new_state_cost = new_state_depth + new_state_heuristic
//
//                count += 1
//
//                if new_state_cost > id_depth:
//                    continue
//
//                new_rots = np.empty(len(prev_rots) + 1, dtype=np.uint8)
//                for idx, val in enumerate(prev_rots):
//                    new_rots[idx] = val
//                new_rots[len(prev_rots)] = rotation
//
//                if is_solved(new_state_base):
//                    flip(new_faces)
//                    flip(new_rots)
//                    return new_faces, new_rots, count
//
//                queue.append((new_state_base, new_state_depth, new_faces, new_rots))
}
