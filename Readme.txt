Iterative Deepening DIBBS Algorithm Data:

Pancake instances are located in:
pancake/problems_n10.txt
pancake/problems_n14.txt
pancake/problems_n16.txt
pancake/problems_n20.txt
pancake/problems_n30.txt
pancake/problems_n40.txt

Pancake instance format is N integers in a permutation from 1 to N defining the starting pancake stack state.  The goal state is a sorted order: 1, 2, 3, ..., N.

Rubiks Cube instances are located in 
rubiks/korf1997.txt

Lines 1 and 2 are easier problem instances used for calibration and testing correctness.  Lines 3-12 are the 10 problem instances benchmarked in the paper, defined as 100 randomly selected moves from the initial solved state.  Each move is a letter for Right, Left, Up, Down, Front, or Back face.  No symbol after the letter indicates a clockwise 90 degree rotation of the given face.  An apostrophe indicates a counter-clockwise rotation.  The number 2 indicates a 180 degree rotation.

Sliding tile instances are located in
sliding_tile/main.cpp

Format is a permutation of the numbers 0 to 15.  0 indicates the empty space on the board.  The position in the list indicates the position on the board, e.g. the first number is the top left corner and last number is the bottom right corner, and the number indicates which piece is located in that position.  The goal state is 0, 1, 2, ..., 15.
