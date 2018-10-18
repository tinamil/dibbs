import re
import copy


class Maze:
    def __init__(self, filename):
        self.__filename = filename
        self.__wallChar = '%'
        self.__startChar = 'P'
        self.__objectiveChar = '.'
        self.__start = None
        self.__objective = []

        with open(filename) as f:
            lines = f.readlines()

        lines = list(filter(lambda x: not re.match(r'^\s*$', x), lines))
        lines = [list(line.strip('\n')) for line in lines]

        self.rows = len(lines)
        self.cols = len(lines[0])
        self.mazeRaw = lines

        if (len(self.mazeRaw) != self.rows) or (len(self.mazeRaw[0]) != self.cols):
            print("Maze dimensions incorrect")
            raise SystemExit

        for row in range(len(self.mazeRaw)):
            for col in range(len(self.mazeRaw[0])):
                if self.mazeRaw[row][col] == self.__startChar:
                    self.__start = (row, col)
                elif self.mazeRaw[row][col] == self.__objectiveChar:
                    self.__objective.append((row, col))

    # Returns True if the given position is the location of a wall
    def isWall(self, row, col):
        return self.mazeRaw[row][col] == self.__wallChar

    # Rturns True if the given position is the location of an objective
    def isObjective(self, row, col):
        return (row, col) in self.__objective

    # Returns the start position as a tuple of (row, column)
    def getStart(self):
        return self.__start

    def setStart(self, start):
        self.__start = start

    # Returns the dimensions of the maze as a (row, column) tuple
    def getDimensions(self):
        return self.rows, self.cols

    # Returns the list of objective positions of the maze
    def getObjectives(self):
        return copy.deepcopy(self.__objective)

    def setObjectives(self, objectives):
        self.__objective = objectives

    # Check if the agent can move into a specific row and column
    def isValidMove(self, row, col):
        return row >= 0 and row < self.rows and col >= 0 and col < self.cols and not self.isWall(row, col)
        
    # Returns list of neighboring squares that can be moved to from the given row,col
    def getNeighbors(self, row, col):
        possibleNeighbors = [
            (row + 1, col),
            (row - 1, col),
            (row, col + 1),
            (row, col - 1)
        ]
        neighbors = []
        for r, c in possibleNeighbors:
            if self.isValidMove(r,c):
                neighbors.append((r,c))
        return neighbors