import argparse
from search import search
import pygame
from pygame.locals import *
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


class Application:
    def __init__(self, scale=20, fps=30):
        self.running = True
        self.displaySurface = None
        self.scale = scale
        self.fps = fps
        self.windowTitle = "Maze search: "

    # Initializes the pygame context and certain properties of the maze
    def initialize(self, filename):
        self.windowTitle += filename

        self.maze = Maze(filename)
        self.gridDim = self.maze.getDimensions()

        self.windowHeight = self.gridDim[0] * self.scale
        self.windowWidth = self.gridDim[1] * self.scale

        self.blockSizeX = int(self.windowWidth / self.gridDim[1])
        self.blockSizeY = int(self.windowHeight / self.gridDim[0])

    # Once the application is initiated, execute is in charge of drawing the game and dealing with the game loop
    def execute(self, filename, searchMethod, save):
        self.initialize(filename)

        if self.maze is None:
            print("No maze created")
            raise SystemExit

        start_time = time.perf_counter()
        path, statesExplored = search(self.maze, searchMethod)
        time_taken = time.perf_counter() - start_time

        pygame.init()
        self.displaySurface = pygame.display.set_mode((self.windowWidth, self.windowHeight), pygame.HWSURFACE)
        self.displaySurface.fill((255, 255, 255))
        pygame.display.flip()
        pygame.display.set_caption(self.windowTitle)

        print("Results")
        print("Path Length:", len(path))
        print("States Explored:", statesExplored)
        print("Time taken:", time_taken)
        print(path)
        self.drawPath(path)

        self.drawMaze()
        self.drawStart()
        self.drawObjective()

        pygame.display.flip()
        if save is not None:
            pygame.image.save(self.displaySurface, save)
            self.running = False

        clock = pygame.time.Clock()

        while self.running:
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            clock.tick(self.fps)

            if keys[K_ESCAPE]:
                raise SystemExit


    # Trivial implementation of a color scheme for the path taken
    def getColor(self, pathLength, index):
        step = 255 / pathLength
        green = index * step
        red = 255 - green
        return (red, green, 0)

    # Draws the path (given as a list of (row, col) tuples) to the display context
    def drawPath(self, path):
        for p in range(len(path)):
            color = self.getColor(len(path), p)
            self.drawSquare(path[p][0], path[p][1], color)

    # Simple wrapper for drawing a wall as a rectangle
    def drawWall(self, row, col):
        pygame.draw.rect(self.displaySurface, (0, 0, 0),
                         (col * self.blockSizeX, row * self.blockSizeY, self.blockSizeX, self.blockSizeY), 0)

    # Simple wrapper for drawing a circle
    def drawCircle(self, row, col, color, radius=None):
        if radius is None:
            radius = min(self.blockSizeX, self.blockSizeY) / 4
        pygame.draw.circle(self.displaySurface, color, (
        int(col * self.blockSizeX + self.blockSizeX / 2), int(row * self.blockSizeY + self.blockSizeY / 2)),
                           int(radius))

    def drawSquare(self, row, col, color):
        pygame.draw.rect(self.displaySurface, color,
                         (col * self.blockSizeX, row * self.blockSizeY, self.blockSizeX, self.blockSizeY), 0)

    # Draws the objectives to the display context
    def drawObjective(self):
        for obj in self.maze.getObjectives():
            self.drawCircle(obj[0], obj[1], (0, 0, 0))

    # Draws start location of path
    def drawStart(self):
        row, col = self.maze.getStart()
        pygame.draw.rect(self.displaySurface, (0, 0, 255), (
        col * self.blockSizeX + self.blockSizeX / 4, row * self.blockSizeY + self.blockSizeY / 4, self.blockSizeX * 0.5,
        self.blockSizeY * 0.5), 0)

    # Draws the full maze to the display context
    def drawMaze(self):
        for row in range(self.gridDim[0]):
            for col in range(self.gridDim[1]):
                if self.maze.isWall(row, col):
                    self.drawWall(row, col)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Maze Search')

    parser.add_argument('filename',
                        help='path to maze file [REQUIRED]')
    parser.add_argument('--method', dest="search", type=str, default="bfs",
                        choices=["bfs", "dfs", "greedy", "astar", "dibbs"],
                        help='search method - default bfs')
    parser.add_argument('--scale', dest="scale", type=int, default=20,
                        help='scale - default: 20')
    parser.add_argument('--fps', dest="fps", type=int, default=30,
                        help='fps for the display - default 30')
    parser.add_argument('--save', dest="save", type=str, default=None,
                        help='save output to image file - default not saved')

    args = parser.parse_args()
    app = Application(args.scale, args.fps)
    app.execute(args.filename, args.search, args.save)

