# main.py

import argparse
import time
import sys

from maze import Maze
from search import search
import pygame
from pygame.locals import *
import rubiks


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
    cube = rubiks.Rubiks()
    cube.rotate(None, None)
    # parser = argparse.ArgumentParser(description='Maze Search')
    #
    # parser.add_argument('filename',
    #                     help='path to maze file [REQUIRED]')
    # parser.add_argument('--method', dest="search", type=str, default="bfs",
    #                     choices=["bfs", "dfs", "greedy", "astar", "dibbs"],
    #                     help='search method - default bfs')
    # parser.add_argument('--scale', dest="scale", type=int, default=20,
    #                     help='scale - default: 20')
    # parser.add_argument('--fps', dest="fps", type=int, default=30,
    #                     help='fps for the display - default 30')
    # parser.add_argument('--save', dest="save", type=str, default=None,
    #                     help='save output to image file - default not saved')
    #
    # args = parser.parse_args()
    # app = Application(args.scale, args.fps)
    # app.execute(args.filename, args.search, args.save)

