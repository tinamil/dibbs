# main.py

import argparse
import time

from maze import Maze
from search import search


def execute(filename, search_method):
        maze = Maze(filename)

        if maze is None:
            print("No maze created")
            raise SystemExit

        start_time = time.perf_counter()
        path, states_explored = search(maze, search_method)
        time_taken = time.perf_counter() - start_time
        print("Results")
        print("Path Length:", len(path))
        print("States Explored:", states_explored)
        print("Time taken:", time_taken)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Maze Search')
    
    parser.add_argument('filename',
                        help='path to maze file [REQUIRED]')
    parser.add_argument('--method', dest="search", type=str, default="bfs",
                        choices=["bfs", "dfs", "greedy", "astar", "aco"],
                        help='search method - default bfs')

    args = parser.parse_args()
    args.execute(args.filename, args.search)
