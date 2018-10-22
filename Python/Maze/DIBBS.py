from __future__ import annotations
from maze import Maze
import math
from typing import Optional, Tuple, List, Callable, Set, Any, FrozenSet, Iterable, Dict
from collections import defaultdict


class Node:
    def __init__(self, maze: Maze, parent: Optional[Node], position: Tuple[int, int], cost=0,
                 heuristic_func: Callable[[Node], int]=None)->None:
        # position is a tuple of (row, column)
        # parent is another Node state which is the path back to start
        self.maze = maze
        self.parent = parent
        self.position = position
        self.cost = cost
        self.heuristic = heuristic_func(self)
        self.combined = self.cost + self.heuristic
        self.f_bar = self.combined + self.cost - heuristic_func(self)
        self.reverse_parent = None

    def get_path(self) -> List[Tuple[int, int]]:
        """Iterates backwards through the parent nodes to produce a path from start to this position"""
        path = [x.position for x in self]
        path.reverse()
        return path

    def __eq__(self, other: Node) -> bool:
        """Equality is true only if positions and remaining objectives are equal"""
        return self.position == other.position

    def __hash__(self):
        """Hash of position and remaining objectives"""
        return hash(self.position)

    def __iter__(self):
        """Define an iterator (for _ in Node) will iterate back through to the parent"""
        return NodeIterator(self)


class NodeIterator:
    """Iterates through the provided node until the starting node (parent == None) is reached.  Used as a helper
    function to generate full path backwards."""
    def __init__(self, node: Node) -> None:
        self.node = node

    def __next__(self) -> Node:
        if self.node is None:
            raise StopIteration()
        else:
            node = self.node
            self.node = self.node.parent
            return node

    def __iter__(self) -> NodeIterator:
        return self


def manhattan_distance(start: Tuple[int, int], end: Tuple[int, int]) -> int:
    """Calculates manhattan distance (|x2 - x1| + |y2 - y1|) of two different positions"""
    x1, y1 = start
    x2, y2 = end
    return abs(x2 - x1) + abs(y2 - y1)


class DIBBS:
    def __init__(self, maze: Maze) -> None:
        self.maze = maze
        assert (len(maze.getObjectives()) == 1)

        self.upper_bound = math.inf
        self.best_node = None
        objective = maze.getObjectives().pop()
        start = maze.getStart()

        self.forward_heuristic = lambda x: manhattan_distance(x.position, objective)
        self.backward_heuristic = lambda x: manhattan_distance(x.position, start)

        start_node = Node(maze, None, start, 0, heuristic_func=self.forward_heuristic)
        end_node = Node(maze, None, objective, 0, heuristic_func=self.backward_heuristic)
        self.forward_frontier: List[Node] = list()
        self.forward_frontier.append(start_node)
        self.backward_frontier: List[Node] = list()
        self.backward_frontier.append(end_node)
        self.forward_closed: Dict[Node, Node] = dict()
        self.backward_closed: Dict[Node, Node] = dict()
        self.forward_costs: defaultdict[Node, int] = defaultdict(lambda: math.inf)
        self.backward_costs: defaultdict[Node, int] = defaultdict(lambda: math.inf)

    def search(self, _) -> Tuple[List[Tuple[int, int]], int]:
        while self.upper_bound > (self.forward_frontier[len(self.forward_frontier) - 1].f_bar +
                                  self.backward_frontier[len(self.backward_frontier) - 1].f_bar) / 2.0:
            if self.choose_direction(self.forward_frontier, self.backward_frontier):
                self.expand(self.forward_frontier, self.forward_closed, self.forward_heuristic,
                            self.backward_frontier, self.backward_closed)
                self.forward_frontier.sort(key=lambda x: x.f_bar, reverse=True)
            else:
                self.expand(self.backward_frontier, self.backward_closed, self.backward_heuristic,
                            self.forward_frontier, self.forward_closed)
                self.backward_frontier.sort(key=lambda x: x.f_bar, reverse=True)

        path = self.best_node.get_path()[:-1]
        reverse_path = self.best_node.reverse_parent.get_path()

        if path[0] != self.maze.getStart():
            path, reverse_path = reverse_path, path

        path.extend(reversed(reverse_path))

        return path, len(self.forward_closed) + len(self.backward_closed)

    def expand(self, frontier: List[Node], closed: Dict[Node, Node], heuristic: Callable, other_frontier: List[Node],
               other_closed: Dict[Node, Node]):
        next_value = frontier.pop()
        """Iterate through node neighbors and add them to the frontier"""
        if next_value not in closed:
            for neighbor in self.maze.getNeighbors(*next_value.position):
                node = Node(self.maze, next_value, neighbor, next_value.cost + 1, heuristic_func=heuristic)
                if node in closed and node.f_bar < closed[node].f_bar:
                    closed.pop(node)
                if node not in closed:
                    frontier.append(node)
                    reverse_found = None
                    if node in other_closed:
                        reverse_found = other_closed[node]
                    elif node in other_frontier:
                        reverse_found = other_frontier[other_frontier.index(node)]
                    if reverse_found is not None and node.cost + reverse_found.cost < self.upper_bound:
                        self.upper_bound = node.cost + reverse_found.cost
                        self.best_node = node
                        self.best_node.reverse_parent = reverse_found
            closed[next_value] = next_value

    @staticmethod
    def choose_direction(forward_frontier, backward_frontier):
        #return len(forward_frontier) < len(backward_frontier)
        return forward_frontier[len(forward_frontier) - 1].f_bar < backward_frontier[len(backward_frontier) - 1].f_bar
