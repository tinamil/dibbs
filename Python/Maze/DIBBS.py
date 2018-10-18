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
        self.forward_closed: Dict[Node, int] = dict()
        self.backward_closed: Dict[Node, int] = dict()
        self.forward_costs: defaultdict[Node, int] = defaultdict(lambda: math.inf)
        self.backward_costs: defaultdict[Node, int] = defaultdict(lambda: math.inf)

    def search(self, _) -> Tuple[List[Tuple[int, int]], int]:
        while self.upper_bound > (self.forward_frontier[len(self.forward_frontier) - 1].f_bar +
                                  self.backward_frontier[len(self.backward_frontier) - 1].f_bar) / 2.0:
            if self.forward_frontier[len(self.forward_frontier) - 1].f_bar < \
                    self.backward_frontier[len(self.backward_frontier) - 1].f_bar:
                self.expand(self.forward_frontier, self.forward_closed, self.forward_heuristic, self.forward_costs,
                            self.backward_costs)
                self.forward_frontier.sort(key=lambda x: x.f_bar, reverse=True)
            else:
                self.expand(self.backward_frontier, self.backward_closed, self.backward_heuristic, self.backward_costs,
                            self.forward_costs)
                self.backward_frontier.sort(key=lambda x: x.f_bar, reverse=True)

        best_forward_fbar = self.forward_frontier[len(self.forward_frontier) - 1].f_bar
        best_backward_fbar = self.backward_frontier[len(self.backward_frontier) - 1].f_bar
        self.forward_frontier = list(filter(lambda x: x.f_bar == best_forward_fbar, self.forward_frontier))
        self.backward_frontier = list(filter(lambda x: x.f_bar == best_backward_fbar, self.backward_frontier))
        for f_node in self.forward_frontier:
            for b_node in self.backward_frontier:
                if f_node.position == b_node.position:
                    path = f_node.get_path()[:-1]
                    reverse_path = b_node.get_path()
                    path.extend(reversed(reverse_path))
                    return path, len(self.forward_closed) + len(self.backward_closed)

    def expand(self, frontier: List[Node], closed: Dict[Node, int], heuristic: Callable,
               my_costs: defaultdict[Node, int], other_costs: defaultdict[Node, int]):
        next_value = frontier.pop()
        """Iterate through node neighbors and add them to the frontier"""
        if next_value not in closed:
            for neighbor in self.maze.getNeighbors(*next_value.position):
                node = Node(self.maze, next_value, neighbor, next_value.cost + 1, heuristic_func=heuristic)
                if node in closed and node.f_bar < closed[node]:
                    closed.pop(node)
                if node not in closed:
                    frontier.append(node)
                    my_costs[node] = node.cost
                    self.upper_bound = min(self.upper_bound, node.cost + other_costs[node])
            closed[next_value] = next_value.f_bar
