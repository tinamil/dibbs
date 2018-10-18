from __future__ import annotations
from typing import Optional, Tuple, List, Callable, Set, Any, FrozenSet, Iterable, Dict
from maze import Maze
import math
from queue import PriorityQueue
import DIBBS


"""Profiling imports from profilehooks and memory_profiler packages, commented out as they are external packages"""
#from profilehooks import profile
#from memory_profiler import profile as mem_profile


class PrioritizedItem:
    """PrioritizedItem is used for sorting a priority queue on a given priority integer value without
    consideration for the associated item being stored.

        Args:
            priority (int): The level of priority of the item, lowest value is considered first priority for retrieval
            item (Any): The item being stored in the queue

    """

    def __init__(self, priority: int, item: Any) -> None:
        self.priority = priority
        self.item = item

    def __lt__(self, other: PrioritizedItem) -> bool:
        return self.priority < other.priority

    def __gt__(self, other: PrioritizedItem) -> bool:
        return self.priority > other.priority

    def __le__(self, other: PrioritizedItem) -> bool:
        return not self.__gt__(other)

    def __ge__(self, other: PrioritizedItem) -> bool:
        return not self.__lt__(other)

    def __eq__(self, other: PrioritizedItem) -> bool:
        return self.item == other.item and self.priority == other.priority

    def __hash__(self) -> int:
        return hash((self.item, self.priority))


class Node:
    """Node stores the current state on a search graph, uniquely identified by the position and objectives_remaining

        Args:
            maze (Maze): The maze associated with this node, used only in get_closest_node method
            parent (Node/None): The parent node state that preceded this one.  If None this was the originating state.
            objectives_remaining (Set[positions]): The set of objectives not yet retrieved
            position (Tuple[int, int]): The position on the maze of this node
            cost (int): The cost to reach this node.  Defaults to 0.
            heuristic_func (Callable): A lambda function that evaluates the current position and objectives_remaining and
            returns an int value estimate to find all remaining objectives.  If undefined, then returns 0.

        Attributes:
            combined (int): The cost + heuristic estimate value for this node
    """

    """prim_data is a static dict[set(positions)] that stores cost.  Used by the prim function to cache minimum spanning
    tree costs associated with a given set of objectives"""
    prim_data: Dict[FrozenSet[Tuple[int, int]], int] = dict()

    def __init__(self, maze: Maze, parent: Optional[Node], objectives_remaining: FrozenSet[Tuple[int, int]],
                 position: Tuple[int, int], cost=0,
                 heuristic_func: Callable[[Node, Tuple[int, int], FrozenSet[Tuple[int, int]]], int]=None)->None:
        # position is a tuple of (row, column)
        # parent is another Node state which is the path back to start
        self.maze = maze
        self.parent = parent
        self.position = position
        self.cost = cost
        self.obj = objectives_remaining
        if heuristic_func:
            self.heuristic = heuristic_func(self, self.position, self.obj)
        else:
            self.heuristic = 0
        self.combined = self.cost + self.heuristic
        self.f_bar = self.combined + self.cost - heuristic_func(self, self.position, maze.getStart())

    def get_path(self) -> List[Tuple[int, int]]:
        """Iterates backwards through the parent nodes to produce a path from start to this position"""
        path = [x.position for x in self]
        path.reverse()
        return path

    def __eq__(self, other: Node) -> bool:
        """Equality is true only if positions and remaining objectives are equal"""
        return self.position == other.position and self.obj == other.obj

    def __hash__(self):
        """Hash of position and remaining objectives"""
        return hash((self.position, frozenset(self.obj)))

    def __iter__(self):
        """Define an iterator (for _ in Node) will iterate back through to the parent"""
        return NodeIterator(self)

    def distance_from(self, target: Tuple[int, int]) -> float:
        """Find the manhattan distance from this node to the target node"""
        return manhattan_distance(self.position, target)

    def is_done(self) -> bool:
        """Have we reached all objectives?"""
        return len(self.obj) == 0

    def get_closest_node(self, start: Tuple[int, int], nodes: Iterable[Tuple[int, int]]) -> Tuple[int, Tuple[int, int]]:
        """Find the closest node in the set of 'nodes' to the start position, returns the length of that path and the
        node reached.  Utilizes A* to a single target with Manhattan Distance heuristic."""
        path, searched_states = search_alg(self.maze, start=start, objectives=set(nodes),
                                           sort_key=lambda z: z.combined, find_any=True,
                                           heuristic=lambda w, pos, o: min((manhattan_distance(pos, x) for x in o),
                                                                           default=0),
                                           wait_for_frontier=True)

        assert len(path) > 0
        return len(path) - 1, path[len(path) - 1]

    def find_closest_tree_node(self, tree: Iterable[Tuple[int, int]],
                               remaining: Iterable[Tuple[int, int]]) -> Tuple[int, Tuple[int, int]]:
        """Find the closest position from the set of remaining to any of the positions in the current tree,
        returns the length of the path and the position"""
        best_cost = math.inf
        best_node = None
        for x in tree:
            cost, node = self.get_closest_node(x, remaining)
            if cost < best_cost:
                best_cost = cost
                best_node = node
        return best_cost, best_node

    def prim(self, start: Tuple[int, int], nodes: FrozenSet[Tuple[int, int]]) -> int:
        """Prim's algorithm applied to build a minimum spanning tree of the provided node positions, gets the length
        of that MST and then adds to that the distance from the start to the closest node."""
        if not nodes:
            return 0

        if nodes not in self.prim_data:
            est_cost = 0
            my_nodes = set(nodes.copy())
            finished_set = {my_nodes.pop()}
            while my_nodes:
                next_cost, next_node = self.find_closest_tree_node(finished_set, my_nodes)
                my_nodes.remove(next_node)
                finished_set.update({next_node})
                est_cost += next_cost
            self.prim_data[frozenset(finished_set)] = est_cost

        est_cost = self.prim_data[nodes]
        est_cost += self.get_closest_node(start, nodes)[0]
        return est_cost


def manhattan_distance(start: Tuple[int, int], end: Tuple[int, int]) -> int:
    """Calculates manhattan distance (|x2 - x1| + |y2 - y1|) of two different positions"""
    x1, y1 = start
    x2, y2 = end
    return abs(x2 - x1) + abs(y2 - y1)


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


def search(maze: Maze, searchMethod: str) -> Tuple[List[Tuple[int, int]], int]:
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
        "dibbs": DIBBS.DIBBS(maze).search
    }.get(searchMethod)(maze)
   

def bfs(maze: Maze) -> Tuple[List[Tuple[int, int]], int]:
    """Breadth first search of a provided maze"""
    objectives = set(maze.getObjectives())
    return search_alg(maze, start=maze.getStart(), objectives=objectives, pop_right=False)


def dfs(maze: Maze) -> Tuple[List[Tuple[int, int]], int]:
    """Depth first search of a provided maze"""
    objectives = set(maze.getObjectives())
    return search_alg(maze, start=maze.getStart(), objectives=objectives)


def greedy(maze: Maze) -> Tuple[List[Tuple[int, int]], int]:
    """Greedy best first search of a provided maze using Manhattan Distance heuristic"""
    objectives = set(maze.getObjectives())
    return search_alg(maze, start=maze.getStart(), objectives=objectives,
                      heuristic=lambda node, pos, targets: max((manhattan_distance(pos, obj) for obj in targets), default=0),
                      sort_key=lambda node: node.heuristic)


def astar(maze: Maze) -> Tuple[List[Tuple[int, int]], int]:
    """A* search of a provided maze using Manhattan Distance for single objective or recursive A* MST for
    multiple objectives"""
    objectives = set(maze.getObjectives())
    start = maze.getStart()
    ret_val = parameterized_astar(maze, start, objectives)
    return ret_val


def parameterized_astar(maze: Maze, start: Tuple[int, int],
                        objectives: Set[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], int]:
    """Searches the provided maze, starting at the provided start position, to find the minimum path to the set of
    provided objective positions.  If no objectives provided, throw a ValueError.  If only one objective, then utilize
    A* with Manhattan distance, otherwise utilize A* with recursive A* MST heuristic"""
    if len(objectives) > 1:
        return optimized_search_alg(maze, start, objectives)
    elif len(objectives) == 1:
        goal = maze.getObjectives().pop()
        return search_alg(maze, start=start, objectives=objectives, sort_key=lambda z: z.combined,
                          heuristic=lambda w, pos, o: manhattan_distance(pos, goal),
                          wait_for_frontier=True)
    else:
        raise ValueError("0 Objectives Received")


def search_alg(maze: Maze, start: Tuple[int, int], objectives: Set[Tuple[int, int]],
               sort_key: Callable[[Node], int]=None, pop_right: bool=True, find_any: bool = False,
               heuristic: Callable=None, wait_for_frontier=False) -> Tuple[List[Tuple[int, int]], int]:
    """
        Generic customizable search algorithm function

        Args:
            maze (Maze): The maze to search
            start (Tuple[int, int]): The starting position to begin the search
            objectives (Set[Tuple[int, int]]): The objective positions to path through
            sort_key (Callable[[Node], int]): A lambda function that takes a node and returns a value to use to sort
                                                the frontier list, lowest to highest.  If None then the frontier is not
                                                sorted.
            pop_right (bool): If true then retrieves the item at the end of the list.  If false, then front of the list.
            find_any (bool): If true then terminate as soon as ANY objective is reached instead of all objectives.
            heuristic (Callable): A lambda function that takes a Node, a start position, and a set of objectives and
                                  returns an admissible and consistent integer estimate of remaining distance.
            wait_for_frontier (bool): If true, then don't stop until the last objective is returned from the frontier.
                                    Otherwise, will stop as soon as last objective is seen as a neighbor.


    """
    start_node = Node(maze, None, frozenset(objectives), start, 0, heuristic_func=heuristic)
    frontier = list()
    frontier.append(start_node)
    completed = set()
    while frontier:
        """Take the next value from the frontier"""
        if sort_key:
            frontier.sort(key=sort_key, reverse=True)
        if pop_right:
            next_value = frontier.pop()
        else:
            next_value = frontier.pop(0)
        """Check for either find_any or find_all termination objectives"""
        if find_any and next_value.position in objectives:
            completed.add(next_value)
            return next_value.get_path(), len(completed)
        elif next_value.is_done():
            completed.add(next_value)
            return next_value.get_path(), len(completed)
        """Iterate through node neighbors and add them to the frontier"""
        if next_value not in completed:
            for neighbor in maze.getNeighbors(*next_value.position):
                if not find_any:
                    """If the neighbor is also an objective, remove objective from the neighbors objective list"""
                    new_obj = next_value.obj.difference([neighbor])
                else:
                    new_obj = objectives

                node = Node(maze, next_value, new_obj, neighbor, next_value.cost + 1, heuristic_func=heuristic)

                """If not waiting for the node to appear in the frontier (A*), and this is the last objective, 
                terminate now"""
                if not wait_for_frontier and ((not find_any and node.is_done())
                                              or (find_any and node.position in objectives)):
                        completed.add(node)
                        return node.get_path(), len(completed)

                if node not in completed:
                    frontier.append(node)
            completed.add(next_value)

    """If we reached this point, there was no path to the objectives."""
    return (), 0
    #raise ValueError("No path to objectives from " + str(start))


def optimized_search_alg(maze: Maze, start: Tuple[int, int],
                         objectives: Set[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], int]:
    """
    An A* optimized version of the generic search function that utilizes a PriorityQueue instead of a list to
    maintain the frontier.  Does not require a consistent heuristic, and defaults to Node.prim for a heuristic.

    Args:
        maze (Maze): The maze to search on
        start (Tuple[int, int]): Starting position to begin the search
        objectives (Set[Tuple[int, int]]): The positions to path through

    Returns:
        Path through all objectives and count of states searched
    """
    frontier = PriorityQueue()
    heuristic = Node.prim
    start_node = Node(maze=maze, parent=None, objectives_remaining=frozenset(objectives), position=start, cost=0,
                      heuristic_func=heuristic)
    frontier.put(PrioritizedItem(priority=start_node.combined, item=start_node))

    completed = dict()
    closest = len(objectives)
    while not frontier.empty():
        """Get the lowest (cost + heuristic) PrioritizedItem from the priority queue"""
        prioritized_item = frontier.get()
        """Extract the actual node from the prioritized item"""
        next_value = prioritized_item.item

        """If this node is closer (less remaining objectives), print a message to the user so they know 
        we're making progress"""
        if len(next_value.obj) < closest:
            closest = len(next_value.obj)
            print("Remaining objectives: " + str(closest))

        """If this node has reached all objectives, then we're done and return it's path and length of explored 
        states"""
        if next_value.is_done():
            completed[next_value] = next_value.cost
            return next_value.get_path(), len(completed)

        """Find the neighboring states that we haven't already seen or that we have seen but at a higher cost
        and add them to the frontier"""
        if next_value not in completed or completed[next_value] > next_value.cost:
            for neighbor in maze.getNeighbors(*next_value.position):
                node = Node(maze, next_value, next_value.obj.difference([neighbor]), neighbor, next_value.cost + 1,
                            heuristic_func=heuristic)
                if node not in completed or completed[node] > node.cost:
                    frontier.put(PrioritizedItem(priority=node.combined, item=node))
            completed[next_value] = next_value.cost

    raise ValueError("No path to objectives")
