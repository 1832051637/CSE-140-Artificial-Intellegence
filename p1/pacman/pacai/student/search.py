"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

from ..util.stack import Stack
from ..util.queue import Queue
from ..util.priorityQueue import PriorityQueue


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of action_lst that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    stck = Stack()
    visited = []
    # Push the start state and empty aciton list
    stck.push((problem.startingState(), []))

    while not stck.isEmpty():
        top = stck.pop()
        curr_node = top[0]
        action_lst = top[1]
        if curr_node not in visited:  # if current node unvisited
            visited.append(curr_node)
            if problem.isGoal(curr_node):
                return action_lst
            for each_state in problem.successorStates(curr_node):
                position = each_state[0]
                direction = each_state[1]
                new_action_lst = action_lst.copy()
                new_action_lst.append(direction)
                stck.push((position, new_action_lst))

    # if failed
    raise NotImplementedError()


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    q = Queue()
    visited = []
    # Push the start state and empty aciton list
    q.push((problem.startingState(), []))

    while not q.isEmpty():
        top = q.pop()
        curr_node = top[0]
        action_lst = top[1]
        if curr_node not in visited:  # if current node unvisited
            visited.append(curr_node)
            if problem.isGoal(curr_node):
                return action_lst
            for each_state in problem.successorStates(curr_node):
                position = each_state[0]
                direction = each_state[1]
                new_action_lst = action_lst.copy()
                new_action_lst.append(direction)
                q.push((position, new_action_lst))

    # if failed
    raise NotImplementedError()


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    # Uniform Cost Search is a version of Dijkstra
    min_heap = PriorityQueue()
    visited = []
    # Push the start state and empty aciton list
    min_heap.push((problem.startingState(), []), 0)

    while not min_heap.isEmpty():
        top = min_heap.pop()
        curr_node = top[0]
        action_lst = top[1]
        if curr_node not in visited:  # if current node unvisited
            visited.append(curr_node)
            if problem.isGoal(curr_node):
                return action_lst
            for each_state in problem.successorStates(curr_node):
                position = each_state[0]
                direction = each_state[1]
                new_action_lst = action_lst.copy()
                new_action_lst.append(direction)
                new_cost = problem.actionsCost(new_action_lst)
                min_heap.push((position, new_action_lst), new_cost)

    # if failed
    raise NotImplementedError()


def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    min_heap = PriorityQueue()
    visited = []
    # Push the start state and empty aciton list
    min_heap.push((problem.startingState(), []),
                  heuristic(problem.startingState(), problem))

    while not min_heap.isEmpty():
        top = min_heap.pop()
        curr_node = top[0]
        action_lst = top[1]
        if curr_node not in visited:  # if current node unvisited
            visited.append(curr_node)
            if problem.isGoal(curr_node):
                return action_lst
            for each_state in problem.successorStates(curr_node):
                position = each_state[0]
                direction = each_state[1]
                new_action_lst = action_lst.copy()
                new_action_lst.append(direction)
                new_cost = problem.actionsCost(new_action_lst) + heuristic(
                    position, problem)
                min_heap.push((position, new_action_lst), new_cost)

    # if failed
    raise NotImplementedError()
