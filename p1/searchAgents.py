"""
This file contains incomplete versions of some agents that can be selected to control Pacman.
You will complete their implementations.

Good luck and happy searching!
"""

import logging
from pacai.student.search import uniformCostSearch
from pacai.core.distance import manhattan, maze
from pacai.core.actions import Actions
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.core.directions import Directions

from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent


class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function.
    See the `pacai.core.search.position.PositionSearchProblem` class for an example of
    a working SearchProblem.

    Additional methods to implement:

    `pacai.core.search.problem.SearchProblem.startingState`:
    Returns the start state (in your search space,
    NOT a `pacai.core.gamestate.AbstractGameState`).

    `pacai.core.search.problem.SearchProblem.isGoal`:
    Returns whether this search state is a goal state of the problem.

    `pacai.core.search.problem.SearchProblem.successorStates`:
    Returns successor states, the actions they require, and a cost of 1.
    The following code snippet may prove useful:
    ```
        successors = []

        for action in Directions.CARDINAL:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            hitsWall = self.walls[next_x][next_y]

            if (not hitsWall):
                # Construct the successor.

        return successors
    ```
    """
    def __init__(self, startingGameState):
        super().__init__()

        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top = self.walls.getHeight() - 2
        right = self.walls.getWidth() - 2

        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                logging.warning("Warning: no food in corner " + str(corner))

        # *** Your Code Here ***
        self.starting_state = (self.startingPosition, [])
        # raise NotImplementedError()

    def startingState(self):
        return self.starting_state

    def isGoal(self, state):
        return len(state[1]) == 4  # If all four corners have been visted

    def successorStates(self, state):
        successors = []
        curr_pos = state[0]

        # Successor = ((nextX, nextY), visited corners)
        for action in Directions.CARDINAL:
            x, y = curr_pos
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            hitsWall = self.walls[next_x][next_y]

            if not hitsWall:
                # Build successors
                visited_corner = state[1].copy()
                if (next_x, next_y) in self.corners and (
                        next_x,
                        next_y,
                ) not in visited_corner:
                    visited_corner.append((next_x, next_y))
                next_state = ((next_x, next_y), visited_corner)
                successors.append((next_state, action, 1))
        # Expand node +1
        self._numExpanded += 1
        return successors

    def actionsCost(self, actions):
        """
        Returns the cost of a particular sequence of actions.
        If those actions include an illegal move, return 999999.
        This is implemented for you.
        """

        if actions is None:
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999

        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
    """

    # Useful information.
    # corners = problem.corners  # These are the corner coordinates
    # walls = problem.walls  # These are the walls of the maze, as a Grid.

    # *** Your Code Here ***
    corners = problem.corners  # All four corners
    unvisited = []  # Unvisited corners
    visited = state[1]  # Visited corners
    curr_pos = state[0]  # Current position
    heuristic_value = 0  # Heuristic value to return

    # List of all Unvisted Corners
    unvisited = list(set(corners) - set(visited))

    # Find the shortest manhattan distance from current coordinate
    # to all other unvisted corners
    while len(unvisited) > 0:
        min_distance = float("inf")
        closest_corner = ()
        for corner in unvisited:
            temp_distance = manhattan(curr_pos, corner)
            if temp_distance < min_distance:
                min_distance = temp_distance
                closest_corner = corner
        heuristic_value += min_distance
        curr_pos = closest_corner
        unvisited.remove(closest_corner)

    return heuristic_value
    # return heuristic.null(state, problem)  # Default to trivial solution


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.
    First, try to come up with an admissible heuristic;
    almost all admissible heuristics will be consistent as well.

    If using A* ever finds a solution that is worse than what uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!
    On the other hand, inadmissible or inconsistent heuristics may find optimal solutions,
    so be careful.

    The state is a tuple (pacmanPosition, foodGrid) where foodGrid is a
    `pacai.core.grid.Grid` of either True or False.
    You can call `foodGrid.asList()` to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, `problem.walls` gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use.
    For example, if you only want to count the walls once and store that value, try:
    ```
    problem.heuristicInfo['wallCount'] = problem.walls.count()
    ```
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount'].
    """

    position, foodGrid = state

    # *** Your Code Here ***
    unvisited_foods = foodGrid.asList()
    heuristic_value = 0

    # Check if we still have food to eat
    if len(unvisited_foods) > 0:
        # Using heuristic = distance(pacman, nearest food)
        #                   + distance(nearest food, farthest food to nearest food)
        min_distance = float("inf")
        closest_food_pos = ()
        for food in unvisited_foods:
            temp_dis = maze(position, food, problem.startingGameState)
            if temp_dis < min_distance:
                min_distance = temp_dis
                closest_food_pos = food

        max_food_distance = 0
        for food in unvisited_foods:
            temp_dis = maze(food, closest_food_pos, problem.startingGameState)
            if temp_dis > max_food_distance:
                max_food_distance = temp_dis

        heuristic_value = min_distance + max_food_distance
    return heuristic_value
    # return heuristic.null(state, problem)  # Default to the null heuristic.


class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches.
    """
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        currentState = state

        while currentState.getFood().count() > 0:
            nextPathSegment = self.findPathToClosestDot(
                currentState)  # The missing piece
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception(
                        "findPathToClosestDot returned an illegal move: %s!\n%s"
                        % (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info("Path found with cost %d." % len(self._actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from gameState.
        """

        # Here are some useful elements of the startState
        # startPosition = gameState.getPacmanPosition()
        # food = gameState.getFood()
        # walls = gameState.getWalls()
        # problem = AnyFoodSearchProblem(gameState)

        # *** Your Code Here ***
        problem = AnyFoodSearchProblem(gameState)

        # Here I just use the UCS, but BFS also works
        return uniformCostSearch(problem)

        # raise NotImplementedError()


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    The state space and successor function do not need to be changed.

    The class definition above, `AnyFoodSearchProblem(PositionSearchProblem)`,
    inherits the methods of `pacai.core.search.position.PositionSearchProblem`.

    You can use this search problem to help you fill in
    the `ClosestDotSearchAgent.findPathToClosestDot` method.

    Additional methods to implement:

    `pacai.core.search.position.PositionSearchProblem.isGoal`:
    The state is Pacman's position.
    Fill this in with a goal test that will complete the problem definition.
    """
    def __init__(self, gameState, start=None):
        super().__init__(gameState, goal=None, start=start)

        # Store the food for later reference.
        self.food = gameState.getFood()

    def isGoal(self, state):
        # Return True if this position is a food
        return self.food[state[0]][state[1]]


class ApproximateSearchAgent(BaseAgent):
    """
    Implement your contest entry here.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Get a `pacai.bin.pacman.PacmanGameState`
    and return a `pacai.core.directions.Directions`.

    `pacai.agents.base.BaseAgent.registerInitialState`:
    This method is called before any moves are made.
    """
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
