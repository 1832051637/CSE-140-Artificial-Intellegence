import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.distance import manhattan


class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [
            self.evaluationFunction(gameState, action) for action in legalMoves
        ]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(
            bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        # print(action)
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.getScaredTimer() for ghostState in newGhostStates
        ]

        # *** Your Code Here ***

        all_scared = True
        for ghost_scared_time in newScaredTimes:
            if ghost_scared_time == 0:
                all_scared = False
                break

        if all_scared:
            return successorGameState.getScore() + 100

        newFood = successorGameState.getFood().asList()
        minFoodist = float("inf")
        nearest_food = None
        for food in newFood:
            if (manhattan(newPosition, food) < minFoodist):
                minFoodist = manhattan(newPosition, food)
                nearest_food = food

        # Get the nearest ghost distance to the nearest food
        min_ghost_dis = float("inf")
        for ghost in successorGameState.getGhostPositions():
            if nearest_food is not None and \
                    (manhattan(nearest_food, ghost) < min_ghost_dis):
                min_ghost_dis = manhattan(nearest_food, ghost)

            # Warning: If the ghost is too close to pacman
            if (manhattan(newPosition, ghost) < 2):
                return -float('inf')

        # Penalty to not move
        deduction = 0
        if action == 'Stop':
            deduction = -30

        # print(min_ghost_dis, minFoodist)

        # So, the closer the nearest food is,
        # and the farther the nearest ghost to the nearest food is ,
        # the evaluation function is higher.
        return successorGameState.getScore(
        ) + min_ghost_dis + 1.0 / minFoodist + deduction


class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getTreeDepth(self):
        return super().getTreeDepth()

    def getEvaluationFunction(self):
        return super().getEvaluationFunction()

    def getAction(self, gameState):
        # Return the action
        val = self.getValue(gameState, self.index, 0)
        return val[1]

    def getValue(self, state, index, depth):
        # print(state.getNumAgents(), mindex, index)

        # Check the ending condition: game over, no more valid action,
        # or tree depth exceeds
        if len(state.getLegalActions(index)) == 0 \
                or depth == self.getTreeDepth() \
                or state.isWin() or state.isLose():
            return (self.getEvaluationFunction()(state), "")
        if index == 0:
            return self.max_value(state, index, depth)
        else:
            return self.min_value(state, index, depth)

    def max_value(self, state, index, depth):
        max_val = (-float("inf"), '')
        agent_num = state.getNumAgents()
        legal_actions = state.getLegalActions(index)

        for action in legal_actions:
            successor = state.generateSuccessor(index, action)
            new_index = index + 1
            new_depth = depth

            # Update depth and index if this is a pacman
            if new_index == agent_num:
                new_index = new_index % agent_num
                new_depth += 1
            temp_val = (self.getValue(successor, new_index,
                                      new_depth)[0], action)
            if temp_val[0] >= max_val[0]:
                max_val = temp_val

        return max_val

    def min_value(self, state, index, depth):
        min_val = (float("inf"), '')
        agent_num = state.getNumAgents()
        legal_actions = state.getLegalActions(index)
        for action in legal_actions:
            successor = state.generateSuccessor(index, action)
            new_index = index + 1
            new_depth = depth

            # Update depth and index if this is a pacman
            if new_index == agent_num:
                new_index = new_index % agent_num
                new_depth += 1
            temp_val = (self.getValue(successor, new_index,
                                      new_depth)[0], action)
            if temp_val[0] <= min_val[0]:
                min_val = temp_val

        return min_val


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getTreeDepth(self):
        return super().getTreeDepth()

    def getEvaluationFunction(self):
        return super().getEvaluationFunction()

    def getAction(self, gameState):
        # Return the action
        val = self.getValue_ab(gameState, self.index, 0, -float("inf"),
                               float("inf"))
        return val[1]

    def getValue_ab(self, state, index, depth, alpha, beta):
        # print(state.getNumAgents(), mindex, index)

        # Check the ending condition: game over, no more valid action,
        # or tree depth exceeds
        if len(state.getLegalActions(index)) == 0 \
                or depth == self.getTreeDepth():
            # \
            # or state.isWin() or state.isLose():
            return (self.getEvaluationFunction()(state), "")
        if index == 0:
            return self.max_value_ab(state, index, depth, alpha, beta)
        else:
            return self.min_value_ab(state, index, depth, alpha, beta)

    def max_value_ab(self, state, index, depth, a, b):
        max_val = (-float("inf"), '')
        agent_num = state.getNumAgents()
        legal_actions = state.getLegalActions(index)

        for action in legal_actions:
            successor = state.generateSuccessor(index, action)
            new_index = index + 1
            new_depth = depth

            # Update depth and index if this is a pacman
            if new_index == agent_num:
                new_index = new_index % agent_num
                new_depth += 1
            temp_val = (self.getValue_ab(successor, new_index, new_depth, a,
                                         b)[0], action)
            if temp_val[0] >= max_val[0]:
                max_val = temp_val

            if max_val[0] > b:
                # We dont need to continue if current max > beta
                return max_val

            # a = max(a, max_val[0])
            if max_val[0] >= a:
                a = max_val[0]

        return max_val

    def min_value_ab(self, state, index, depth, a, b):
        min_val = (float("inf"), '')
        agent_num = state.getNumAgents()
        legal_actions = state.getLegalActions(index)
        for action in legal_actions:
            successor = state.generateSuccessor(index, action)
            new_index = index + 1
            new_depth = depth

            # Update depth and index if this is a pacman
            if new_index == agent_num:
                new_index = new_index % agent_num
                new_depth += 1
            temp_val = (self.getValue_ab(successor, new_index, new_depth, a,
                                         b)[0], action)

            if temp_val[0] <= min_val[0]:
                min_val = temp_val

            if min_val[0] < a:
                # We dont need to continue if current min <= alpha
                return min_val

            # b = min(b, min_val[0])
            if min_val[0] <= b:
                b = min_val[0]
        return min_val


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        # Return the action
        val = self.getValue(gameState, self.index, 0)
        return val[1]

    def getValue(self, state, index, depth):
        # print(state.getNumAgents(), mindex, index)

        # Check the ending condition: game over, no more valid action,
        # or tree depth exceeds
        if len(state.getLegalActions(index)) == 0 \
                or depth == self.getTreeDepth() \
                or state.isWin() or state.isLose():
            return (self.getEvaluationFunction()(state), "")
        if index == 0:
            return self.max_value(state, index, depth)
        else:
            return self.exp_value(state, index, depth)

    def max_value(self, state, index, depth):
        max_val = (-float("inf"), '')
        agent_num = state.getNumAgents()
        legal_actions = state.getLegalActions(index)

        for action in legal_actions:
            successor = state.generateSuccessor(index, action)
            new_index = index + 1
            new_depth = depth

            # Update depth and index if this is a pacman
            if new_index == agent_num:
                new_index = new_index % agent_num
                new_depth += 1
            temp_val = (self.getValue(successor, new_index,
                                      new_depth)[0], action)
            if temp_val[0] >= max_val[0]:
                max_val = temp_val

        return max_val

    def exp_value(self, state, index, depth):
        exp_val = 0  # expectation value
        agent_num = state.getNumAgents()
        legal_actions = state.getLegalActions(index)

        probility = 1.0 / len(legal_actions)

        for action in legal_actions:
            successor = state.generateSuccessor(index, action)
            new_index = index + 1
            new_depth = depth

            # Update depth and index if this is a pacman
            if new_index == agent_num:
                new_index = new_index % agent_num
                new_depth += 1
            temp_val = (self.getValue(successor, new_index,
                                      new_depth)[0], action)
            # if temp_val[0] <= exp_val[0]:
            #     exp_val = temp_val
            # Calculate the expectation for expectimax
            exp_val = (exp_val + probility * temp_val[0])

        return exp_val, action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """

    return currentGameState.getScore()


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
