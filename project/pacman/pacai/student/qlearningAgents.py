from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.util.probability import flipCoin
import random

# import inspect


class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        # You can initialize Q-values here.
        self.q_values = {}

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """
        return self.q_values.get((state, action), 0.0)

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """

        actions = self.getLegalActions(state)

        if len(actions) == 0:
            return 0.0

        max_value = -float("inf")
        for action in actions:
            temp_value = self.getQValue(state, action)
            if temp_value > max_value:
                max_value = temp_value

        return max_value

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """

        actions = self.getLegalActions(state)

        if len(actions) == 0:
            return None

        max_value = -float("inf")
        max_action = None
        for action in actions:
            temp_value = self.getQValue(state, action)
            if temp_value > max_value:
                max_value = temp_value
                max_action = action

        return max_action

    def getAction(self, state):
        #     `pacai.agents.base.BaseAgent.getAction`:
        # Compute the action to take in the current state.
        # With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
        # we should take a random action and take the best policy action otherwise.
        # Note that if there are no legal actions, which is the case at the terminal state,
        # you should choose None as the action.
        legalActions = self.getLegalActions(state)

        if len(legalActions) == 0:
            return None

        # If flipCoin, do a random choice
        if flipCoin(self.getEpsilon()):
            return random.choice(legalActions)

        return self.getPolicy(state)

    def update(self, state, action, next_state, reward):
        #     `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
        # The parent class calls this to observe a state transition and reward.
        # You should do your Q-Value update here.
        # Note that you should never call this function, it will be called on your behalf.
        if (state, action) not in self.q_values:
            self.q_values[(state, action)] = 0.0
        alpha = self.getAlpha()
        discount = self.getDiscountRate()
        # Call getValue() to get the best next value
        next_val = self.getValue(next_state)

        new_val = _calc_new_val(reward, discount, next_val)

        self.q_values[(state, action)] = (1 - alpha) * self.q_values[
            (state, action)
        ] + (alpha * new_val)
        return None


def _calc_new_val(reward, discount, next_value):
    # HELPER Function: Calculate the new value for current state
    # by given the best value for the next state
    return reward + (discount * next_value)


class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(
        self, index, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **kwargs
    ):
        kwargs["epsilon"] = epsilon
        kwargs["gamma"] = gamma
        kwargs["alpha"] = alpha
        kwargs["numTraining"] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action


class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(
        self,
        index,
        extractor="pacai.core.featureExtractors.IdentityExtractor",
        **kwargs
    ):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)
        # You might want to initialize weights here.
        self.weights = {}

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            # print(self.weights)
            return self.weights
            # raise NotImplementedError()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        # QLearningAgent.getQValue`:
        # Should return `Q(state, action) = w * featureVector`,
        # where `*` is the dotProduct operator.

        features = self.featExtractor.getFeatures(self, state, action)
        # print(features)
        q_value = 0.0

        # Do dot-product of two vector
        for f in features.keys():
            # print(f)
            q_value += self.weights.get(f, 0.0) * features[f]
        return q_value

    def update(self, state, action, next_state, reward):
        # Should update your weights based on transition.
        features = self.featExtractor.getFeatures(self, state, action)
        next_value = self.getValue(next_state)
        q_value = self.getQValue(state, action)
        for f in features.keys():
            if f not in self.weights:
                self.weights[f] = 0.0
            correction = (
                _calc_new_val(reward, self.getDiscountRate(), next_value) - q_value
            )
            # Just follow the given formula
            self.weights[f] += self.alpha * correction * features[f]
        return None
