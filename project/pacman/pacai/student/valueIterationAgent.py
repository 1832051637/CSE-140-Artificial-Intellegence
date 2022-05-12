from pacai.agents.learning.value import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate=0.9, iters=100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}  # A dictionary which holds the q-values for each state.

        # Compute the values here.
        states = self.mdp.getStates()
        temp_values = {}

        for _ in range(self.iters):
            for state in states:
                if self.mdp.isTerminal(state):
                    temp_values[state] = 0.0
                    continue
                max_value = -float("inf")
                for action in self.mdp.getPossibleActions(state):
                    q_value = self.getQValue(state, action)
                    if q_value > max_value:
                        max_value = q_value

                temp_values[state] = max_value
            # Update values dictionary
            self.values = temp_values.copy()
            # for sta, val in temp_values.items():
            #     self.values[sta] = val

        # raise NotImplementedError()

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        # print(self.values.keys())
        return self.values.get(state, 0.0)

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)

    def getQValue(self, state, action):
        #     pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
        # The q-value of the state action pair (after the indicated number of value iteration passes).
        # Note that value iteration does not necessarily create this quantity,
        # and you may have to derive it on the fly.
        q_value = 0
        # print(self.mdp.getTransitionStatesAndProbs(state, action))
        for nextState, probability in self.mdp.getTransitionStatesAndProbs(
            state, action
        ):
            next_value = self.values.get(nextState, 0.0)
            q_value += probability * (
                self.mdp.getReward(state, action, nextState)
                + (self.discountRate * next_value)
            )

        return q_value
        # return self.computeQValueFromValues(state, action)

    def getPolicy(self, state):
        possibleActions = self.mdp.getPossibleActions(state)
        max_action = None
        max_value = -float("inf")

        for action in possibleActions:
            temp_value = self.getQValue(state, action)
            if temp_value > max_value:
                max_value = temp_value
                max_action = action

        # print(max_action)
        return max_action
        # return self.computeActionFromValues(state)

