# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        iteration_values = util.Counter()

        for i in range(iterations):
            for state in mdp.getStates():
          
                max_action_value = -99999
                max_action = None

                actions = mdp.getPossibleActions(state)
                if len(actions) == 0:
                    max_action_value = mdp.getReward(state, None, None)
                    max_action = None
                    max_state_prime = None
                else:
                    for action in actions:
                        summation = 0
                        for state_prime, prob in mdp.getTransitionStatesAndProbs(state, action):
                            if mdp.isTerminal(state_prime):
                                iteration_values[state] = mdp.getReward(state, 'exit', 'TERMINAL_STATE')
                            else:
                                utility = self.values[state_prime]
                                summation += utility*prob

                        if summation > max_action_value:
                            max_action_value = summation
                            max_action = action

                iteration_values[state] = mdp.getReward(state, None, None) + discount*max_action_value
            # Update at the end of each iteration
            self.values = iteration_values.copy()


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        q = 0
        for state_prime, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            if self.mdp.isTerminal(state_prime):
                q += prob*self.mdp.getReward(state, 'exit', 'TERMINAL_STATE')
            else:
                q += prob*self.values[state_prime]*self.discount
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        best_action = None
        max_action_value = -9999

        for action in self.mdp.getPossibleActions(state):
            q = self.computeQValueFromValues(state, action)
            if q > max_action_value:
                max_action_value = q
                best_action = action

        return best_action


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
