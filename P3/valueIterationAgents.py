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
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        """*** YOUR CODE HERE ***"""

        for i in range(self.iterations):
            val = self.values.copy()
            for state in self.mdp.getStates():

                if self.mdp.isTerminal(state) is False:
                    maxQ = float("-inf")
                    for action in self.mdp.getPossibleActions(state):
                        computeQ = self.computeQValueFromValues(state, action)
                        if computeQ > maxQ:
                            maxQ = computeQ
                    val[state] = maxQ

            self.values = val

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
        "*** YOUR CODE HERE ***"
        qVal = 0
        for s, p in self.mdp.getTransitionStatesAndProbs(state, action):
            r = self.mdp.getReward(state, action, s)
            qVal += p * (r + self.discount * self.values[s])
        return qVal

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        nextAction = None
        if self.mdp.isTerminal(state) is False:
            val = float("-inf")
            for action in self.mdp.getPossibleActions(state):
                qVal = self.computeQValueFromValues(state, action)
                if qVal > val:
                    val = qVal
                    nextAction = action

        return nextAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*
        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """*** YOUR CODE HERE ***"""

        for i in range(self.iterations):
            states = self.mdp.getStates()
            s = states[i % len(states)]
            if self.mdp.isTerminal(s) is False:
                valAction = util.Counter()
                for action in self.mdp.getPossibleActions(s):
                    valAction[action] = self.computeQValueFromValues(s, action)
                self.values[s] = valAction[valAction.argMax()]


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """*** YOUR CODE HERE ***"""

        predecessors = dict()
        priorityQueue = util.PriorityQueue()
        states = self.mdp.getStates()
        for s in states:
            predecessors[s] = set()

        for s in states:
            if self.mdp.isTerminal(s) is False:

                qVal = []
                for action in self.mdp.getPossibleActions(s):
                    for nextState, probability in self.mdp.getTransitionStatesAndProbs(s, action):
                        if probability > 0:
                            predecessors[nextState].add(s)
                    qVal.append(self.computeQValueFromValues(s, action))
                priorityQueue.push(s, -abs(self.values[s] - max(qVal)))
        for i in range(self.iterations):
            if not priorityQueue.isEmpty():
                s = priorityQueue.pop()
                if self.mdp.isTerminal(s) is False:
                    actions = self.mdp.getPossibleActions(s)
                    qVal = [self.computeQValueFromValues(s, action) for action in actions]
                    self.values[s] = max(qVal)
                for p in predecessors[s]:
                    actions = self.mdp.getPossibleActions(p)
                    qVal = [self.computeQValueFromValues(p, action) for action in actions]
                    absVal = abs(self.values[p] - max(qVal))
                    if absVal > self.theta:
                        priorityQueue.update(p, -absVal)
