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
from util import PriorityQueue
from mdp import MarkovDecisionProcess
from statistics import mean

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
        self.mdp: MarkovDecisionProcess = mdp
        self.discount = discount
        self.iterations = iterations
        self.nextValues = util.Counter()
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()
        

    def runValueIteration(self):
        for i in range(self.iterations):
            for s in self.mdp.getStates():
                actions = self.mdp.getPossibleActions(s)
                if self.mdp.isTerminal(s): self.values[s] = 0
                else: self.nextValues[s] = max([self.getQValue(s, act) for act in actions])

            self.values = self.nextValues.copy()
        # Write value iteration code here
        # for i in range(self.iterations):
        #     for state in self.mdp.getStates():
        #         if self.mdp.isTerminal(state): self.values[state] = 0
        #         else: self.values[state] = max([self.getQValue(state, act) for act in self.mdp.getPossibleActions(state)])


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
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            r = self.mdp.getReward(state, action, nextState)
            q += prob * (r + self.discount * self.getValue(nextState))
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state): return None

        bestAction = None
        bestValue = -100_000
        for action in self.mdp.getPossibleActions(state):
            q = self.computeQValueFromValues(state, action)
            if q > bestValue:
                bestValue = q
                bestAction = action
        return bestAction

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
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
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
        states = self.mdp.getStates()
        size = len(states)
        step = 1
        index = 0
        for i in range(self.iterations):
            s = states[index]
            actions = self.mdp.getPossibleActions(s)
            if self.mdp.isTerminal(s): self.values[s] = 0
            else: self.nextValues[s] = max([self.getQValue(s, act) for act in actions])
            self.values = self.nextValues.copy()

            if index == size - 1: index = -1
            index += step                


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def calcDiff(self, state):
        return abs(self.values[state] - max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)]))


    def runValueIteration(self):
        predecessors = {}
        for st in self.mdp.getStates():
            predecessors[st] = set()
        
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    predecessors[nextState].add(state)

        pq = PriorityQueue()

        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state): continue
            diff = self.calcDiff(state)
            pq.push(state, -diff)

        for i in range(self.iterations):
            if pq.isEmpty(): break
            state = pq.pop()
            if self.mdp.isTerminal(state): continue

            self.values[state] =  max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])

            for pred in predecessors[state]:
                diff = self.calcDiff(pred)
                if diff > self.theta: pq.update(pred, -diff)
