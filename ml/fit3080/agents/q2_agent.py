# qlearningAgents.py
# ------------------
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


from game import *
from agents.learningAgents import ReinforcementAgent
from featureExtractors import *
from pacman import GameState

import random,util,math
import numpy as np
from game import Directions


class Q2Agent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, pretrained_values=None, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """

        self.index = 0  # This is always Pacman

        ReinforcementAgent.__init__(self, **args)

        if pretrained_values:
            flattenedQ = np.loadtxt(pretrained_values)
            width, height = flattenedQ.shape
            self.Q_values = flattenedQ.reshape(int(width/5), height, 5)
            self.learningQvalues = False
            self.numTraining = 0 # no training
            self.epsilon = 0.0  # no exploration
            self.alpha = 0.0  # no learning
        else:
            self.Q_values = None
            self.learningQvalues = True
            self.epsilon_to_write = self.epsilon
            self.alpha_to_write = self.epsilon

    def registerInitialState(self, state):
        """
        Don't modify this method!
        """

        if self.Q_values is None:
            self.Q_values = np.zeros((state.getWalls().width, state.getWalls().height, 5))
            self.learningQvalues = True

        elif self.isInTesting() and self.learningQvalues:
            width, height, depth = self.Q_values.shape
            flattenedQ = self.Q_values.reshape((width*depth, height))

            np.savetxt(f"./logs/{state.data.layout.layoutFileName[:-4]}.model", flattenedQ,
                       header=f"{{'gamma':{self.discount}, 'num_training':{self.numTraining}, 'epsilon':{self.epsilon_to_write}, 'alpha':{self.alpha_to_write}}}")

            self.learningQvalues = False

        self.startEpisode()
        if self.episodesSoFar == 0:
            print('Beginning %d episodes of Training' % (self.numTraining))

    def getActionIndex(self, action):
        """
        This function maps an action to the correct index in the Q-table
        """
        if action == Directions.NORTH:
            return 0
        elif action == Directions.SOUTH:
            return 1
        elif action == Directions.EAST:
            return 2
        elif action == Directions.WEST:
            return 3
        else:
            return 4

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    ########################################################################
    ####            CODE FOR YOU TO MODIFY STARTS HERE                  ####
    ########################################################################

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        (x, y) = state.getPacmanPosition()
        return self.Q_values[x][y][self.getActionIndex(action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.

          Note that if there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.

          HINT: You might want to use self.getLegalActions(state)
        """

        "*** YOUR CODE HERE ***"

        if (len(self.getLegalActions(state)) == 0):
            return 0.0

        maxQValue = -math.inf
        for action in self.getLegalActions(state):
            qValue = self.getQValue(state, action)
            if qValue > maxQValue:
                maxQValue = qValue

        return maxQValue
    
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
          HINT: You might want to use self.getLegalActions(state)
        """
        "*** YOUR CODE HERE ***"

        maxQValue = -math.inf
        bestAction = None
        for action in self.getLegalActions(state):
            qValue = self.getQValue(state, action)
            if qValue > maxQValue:
                maxQValue = qValue
                bestAction = action

        return bestAction
    
    def getAction(self, state: GameState):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
          HINT: You might want to use self.getLegalActions(state)
        """

        legalActions = self.getLegalActions(state)
        action = None

        "*** YOUR CODE STARTS HERE ***"

        if (len(legalActions) == 0):
            return action
    
        if (util.flipCoin(1 - self.epsilon)):
            action = self.computeActionFromQValues(state)
        else:
            action = random.choice(legalActions)
        
        "*** YOUR CODE ENDS HERE ***"

        self.doAction(state, action)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        (x, y) = state.getPacmanPosition()
        qValue = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))
        self.Q_values[x][y][self.getActionIndex(action)] = qValue
