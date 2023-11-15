# pacmanAgents.py
# ---------------
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


from pacman import Directions
from game import Agent
import random
import game
import util
import trainPerceptron
import samples

from trainPerceptron import enhancedFeatureExtractorPacman, FEATURE_NAMES
from perceptronPacman import SingleLayerPerceptronPacman
import numpy as np
from pacman import GameState


class Q3Agent(Agent):

    def __init__(self, weights_path="./logs/q3_weights.model"):

        print('-------------Testing trained Perceptron Pacman-------------------')

        # Use the same feature function to extract features from
        self.featureFunction = enhancedFeatureExtractorPacman


        self.perceptron = SingleLayerPerceptronPacman()
        weights_and_scaling_values = np.loadtxt(weights_path)
        # Max and Min values in the data set. Used in agent to scale features.

        self.perceptron.weights = weights_and_scaling_values[0]
        self.max_values = weights_and_scaling_values[1]
        self.min_values = weights_and_scaling_values[2]


    def getAction(self, state: GameState):
        """
        Takes a game state object and selects an action for Pac-man using the trained perceptron
        to determine the quality of each action.
        """

        features = self.featureFunction(state)[0]

        action_values = {}
        for action, feature_dict in features.items():
            feature_vector = np.array([feature_dict[feature_name] for feature_name in FEATURE_NAMES])
            feature_vector = np.hstack([[1], feature_vector])
            feature_vector = (feature_vector - self.min_values) / (self.max_values - self.min_values)
            value = self.perceptron.predict(feature_vector)
            action_values[action] = value

        return max(action_values, key=action_values.get)


