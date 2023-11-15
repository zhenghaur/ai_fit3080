# trainPerceptron.py
# -----------------
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


# This file contains feature extraction methods and harness
# code for data classification

import sys
import util
from pacman import Directions
from perceptronPacman import SingleLayerPerceptronPacman
import samples
import numpy as np
import math

TRAINING_SET_SIZE = 10000
TEST_SET_SIZE = 1000
DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 70

# quick and dirty
FEATURE_NAMES = ["closestFood", "closestGhost", "eatsFood", "foodCount"]


def walls(state):
    # Returns a list of (x, y) pairs of wall positions
    #
    # This version just returns all the current wall locations
    # extracted from the state data.  In later versions, this will be
    # restricted by distance, and include some uncertainty.

    wallList = []
    wallGrid = state.getWalls()
    width = wallGrid.width
    height = wallGrid.height
    for i in range(width):
        for j in range(height):
            if wallGrid[i][j] == True:
                wallList.append((i, j))
    return wallList


def inFront(object, facing, state):
    # Returns true if the object is along the corridor in the
    # direction of the parameter "facing" before a wall gets in the
    # way.

    pacman = state.getPacmanPosition()
    pacman_x = pacman[0]
    pacman_y = pacman[1]
    wallList = walls(state)

    # If Pacman is facing North
    if facing == Directions.NORTH:
        # Check if the object is anywhere due North of Pacman before a
        # wall intervenes.
        next = (pacman_x, pacman_y + 1)
        while not next in wallList:
            if next == object:
                return True
            else:
                next = (pacman_x, next[1] + 1)
        return False

    # If Pacman is facing South
    if facing == Directions.SOUTH:
        # Check if the object is anywhere due North of Pacman before a
        # wall intervenes.
        next = (pacman_x, pacman_y - 1)
        while not next in wallList:
            if next == object:
                return True
            else:
                next = (pacman_x, next[1] - 1)
        return False

    # If Pacman is facing East
    if facing == Directions.EAST:
        # Check if the object is anywhere due East of Pacman before a
        # wall intervenes.
        next = (pacman_x + 1, pacman_y)
        while not next in wallList:
            if next == object:
                return True
            else:
                next = (next[0] + 1, pacman_y)
        return False

    # If Pacman is facing West
    if facing == Directions.WEST:
        # Check if the object is anywhere due West of Pacman before a
        # wall intervenes.
        next = (pacman_x - 1, pacman_y)
        while not next in wallList:
            if next == object:
                return True
            else:
                next = (next[0] - 1, pacman_y)
        return False



def StringNameToNumber(numberString):

    if numberString == Directions.NORTH:
        return 0
    elif numberString == Directions.EAST:
        return 1
    elif numberString == Directions.SOUTH:
        return 2
    elif numberString == Directions.WEST:
        return 3
    elif numberString == Directions.STOP:
        return 4


def basicFeatureExtractorPacman(state):
    """
    A basic feature extraction function.
    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions
    ##
    """
    features = util.Counter()
    for action in state.getLegalActions():
        successor = state.generateSuccessor(0, action)
        foodCount = successor.getFood().count()
        featureCounter = util.Counter()
        featureCounter['foodCount'] = foodCount
        features[action] = featureCounter

    return features, state.getLegalActions()


def enhancedFeatureExtractorPacman(state):
    """
    Your feature extraction playground.
    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions
    ##
    """

    features = basicFeatureExtractorPacman(state)[0]
    for action in state.getLegalActions():
        features[action] = util.Counter(features[action], **furtherEnhancedPacmanFeatures(state, action))
    return features, state.getLegalActions()


def furtherEnhancedPacmanFeatures(state, action):
    """
    For each state, this function is called with each legal action.
    It should return a counter with { <feature name> : <feature value>, ... }
    """
    features = util.Counter()
    successor = state.generateSuccessor(0, action)  # Generate successor for current state
    pacmanPosition = successor.getPacmanPosition()
    closestFood = 0
    foodList = successor.getFood().asList()  # Make list of food
    closestGhost = 0
    ghostPosition = successor.getGhostPositions()  # Find ghost positions

    if not foodList:
        pass
    else:  # Find minimum distance to food
        closestFood = min([util.manhattanDistance(i, pacmanPosition) for i in foodList])

    if closestFood:
        closestFood = 5/closestFood
    else:
        closestFood = 25
    features['closestFood'] = closestFood  # Designate feature name for closest food

    if not ghostPosition:
        pass
    else:  # Find minimum distance to ghost
        closestGhost = min(util.manhattanDistance(i, pacmanPosition) for i in ghostPosition)
    if closestGhost:
        closestGhost = 10 * np.exp(closestGhost)
    else:
        closestGhost

    features['closestGhost'] = closestGhost  # Designate feature name for closest ghost
    features["eatsFood"] = 10 * (state.getNumFood() - successor.getNumFood())

    return features


def default(str):
    return str + ' [Default: %default]'


USAGE_STRING = """
  USAGE:      python trainPerceptron.py <options>
  EXAMPLES:   (1) python trainPerceptron.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python trainPerceptron.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """


def readCommand(argv):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-t', '--training', help=default('The size of the training set'), default=TRAINING_SET_SIZE, type="int")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=20, type="int")
    parser.add_option('-l', '--learning_rate', help=default("Learning rate to use in training"), default=1, type="float")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print("Doing classification")
    print("--------------------")

    if options.training <= 0:
        print("Training set size should be a positive integer (you provided: %d)" % options.training)
        print(USAGE_STRING)
        sys.exit(2)

    if options.test <= 0:
        print("Testing set size should be a positive integer (you provided: %d)" % options.test)
        print(USAGE_STRING)
        sys.exit(2)

    args['num_iterations'] = options.iterations
    args['training_size'] = options.training
    args['testing_size'] = options.test
    args["learning_rate"] = options.learning_rate

    print(args)
    print(options)

    return args, options


# Dictionary containing full path to .pkl file that contains the agent's training, validation, and testing data.
MAP_AGENT_TO_PATH_OF_SAVED_GAMES = {
    'FoodAgent': ('pacmandata/food_training.pkl', 'pacmandata/food_validation.pkl', 'pacmandata/food_test.pkl'),
    'StopAgent': ('pacmandata/stop_training.pkl', 'pacmandata/stop_validation.pkl', 'pacmandata/stop_test.pkl'),
    'SuicideAgent': (
    'pacmandata/suicide_training.pkl', 'pacmandata/suicide_validation.pkl', 'pacmandata/suicide_test.pkl'),
    'GoodReflexAgent': (
    'pacmandata/good_reflex_training.pkl', 'pacmandata/good_reflex_validation.pkl', 'pacmandata/good_reflex_test.pkl'),
    'ContestAgent': (
    'pacmandata/contest_training.pkl', 'pacmandata/contest_validation.pkl', 'pacmandata/contest_test.pkl')
}


def normalise_data(data, max_values, min_values):
    for feature_vector in data:
        for feature_name, value in feature_vector.items():
            feature_vector[feature_name] = (feature_vector[feature_name] - min_values[feature_name])/(max_values[feature_name] - min_values[feature_name])

    return data


def find_max_and_min_feature_values(data):
    feature_names = data[0].keys()
    max_values = dict(zip(feature_names, [-1*math.inf]*len(feature_names)))
    min_values = dict(zip(feature_names, [math.inf]*len(feature_names)))

    for feature_vector in data:
        for feature_name, value in feature_vector.items():
            if max_values[feature_name] < value:
                max_values[feature_name] = value
            if min_values[feature_name] > value:
                min_values[feature_name] = value

    return max_values, min_values


def convertToBinary(features, labels):
    binary_labels = []
    for i in range(len(features)):
        # print(labels[i])
        temp = [1 if StringNameToNumber(move) == labels[i] else 0 for move in features[i][1]]
        binary_labels.extend(temp)

    return binary_labels


def to_numpy_binary_data(trainingData):
    binary_features = []
    for i in range(len(trainingData)):
        legal_moves = trainingData[i][1]

        for move in legal_moves:
            features = trainingData[i][0][move]

            binary_features.append([features[feature_name] for feature_name in FEATURE_NAMES])
            # binary_features.append(features)

    return np.array(binary_features)


def runClassifier(args):
    featureFunction = enhancedFeatureExtractorPacman

    # data sizes and number of training iterations
    numTraining = args['training_size']
    numTest = args['testing_size']
    num_iterations = args['num_iterations']
    learning_rate = args["learning_rate"]

    # load the data sets
    print("loading data...")
    trainingData = MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][0]
    validationData = MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][1]
    testData = MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][2]
    rawTrainingData, trainingLabels = samples.loadPacmanData(trainingData, numTraining)
    rawValidationData, validationLabels = samples.loadPacmanData(validationData, numTest)
    rawTestData, testLabels = samples.loadPacmanData(testData, numTest)

    # Extract features
    print("Extracting features...")

    trainingData = list(map(featureFunction, rawTrainingData))[:-1]
    validationData = list(map(featureFunction, rawValidationData))[:-1]
    testData = list(map(featureFunction, rawTestData))[:-1]

    trainingLabels = list(map(StringNameToNumber, trainingLabels[:-1]))
    validationLabels = list(map(StringNameToNumber, validationLabels[:-1]))
    testLabels = list(map(StringNameToNumber, testLabels[:-1]))


    # convert the data to binary labels format as numpy arrays
    trainingLabels = convertToBinary(trainingData, trainingLabels)
    trainingData = to_numpy_binary_data(trainingData)

    validationLabels = convertToBinary(validationData, validationLabels)
    validationData = to_numpy_binary_data(validationData)

    testLabels = convertToBinary(testData, testLabels)
    testData = to_numpy_binary_data(testData)

    # print(trainingData)

    # find the minimum and maximum values in the data set and put a one at the front for bias max snd 0 for bias min
    max_values = np.concatenate([[1], np.max(trainingData, axis=0)])
    min_values = np.concatenate([[0], np.min(trainingData, axis=0)])

    # print(max_values)
    # print(min_values)
    # np.savetxt("max_and_min_values.txt", np.vstack([max_values, min_values]))

    # add bias to each data set
    trainingData = np.c_[np.ones(len(trainingData)), trainingData]
    validationData = np.c_[np.ones(len(validationData)), validationData]
    testData = np.c_[np.ones(len(testData)), testData]

    # scale the training data
    trainingData = (trainingData - min_values)/(max_values - min_values)
    validationData = (validationData - min_values)/(max_values - min_values)
    testData = (testData - min_values)/(max_values - min_values)

    # create the classifier
    classifier = SingleLayerPerceptronPacman(num_iterations=num_iterations, learning_rate=learning_rate)

    # Conduct training and testing
    print("Training...")
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)

    print("Testing...")
    test_performance = classifier.evaluate(testData, testLabels)
    print(test_performance)

    # weight_save_path = args['weights_path']
    np.savetxt(f"./logs/q3_weights.model", np.vstack([classifier.weights, max_values, min_values]),
                header=f"{{'num_iterations': {classifier.max_iterations}, 'learning_rate': {classifier.learning_rate}}}")

    return classifier


if __name__ == '__main__':
    # Read input
    args, options = readCommand(sys.argv[1:])
    # Run classifier
    runClassifier(args)