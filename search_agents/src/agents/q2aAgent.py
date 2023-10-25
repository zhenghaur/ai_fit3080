import logging
import random

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState
from util import manhattanDistance
from math import inf
import heapq


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class Q2A_Agent(Agent):

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '3'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    @log_function
    def getAction(self, gameState: GameState):
        """
            Returns the minimax action from the current gameState using self.depth
            and self.evaluationFunction.

            Here are some method calls that might be useful when implementing minimax.

            gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

            gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

            gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        logger = logging.getLogger('root')
        logger.info('MinimaxAgent')
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        def minSearch(gameState: GameState, depth, agentIndex, alpha, beta):
            if gameState.getNumFood() == 0 or gameState.isLose() or depth == self.depth:
                return heuristics(gameState)
            retValue = inf
            currBeta = beta 
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            if nextAgent == 0:
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    retValue = min(retValue, maxSearch(successor, depth + 1, nextAgent, alpha, currBeta))
                    if retValue < alpha:
                        return retValue
                    if retValue < currBeta:
                        currBeta = retValue
            else: 
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    retValue = min(retValue, minSearch(successor, depth, nextAgent, alpha, currBeta))
                    if retValue < alpha:
                        return retValue
                    if retValue < currBeta:
                        currBeta = retValue
            return retValue
        
        def maxSearch(gameState: GameState, depth, agentIndex, alpha, beta):
            if gameState.getNumFood() == 0 or gameState.isLose() or depth == self.depth:
                return heuristics(gameState)
            retValue = -inf
            currAlpha = alpha
            for action in gameState.getLegalPacmanActions():
                successor = gameState.generateSuccessor(agentIndex, action)
                if (successor.isWin()):
                    return inf
                retValue = max(retValue, minSearch(successor, depth, agentIndex + 1, currAlpha, beta))
                if retValue > beta:
                    return retValue
                if retValue < currAlpha:
                    currAlpha = retValue
            return retValue
        
        maxVal = -inf
        nextAction = "Stop"
        for action in gameState.getLegalPacmanActions():
            successor = gameState.generatePacmanSuccessor(action)
            if (successor.isWin()):
                return action
            actionVal = minSearch(successor, 0, 1, -inf, inf)
            if action != "Stop" and actionVal > maxVal:
                nextAction = action
                maxVal = actionVal
        return nextAction

def heuristics(gameState: GameState):


    pacman = gameState.getPacmanPosition()

    foodDistance = []
    foodList = gameState.getFood().asList()
    nearestFood = inf

    if len(foodList) > 20:
        reducedFoodList = random.sample(foodList, 15)
    else:
        reducedFoodList = foodList
    for food in reducedFoodList:
        dist = util.manhattanDistance(pacman, food) + 0.1
        heapq.heappush(foodDistance, (dist, len(foodDistance), dist))
    (_, _, nearestFood) = heapq.heappop(foodDistance)
        
    ghostDistance = []
    scaredGhostDistance = []
    nearestGhost = inf
    nearestScaredGhost = inf
    for i in range(1, gameState.getNumAgents()):
        dist = util.manhattanDistance(pacman, gameState.getGhostPosition(i)) + 0.1
        if gameState.getGhostState(i).scaredTimer > 0:
            heapq.heappush(scaredGhostDistance, (dist, len(scaredGhostDistance), dist))
        else:
            heapq.heappush(ghostDistance, (dist, len(ghostDistance), dist))
    if len(ghostDistance) > 0:
        (_, _, nearestGhost) = heapq.heappop(ghostDistance)
    if len(scaredGhostDistance) > 0:
        (_, _, nearestScaredGhost) = heapq.heappop(scaredGhostDistance)
    if nearestGhost > 10:
        nearestGhost = -nearestGhost

    capsuleDistance = []
    capsuleList = gameState.getCapsules()
    nearestCapsule = inf
    if len(capsuleList) > 0:
        for capsule in capsuleList:
            dist = util.manhattanDistance(pacman, capsule) + 0.1
            heapq.heappush(capsuleDistance, (dist, len(capsuleDistance), dist))
        (_, _, nearestCapsule) = heapq.heappop(capsuleDistance)


    heuristic = 9/nearestFood - 35/nearestGhost + 190/nearestScaredGhost + 30/nearestCapsule - len(foodList)*10 - len(capsuleList)*50
    return gameState.getScore() + heuristic