import logging

import util
from problems.q1b_problem import q1b_problem


def q1b_solver(problem: q1b_problem):
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()  # Your code replaces this line
    
    frontier = util.PriorityQueue()
    explored = []
    frontier.push(((problem.getStartState(), None, 0), None), 0)
    foodLeft = problem.getStartState().getNumFood()
    exploredMap = []
    walls = problem.getStartState().getWalls()
    for i in range(walls.width):
        exploredMap.append([])
        for j in range(walls.height):
            exploredMap[i].append(False)
    current = None

    while current is None or not problem.isGoalState(current[0][0]):

        current = frontier.pop()
        (posX, posY) = current[0][0].getPacmanPosition()
        while exploredMap[posX][posY]:
            current = frontier.pop()
            (posX, posY) = current[0][0].getPacmanPosition()
            
        explored.append(current)
        exploredMap[posX][posY] = True

        if current[0][0].getNumFood() == 0:
            break
        
        if current[0][0].getNumFood() < foodLeft:
            for i in range(walls.width):
                for j in range(walls.height):
                    exploredMap[i][j] = False
            frontier = util.PriorityQueue()

        if current[0][0].getNumFood() <= foodLeft:
            foodList = current[0][0].getFood().asList()
            foodQueue = util.PriorityQueue()
            for food in foodList:
                foodQueue.push(food, util.manhattanDistance(current[0][0].getPacmanPosition(), food))
            foodLeft = current[0][0].getNumFood()
            food = foodQueue.pop()

        for successor in problem.getSuccessors(current[0][0]):
            (sucX, sucY) = successor[0].getPacmanPosition()
            if successor[1] != 'Stop' and not exploredMap[sucX][sucY]:
                frontier.push((successor, current[0]), util.manhattanDistance(successor[0].getPacmanPosition(), food))

    solution = []
    while current[1] is not None:
        solution.append(current[0][1])
        for state in explored:
            if state[0] == current[1]:
                current = state
    
    return solution[::-1]


