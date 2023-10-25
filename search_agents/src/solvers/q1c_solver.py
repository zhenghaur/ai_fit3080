import logging

import util
from problems.q1c_problem import q1c_problem


def q1c_solver(problem: q1c_problem):
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()  # Your code replaces this line
    
    preference = ["", "North", "South", "East", "West"]
    preferenceSolution = []
    for i in range(len(preference)):

        frontier = util.PriorityQueue()
        explored = []
        frontier.push(((problem.getStartState(), None, 0), None), 0)
        foodLeft = problem.getStartState().getNumFood()
        exploredMap = []
        walls = problem.getStartState().getWalls()
        for x in range(walls.width):
            exploredMap.append([])
            for y in range(walls.height):
                exploredMap[x].append(False)
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
                for x in range(walls.width):
                    for y in range(walls.height):
                        exploredMap[x][y] = False
                frontier = util.PriorityQueue()

            if current[0][0].getNumFood() <= foodLeft:
                foodList = current[0][0].getFood().asList()
                foodQueue = util.PriorityQueue()
                for food in foodList:
                    foodQueue.push(food, util.manhattanDistance(current[0][0].getPacmanPosition(), food))
                foodLeft = current[0][0].getNumFood()
                food = foodQueue.pop()

            successors = problem.getSuccessors(current[0][0])
            for successor in successors:
                if successor[1] == preference[i]:
                    successors.remove(successor)
                    successors.insert(0, successor)

            for successor in successors:
                (sucX, sucY) = successor[0].getPacmanPosition()
                if successor[1] != 'Stop' and not exploredMap[sucX][sucY]:
                    gscore = 0 - (successor[0].getScore() - (problem.getStartState().getNumFood() - successor[0].getNumFood()) * 10)
                    heuristic = util.manhattanDistance(successor[0].getPacmanPosition(), food)
                    if successor[1] == preference[i]:
                        heuristic -= 2
                    frontier.push((successor, current[0]), gscore + heuristic)

        solution = []
        while current[1] is not None:
            solution.append(current[0][1])
            for state in explored:
                if state[0] == current[1]:
                    current = state
        
        preferenceSolution.append(solution)

    ret = preferenceSolution[0]
    for i in range(len(preferenceSolution)):
        if len(preferenceSolution[i]) < len(ret):
            ret = preferenceSolution[i]

    return ret[::-1]


