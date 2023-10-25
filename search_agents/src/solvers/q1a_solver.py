import logging

import util
from problems.q1a_problem import q1a_problem


def q1a_solver(problem: q1a_problem):
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()  # Your code replaces this line

    frontier = util.PriorityQueue()
    explored = []
    frontier.push(((problem.getStartState(), None, 0), None), 0)
    exploredMap = []
    walls = problem.getStartState().getWalls()
    for i in range(walls.width):
        exploredMap.append([])
        for j in range(walls.height):
            exploredMap[i].append(False)
    current = None
    food = problem.getStartState().getFood().asList()[0]

    while current is None or not frontier.isEmpty() and not problem.isGoalState(current[0][0]):
        current = frontier.pop()
        (posX, posY) = current[0][0].getPacmanPosition()
        while exploredMap[posX][posY]:
            current = frontier.pop()
            (posX, posY) = current[0][0].getPacmanPosition()
        
        explored.append(current)
        exploredMap[posX][posY] = True
        for successor in problem.getSuccessors(current[0][0]):
            (sucX, sucY) = successor[0].getPacmanPosition()
            if successor[1] != 'Stop' and not exploredMap[sucX][sucY]:
                frontier.push((successor, current[0]), util.manhattanDistance((sucX, sucY), food))

    solution = []
    while current[1] is not None:
        solution.append(current[0][1])
        for state in explored:
            if state[0] == current[1]:
                current = state

    return solution[::-1]

    


