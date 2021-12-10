from typing import Callable
# from statistics import mean

def mean(x):
    res = 0
    for a in x:
        if a is None: return 0
        res += a
    return res/len(x)

def foodHeuristic(state) -> float:
    position = state.getPacmanPosition()
    foodGrid = state.data.food
    x1, y1 = position
    length = foodGrid.count()
    lengs = []
    for x2, y2 in foodGrid.asList():
        lengs.append(abs(x1 - x2) + abs(y1 - y2))
    if len(lengs) > 0: return sum(lengs)/length 
    return 0

def closestGhost(state) -> float:
    x1, y1 = state.getPacmanPosition()
    res = 1000
    for x2, y2 in state.getGhostPositions():
        res += abs(x1 - x2) + abs(y1 - y2)
    res /= len(state.getGhostPositions())
    return 10 / (res + 0.01)

def SEV(state) -> float:
    return 2 * state.getScore() + 100/(foodHeuristic(state)**2) - closestGhost(state)

def expectiSEV(state) -> float:
    return state.getScore() + 100/(foodHeuristic(state)**2)

class Node:
     
    def __init__(self, depth: int, gameState, parent = None):
        
        self.depth: int = depth
        self.state = gameState
        self.parent: Node = parent
        self.bestChild: Node = None

    def getChildren(self):
        gameStates = [self.state.generateSuccessor(0, action) for action in self.state.getLegalActions()]
        return [Node(self.depth - 1, gs, self) for gs in gameStates]

def minPerRow(matrix):
    return [min(x) for x in list(map(list, zip(*matrix)))]

def meanPerRow(matrix):
    return [mean(x) for x in list(map(list, zip(*matrix)))]

def expectimax(node: Node, func: Callable[[any], float]):
    if node.state.isWin(): return 1000
    if node.state.isLose(): return -1000
    if (node.depth == 0):
        return func(node.state)
    res = [0 for x in range(4)]
    for i in range(4):
        children = node.getChildren()
        res[i] = [0 for x in range(len(children))]
        for j in range(len(children)):
            res[i][j] = expectimax(children[j], func)
    resList = meanPerRow(res)
    maximum = max(resList)
    node.bestChild = node.getChildren()[resList.index(maximum)]
    return maximum

def minimax(node: Node, func):
    if node.state.isWin(): return 1000
    if node.state.isLose(): return -1000
    if (node.depth == 0):
        return func(node.state)
    res = [0 for x in range(4)]
    for i in range(4):
        children = node.getChildren()
        res[i] = [0 for x in range(len(children))]
        for j in range(len(children)):
            res[i][j] = expectimax(children[j], func)
    resList = minPerRow(res)
    maximum = max(resList)
    node.bestChild = node.getChildren()[resList.index(maximum)]
    return maximum


def abPrune(node: Node, func):
    MAX, MIN = 1000, -1000

    def alphaBeta(node: Node, func, alpha: float, beta: float):
        if node.depth == 0 or node.state.isEnded():
            return func(node.state)

        best: float = MIN
        # Recur for left and right children
        for child in node.getChildren():                
            val = alphaBeta(child, func, alpha, beta)
            if val > best:
                best = val
                node.bestChild = child                
            alpha = max(alpha, best)
            # Alpha Beta Pruning
            if beta <= alpha:
                break
        
        return best 
    return alphaBeta(node, func, MIN, MAX)
