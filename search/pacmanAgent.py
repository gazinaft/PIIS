from Node import Node, SEV, abPrune, expectimax, expectiSEV, minimax
from game import Agent

MINIMAX_DEPTH = 2
EXPECTIMAX_DEPTH = 2

class MiniMaxAgent (Agent):
    def getAction( self, state ):
        root = Node(MINIMAX_DEPTH, state.deepCopy())
        minimax(root, SEV)
        return root.bestChild.state.stepToGet
        
class ExpectimaxAgent (Agent):

    def getAction(self, state):
        root = Node(EXPECTIMAX_DEPTH, state.deepCopy())
        expectimax(root, expectiSEV)
        return root.bestChild.state.stepToGet