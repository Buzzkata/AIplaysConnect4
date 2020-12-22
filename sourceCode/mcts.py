import math
import numpy as np
from configuration import CFG
from copy import deepcopy

class TreeNode(object):
    #Class which represents the board state and stores valuable data related to a board state.
    def __init__(self, parent=None, action=None, psa=0.0, childProbs=[]):
        self.Nsa = 0
        self.Wsa = 0.0
        self.Qsa = 0.0
        self.Psa = psa
        self.action = action
        self.parent = parent
        self.children = []
        self.childProbs = childProbs      

    def checkLeaf(self): 
        #Checks if a TreeNode is a leaf.      
        if len(self.children) > 0:
            return True
        return False

    def selectChild(self):
        #Selects a child node based on the AlphaZero PUCT formula.
       
        cPuct = CFG.cPuct
        highestUct = 0
        highestIndex = 0

        # Select the child with the highest exploration vs exploitation value.
        for idx, child in enumerate(self.children):
            uct = child.Qsa + child.Psa * cPuct * (
                    math.sqrt(self.Nsa) / (1 + child.Nsa))
            if uct > highestUct:
                highestUct = uct
                highestIndex = idx

        return self.children[highestIndex]

    def expandNode(self, game, probVector): 
        #Function to expand the current node by adding valid moves as children.     
        self.childProbs = deepcopy(probVector)
        validMoves = game.getValidMoves(game.currentPlayer)
        for idx, move in enumerate(validMoves):
            if move[0] != 0:
                action = deepcopy(move)
                self.addChildNode(parent=self, action=action, psa=probVector[idx])

    def addChildNode(self, parent, action, psa=0.0):
        #Creates a child TreeNode and adds it to the current node.     
        childNode = TreeNode(parent=parent, action=action, psa=psa)
        self.children.append(childNode)
        return childNode

    def backProp(self, wsa, v):
        #This function updates the current nodes stats as the game is played.  
        self.Nsa += 1
        self.Wsa = wsa + v
        self.Qsa = self.Wsa / self.Nsa


class MonteCarloTreeSearch(object):
    #Monte Carlo Tree Search Algorithm.
    def __init__(self, net):
        self.root = None
        self.game = None
        self.net = net

    def search(self, game, node, temperature):
        #MCTS which loops and searches for the best move at a certain state.
        self.root = node
        self.game = game

        for i in range(CFG.num_mcts_sims):
            node = self.root
            game = self.game.clone()  # Create a fresh clone for each loop.

           
            while node.checkLeaf():
                node = node.selectChild()
                game.playAction(node.action)

            #Get the move probabilities and values from the neural network for the given state.
            probVector, v = self.net.predict(game.state)

            if node.parent is None:
                probVector = self.addDirichletNoise(game, probVector)

            validMoves = game.getValidMoves(game.currentPlayer)
            for idx, move in enumerate(validMoves):
                if move[0] == 0:
                    probVector[idx] = 0

            probVectorSum = sum(probVector)

            if probVectorSum > 0:
                probVector /= probVectorSum

            #Try expanding the current node.
            node.expandNode(game=game, probVector=probVector)

            gameOver, wsa = game.checkGameOver(game.currentPlayer)

            #Back propagate node stats up to the root node.
            while node is not None:
                wsa = -wsa
                v = -v
                node.backProp(wsa, v)
                node = node.parent

        highestNsa = 0
        highestIndex = 0

        # Select the child's move using a temperature parameter.
        for idx, child in enumerate(self.root.children):
            temperatureExponent = int(1 / temperature)

            if child.Nsa ** temperatureExponent > highestNsa:
                highestNsa = child.Nsa ** temperatureExponent
                highestIndex = idx

        return self.root.children[highestIndex]

    def addDirichletNoise(self, game, probVector):
        #Dirichlet noise probability vector.  
        dirichletInput = [CFG.dirichlet_alpha for x in range(game.action_size)]

        dirichletList = np.random.dirichlet(dirichletInput)
        noiseProbVector = []

        for idx, psa in enumerate(probVector):
            noiseProbVector.append(
                (1 - CFG.epsilon) * psa + CFG.epsilon * dirichletList[idx])

        return noiseProbVector