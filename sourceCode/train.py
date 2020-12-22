import numpy as np
from configuration import CFG
from mcts import MonteCarloTreeSearch, TreeNode
from neural_net import NeuralNetworkWrapper
from evaluate import Evaluate
from copy import deepcopy

class Train(object):
    #Class which trains the Neural Network using Monte Carlo Tree Search. 

    def __init__(self, game, net):
        self.game = game
        self.net = net
        self.evalNet = NeuralNetworkWrapper(game)

    def start(self):
        for i in range(CFG.num_iterations):
            print("Iteration", i + 1)

            trainingData = [] 

            for j in range(CFG.numGames):
                print("Start Training Self-Play Game", j + 1)
                game = self.game.clone()  # Fresh clone for each new game.
                self.playGame(game, trainingData)

       
            self.net.save_model()        
            self.evalNet.load_model()       
            self.net.train(trainingData)

            # Initialize MCTS objects for the evaluation and self play neural nets.
            currentMcts = MonteCarloTreeSearch(self.net)
            evalMcts = MonteCarloTreeSearch(self.evalNet)

            evaluator = Evaluate(currentMcts=currentMcts, evalMcts=evalMcts, game=self.game)
            wins, losses = evaluator.evaluate()

            print("wins:", wins)
            print("losses:", losses)

            numGames = wins + losses

            if numGames == 0:
                winRate = 0
            else:
                winRate = wins / numGames

            print("win rate:", winRate)

            if winRate > CFG.eval_win_rate:
                # Save current model as the best model.
                print("New model saved as best model.")
                self.net.save_model("best_model")
            else:
                print("New model discarded and previous model loaded.")
                # Discard current model and use previous best model.
                self.net.load_model()

    def playGame(self, game, trainingData):
        #Loop for each self-play game.
       
        mcts = MonteCarloTreeSearch(self.net)

        gameOver = False
        value = 0
        selfPlayData = []
        count = 0

        node = TreeNode()

        # Loop until the game is over.
        while not gameOver:
            # MCTS simulations which gets the best child node.
            if count < CFG.tempThresh:
                bestChild = mcts.search(game, node, CFG.tempInit)
            else:
                bestChild = mcts.search(game, node, CFG.tempFinal)

            # Store the state, prob and v for training.
            selfPlayData.append([deepcopy(game.state),deepcopy(bestChild.parent.childProb),0])

            action = bestChild.action
            game.playAction(action)  
            count += 1

            gameOver, value = game.checkGameOver(game.currentPlayer)

            bestChild.parent = None
            node = bestChild  # Make the child node the root node.

        # Update value 'v' as the value of the game result.
        for gameState in selfPlayData:
            value = -value
            gameState[2] = value
            self.augmentData(gameState, trainingData, game.row, game.column)

    def augmentData(self, gameState, trainingData, row, column):
        state = deepcopy(gameState[0])
        probVector = deepcopy(gameState[1])

        if CFG.game == 2 or CFG.game == 1:
            trainingData.append([state, probVector, gameState[2]])
        else:
            probVector = np.reshape(probVector, (row, column))

            # Augment our data by rotating and flipping the state of the game.
            for i in range(4):
                trainingData.append([np.rot90(state, i), np.rot90(probVector, i).flatten(), gameState[2]])

                trainingData.append([np.fliplr(np.rot90(state, i)), np.fliplr(np.rot90(probVector, i)).flatten(),gameState[2]])