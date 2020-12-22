from configuration import CFG
from mcts import TreeNode

class Evaluate(object):
    #This class evaluates the neural network.

    def __init__(self, currentMcts, evaluationMcts, game):
        self.currentMcts = currentMcts
        self.evaluationMcts = evaluationMcts
        self.game = game

    def evaluate(self):      
        losses = 0
        wins = 0
        draws = 0
        for i in range(configuration.num_eval_games):
            print("Start The Evaluation Self-Play Game:", i, "\n")
            game = self.game.clone()  
            gameOver = False
            node = TreeNode()
            value = 0
            player = game.currentPlayer            
            while not gameOver:
                if game.currentPlayer == 1:
                    bestChild = self.currentMcts.search(game, node, configuration.finalTemp)
                else:
                    bestChild = self.evaluationMcts.search(game, node, configuration.finalTemp)
                action = bestChild.action
                game.playAction(action) 

                game.printBoard()
                gameOver, value = game.gameOverCheck(player)

                bestChild.parent = None
                node = bestChild  

            if value == 1:
                print("Win")
                wins += 1
            elif value == -1:
                print("Loss")
                losses += 1
            else:
                print("Draw")
                draws += 1
            print("\n")

        return wins, losses, draws  #Return the results of the game.