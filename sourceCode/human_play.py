from mcts import MonteCarloTreeSearch, TreeNode
from configuration import CFG

class HumanPlay(object):
    #Class that allows human vs AI play
    def __init__(self, game, net):
        self.game = game
        self.net = net

    def play(self):
        #Function which allows human play vs AI.
        print("Start Human vs AI\n")

        mcts = MonteCarloTreeSearch(self.net)
        game = self.game.clone()  # Create a fresh clone for each game.
        gameOver = False
        value = 0
        node = TreeNode()

        print("Please enter a move in the following form: row, column.")
        startFirst = input("Would you like to start first: y/n?")

        if startFirst.lower().strip() == 'y':
            print("You play as X")
            humanValue = 1

            game.printBoard()
        else:
            print("You play as O")
            humanValue = -1

        # Keep playing until the game is over.
        while not gameOver:
            if game.currentPlayer == humanValue:
                action = input("Enter your move: ")
                if isinstance(action, str):
                    action = [int(n, 10) for n in action.split(",")]
                    action = (1, action[0], action[1])

                bestChild = TreeNode()
                bestChild.action = action
            else:
                bestChild = mcts.search(game, node, CFG.tempFinal)

            action = bestChild.action
            game.playAction(action)  

            game.printBoard()
            gameOver, value = game.checkGameOver(game.currentPlayer)

            bestChild.parent = None
            node = bestChild  #The child node is the root node.

        if value == humanValue * game.currentPlayer:
            print("You have won.")
        elif value == -humanValue * game.currentPlayer:
            print("You have lost.")
        else:
            print("Draw game.")
        print("\n")