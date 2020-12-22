#Base Connect 4 Game Class.
class Game(object):

    def __init__(self):
        #Initialize Game with initial state.
        pass

    def clone(self):
        #Clone function which deep clones the game objectCreates a deep clone of the game object.   
        pass

    def playAction(self, action):
        #Plays an action (move) 
        pass

    def getValidMoves(self, currentPlayer):
        #Function that returns all of the valid moves.Returns a list of moves along with their validity.
        pass

    def checkGameOver(self, currentPlayer):
        #Function that determines if the game is over.
        pass

    def printBoard(self):
        #Function to print the board.
        pass