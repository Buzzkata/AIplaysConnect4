from copy import deepcopy
import numpy as np
from game import Game

class ConnectFourGame(Game):
#Class that represents the Connect 4 game.
    def __init__(self):
        super().__init__()
        self.row = 6
        self.column = 7
        self.connect = 4
        self.currentPlayer = 1 
        self.state = []
        self.actionSize = self.row * self.column  

        # n x n matrix which represents the board.
        for i in range(self.row):
            self.state.append([0 * j for j in range(self.column)])

        self.state = np.array(self.state)

        self.directions = {
            0: (-1, -1),
            1: (-1, 0),
            2: (-1, 1),
            3: (0, -1),
            4: (0, 1),
            5: (1, -1),
            6: (1, 0),
            7: (1, 1)
        }

    def clone(self):
        #Function to deep copy the game state.
        gameClone = ConnectFourGame()
        gameClone.state = deepcopy(self.state)
        gameClone.currentPlayer = self.currentPlayer
        return gameClone

    def playAction(self, action): 
        #Play a move function.
        x = action[1]
        y = action[2]

        self.state[x][y] = self.currentPlayer
        self.currentPlayer = -self.currentPlayer

    def getValidMoves(self, currentPlayer): 
       #Returns the valid moves available.
        validMoves = []

        for x in range(self.row):
            for y in range(self.column):
                if self.state[x][y] == 0:
                    if x + 1 == self.row:
                        validMoves.append((1, x, y))
                    elif x + 1 < self.row:
                        if self.state[x + 1][y] != 0:
                            validMoves.append((1, x, y))
                        else:
                            validMoves.append((0, None, None))
                else:
                    validMoves.append((0, None, None))

        return np.array(validMoves)

    def checkGameOver(self, currentPlayer): 
        #Checks to see if the game is over.

        playerA = currentPlayer
        playerB = -currentPlayer

        for x in range(self.row):
            for y in range(self.column):
                playerA_count = 0
                playerB_count = 0

                # Search for player a.
                if self.state[x][y] == playerA:
                    playerA_count += 1

                    # Search in all of the 8 directions for a similar piece.
                    for i in range(len(self.directions)):
                        d = self.directions[i]

                        r = x + d[0]
                        c = y + d[1]

                        if r < self.row and c < self.column:
                            count = 1

                            # Keep searching for a connect to be found.
                            while True:
                                r = x + d[0] * count
                                c = y + d[1] * count

                                count += 1

                                if 0 <= r < self.row and 0 <= c < self.column:
                                    if self.state[r][c] == playerA:
                                        playerA_count += 1
                                    else:
                                        break
                                else:
                                    break

                        if playerA_count >= self.connect:
                            return True, 1

                        playerA_count = 1

                # Search for player b.
                if self.state[x][y] == playerB:
                    playerB_count += 1

                    # Search in all of the 8 directions for a similar piece.
                    for i in range(len(self.directions)):
                        d = self.directions[i]

                        r = x + d[0]
                        c = y + d[1]

                        if r < self.row and c < self.column:
                            count = 1

                            # Keep searching for a connect to be found.
                            while True:
                                r = x + d[0] * count
                                c = y + d[1] * count

                                count += 1

                                if 0 <= r < self.row and 0 <= c < self.column:
                                    if self.state[r][c] == playerB:
                                        playerB_count += 1
                                    else:
                                        break
                                else:
                                    break

                        if playerB_count >= self.connect:
                            return True, -1

                        playerB_count = 1

        #Game is not over if there are still moves left.
        validMoves = self.getValidMoves(currentPlayer)

        for move in validMoves:
            if move[0] == 1:
                return False, 0

        #In case there are no moves left, the game is over which results in a draw.
        return True, 0

    def printBoard(self): 
        #Print board state.
        print("   0    1    2    3    4    5    6")
        for x in range(self.row):
            print(x, end='')
            for y in range(self.column):
                if self.state[x][y] == 0:
                    print('  -  ', end='')
                elif self.state[x][y] == 1:
                    print('  X  ', end='')
                elif self.state[x][y] == -1:
                    print('  O  ', end='')
            print('\n')
        print('\n')