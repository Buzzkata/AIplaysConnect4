import argparse
import os
from connect_four import ConnectFourGame
from neural_net import NeuralNetworkWrapper
from train import Train
from human_play import HumanPlay
from configuration import CFG

# Two argument variables are implemeted to be parsed as the program is run.
parser = argparse.ArgumentParser()

parser.add_argument("--load_model",
                    help="Boolean which initializes the network with the best model.",
                    dest="load_model",
                    type=int,
                    default=CFG.load_model)

parser.add_argument("--human_play",
                    help="Boolean which specifies play as Human vs the AI.",
                    dest="human_play",
                    type=int,
                    default=CFG.human_play)


if __name__ == '__main__':   
    arguments = parser.parse_args()
    CFG.load_model = arguments.load_model
    CFG.human_play = arguments.human_play 

    # Initialize the game object with the connect 4 game.
    game = object
    game = ConnectFourGame()
    net = NeuralNetworkWrapper(game)

    # Initialize the network with the best model.
    if CFG.load_model:
        file_path = CFG.model_directory + "best_model.meta"
        if os.path.exists(file_path):
            net.load_model("best_model")
        else:
            print("Trained model doesn't exist. Starting from scratch.")
    else:
        print("Trained model not loaded. Starting from scratch.")

    # Choose human play option or train (play against itself)
    if CFG.human_play:
        human_play = HumanPlay(game, net)
        human_play.play()
    else:
        train = Train(game, net)
        train.start()