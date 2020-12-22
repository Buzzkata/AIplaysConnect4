#Configuration file
class CFG(object): # change class name to configuration
   
    numIterations = 4 #Number of iterations
    numGames = 30   #Number of self play games
    numMctsSimulations = 30  #Number of Monte Carlo Tree Search simulations
    cPuct = 1 #Exploration level for the MCTS
    lvlTwoVal = 0.0001 #L2 weight regularization for training.
    momentum = 0.9 #Momentum optimizer variable.
    learningRate = 0.01  #Learning rate for the momentum optimizer.
    tPolicyVal = 0.0001  #Policy prediction value
    tempInitial = 1  #Initial temperature variable for exploration control.
    tempFinal = 0.001  #Final temperature variable for exploration control.
    tempThresh = 10  #Threshold where initial temperature changes to final.
    epochs = 10  #Number of epochs for training.
    batch_size = 128  #Batch size for training.
    dirichletAlpha = 0.5  #Alpha value for dirichlet noise.
    epsilon = 0.25  #Value for calculating dirichlet noise.
    model_directory = "C:\\Users\\Darko\\Desktop"  #This directory is where the python modules reside and is where the program should be run.
    numEvalGames = 12  #Number of self play games used for evaluation.
    evalWinRate = 0.55  #Represents the value needed in order to store model as our best model.
    load_model = 1  #Variable to initialize the neural network with the best model.
    human_play = 0  #Variable which indicates human play against the AI or not.
    resnet_blocks = 5  #Number of residual blocks in the resnet.
    recordLoss = 1  #Variable to store policy and value loss.
    loss_file = "loss.txt"  #Name of file which records the loss.
    