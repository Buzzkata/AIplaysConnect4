import os
import tensorflow.compat.v1 as tf  # Initialize tensorflow as version 1 in order to prevent error
tf.disable_v2_behavior()   # Here we disable version 2 behaviour to prevent the error. Tensor versions 2 and above cause errors for our program.
import numpy as np
from configuration import CFG

class NeuralNetwork(object):
# Sets up neural network with the residual neural network graph. 
# Input layer, convolutional block, residual block, policy and value head and loss function instantiated.
    def __init__(self, game):   
        # Initialize Neural Net with the Resnet.
        self.row = game.row
        self.column = game.column
        self.actionSize = game.actionSize
        self.pi = None
        self.v = None

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.states = tf.placeholder(tf.float32,shape=[None, self.row, self.column])
            self.training = tf.placeholder(tf.bool)

            # Input Layer
            inputLayer = tf.reshape(self.states,[-1, self.row, self.column, 1])

            # Convolutional Block
            convBlock = tf.layers.conv2d(
                inputs=inputLayer,
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                strides=1)

            batchNorm = tf.layers.batch_normalization(
                inputs=convBlock,
                training=self.training)

            relu1 = tf.nn.relu(batchNorm)

            resnetInOut = relu1

            # Residual Tower
            for i in range(CFG.resnet_blocks):
                # Res Block
                convBlock2 = tf.layers.conv2d(
                    inputs=resnetInOut,
                    filters=256,
                    kernel_size=[3, 3],
                    padding="same",
                    strides=1)

                batchNormalization2 = tf.layers.batch_normalization(
                    inputs=convBlock2,
                    training=self.training)

                relu2 = tf.nn.relu(batchNormalization2)

                convBlock3 = tf.layers.conv2d(
                    inputs=relu2,
                    filters=256,
                    kernel_size=[3, 3],
                    padding="same",
                    strides=1)

                batchNormalization3 = tf.layers.batch_normalization(
                    inputs=convBlock3,
                    training=self.training)

                resnetSkip = tf.add(batchNormalization3, resnetInOut)
                resnetInOut = tf.nn.relu(resnetSkip)

            convBlock4 = tf.layers.conv2d(
                inputs=resnetInOut,
                filters=2,
                kernel_size=[1, 1],
                padding="same",
                strides=1)

            batchNormalization4 = tf.layers.batch_normalization(
                inputs=convBlock4,
                training=self.training)

            relu4 = tf.nn.relu(batchNormalization4)
            relu4Flat = tf.reshape(relu4, [-1, self.row * self.column * 2])
            logits = tf.layers.dense(inputs=relu4Flat, units=self.actionSize)
            self.pi = tf.nn.softmax(logits)

            convBlock5 = tf.layers.conv2d(
                inputs=resnetInOut,
                filters=1,
                kernel_size=[1, 1],
                padding="same",
                strides=1)

            batchNormalization5 = tf.layers.batch_normalization(
                inputs=convBlock5,
                training=self.training)

            relu5 = tf.nn.relu(batchNormalization5)
            relu5Flat = tf.reshape(relu5, [-1, self.actionSize])
            dense1 = tf.layers.dense(inputs=relu5Flat,units=256)

            relu6 = tf.nn.relu(dense1)
            dense2 = tf.layers.dense(inputs=relu6,units=1)
            self.v = tf.nn.tanh(dense2)

            # Loss Method
            self.trainPis = tf.placeholder(tf.float32, shape=[None, self.actionSize])
            self.trainVs = tf.placeholder(tf.float32, shape=[None])
            self.lossPi = tf.losses.softmax_cross_entropy(self.trainPis, self.pi)
            self.lossV = tf.losses.mean_squared_error(self.trainVs, tf.reshape(self.v, shape=[-1, ]))
            self.total_loss = self.lossPi + self.lossV
         
            optimizer = tf.train.MomentumOptimizer(
                learningRate=CFG.learningRate,
                momentum=CFG.momentum,
                use_nesterov=False)

            self.train_op = optimizer.minimize(self.total_loss)

            # Saver for training checkpoints.
            self.saver = tf.train.Saver()

            # Session for running Ops on the Graph.
            self.sess = tf.Session()

            # Initialize session.
            self.sess.run(tf.global_variables_initializer())


class NeuralNetworkWrapper(object):
    def __init__(self, game):
        self.game = game
        self.net = NeuralNetwork(self.game)
        self.sess = self.net.sess

    def predict(self, state):
        #Predicts move probabilities and values given a certain game state.

        state = state[np.newaxis, :, :]

        pi, v = self.sess.run([self.net.pi, self.net.v],feedDictionary={self.net.states: state,self.net.training: False})

        return pi[0], v[0][0]

    def train(self, trainingData):  
        print("\nTraining the network.\n")

        for epoch in range(CFG.epochs):
            print("Epoch", epoch + 1)

            numExamples = len(trainingData)

            # Separate epochs into batches.
            for i in range(0, numExamples, CFG.batch_size):
                states, pis, vs = map(list, zip(*trainingData[i:i + CFG.batch_size]))

                feedDictionary = {self.net.states: states,
                             self.net.trainPis: pis,
                             self.net.trainVs: vs,
                             self.net.training: True}

                self.sess.run(self.net.train_op,feedDictionary=feedDictionary)

                piLoss, vLoss = self.sess.run(
                    [self.net.lossPi, self.net.lossV],
                    feedDictionary=feedDictionary)

                # Save pi and v loss in a file.
                if CFG.recordLoss:
                    # Create directory if it doesn't exist.
                    if not os.path.exists(CFG.model_directory):
                        os.mkdir(CFG.model_directory)

                    file_path = CFG.model_directory + CFG.loss_file

                    with open(file_path, 'a') as loss_file:
                        loss_file.write('%f|%f\n' % (piLoss, vLoss))

        print("\n")

    def save_model(self, filename="current_model"):
        #Saves the neural network model at a specified file path.    
        # Create directory if it doesn't exist or none specified.
        if not os.path.exists(CFG.model_directory):
            os.mkdir(CFG.model_directory)

        file_path = CFG.model_directory + filename

        print("Saving model:", filename, "at", CFG.model_directory)
        self.net.saver.save(self.sess, file_path)

    def load_model(self, filename="current_model"):
        #Loads the neural network model from a specified file path.
        file_path = CFG.model_directory + filename

        print("Loading model:", filename, "from", CFG.model_directory)
        self.net.saver.restore(self.sess, file_path)