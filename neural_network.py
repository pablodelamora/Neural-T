# Back-Propagation Neural Networks
#
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>

import math
import random
import string
import sys
import json
import datetime
import argparse

def nowAsString():
    retval = "{:%y%m%d%H%M%S}".format(datetime.datetime.today())
    retval += ".nn.json" #Filename extension
    return retval

random.seed(0)
scripting = False

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh + 1 # +1 for bias node
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        for p in patterns:
            print((p[0], '->', self.update(p[0])))

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print((self.wi[i]))
        print()
        print('Output weights:')
        for j in range(self.nh):
            print((self.wo[j]))

    def train(self, patterns, iterations, N):
        global scripting
        # N: learning rate
        # M: momentum factor
        M = N/5
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % (iterations/10) == 0:
                if not scripting:
                    print(('error %-.5f' % error))
        if not scripting:
            print(('error %-.5f' % error))

    def save(self):
        global scripting
        filename = nowAsString()
        with open(filename, "w") as text_file:
            text_file.write(json.dumps({
                'wo': self.wo,
                'wi': self.wi
            }, indent=2))
        if scripting:
            print(filename)

    @staticmethod
    def load(filename):
        content = "";
        with open(filename, 'r') as content_file:
            content = content_file.read()
        source = json.loads(content)

        self = NN(0, 0, 0);

        # number of input, hidden, and output nodes
        self.ni = len(source["wi"])
        self.nh = len(source["wo"])
        self.no = len(source["wo"][0])

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # create weights
        self.wi = source["wi"]
        self.wo = source["wo"]

        # last change in weights for momentum
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)
        return self

class Player:
    def __init__(self, filename):
        self.mynet = NN.load(filename)

    def play(self, turn, board):
        width = len(board[0])
        height = len(board)
        nnInput = []
        for row in range(height - 1, -1, -1):
            for col in range(0, width):
                nnInput.append(board[row][col])

        nnInput.append(turn)

        outputActivations = self.mynet.update(nnInput)
        bestOption = outputActivations.index(max(outputActivations))
        while board[height - 1][bestOption] != 0:
            outputActivations[bestOption] = -1
            bestOption = outputActivations.index(max(outputActivations))
        return bestOption

def demo():
    global scripting
    parser = argparse.ArgumentParser(description='Train neural network.')
    #parser.add_argument('-s', action="store_true", help='save neural network to file.')
    parser.add_argument('--scripting', action="store_true", default=False, help='Use for scripting purposes; replace stdout output with filename holding the resulting nn.')
    parser.add_argument('-l', action="store", nargs=1, type=str, metavar="file", help='Train an existing neural network defined in file.')
    parser.add_argument('-r', action="store", nargs=1, type=float, metavar="rate", help='Learning rate. [default: 0.000001]', default=[0.000001])
    parser.add_argument('-i', action="store", nargs=1, type=int, metavar="iterate", help='Number of iterations for each sample. [default: 100]', default=[100])
    parser.add_argument('samples', action="store", nargs=1, type=str, help='Samples for training.')
    args = parser.parse_args()
    scripting = args.scripting
    content = ""
    with open(args.samples[0], 'r') as content_file:
        content = content_file.read()
    samples = eval(content)
    nn = {}
    if args.l != None:
        content = "";
        nn = NN.load(args.l[0])
    else:
        nn = NN(43, 10, 7)
        content = "";
    nn.train(samples, args.i[0], args.r[0])
    nn.save()

if __name__ == '__main__':
    demo()
