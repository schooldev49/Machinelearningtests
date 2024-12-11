import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Cost function - for neural network to correct itself


class NeuralNetworkTest:
    def __init__(self):
        print("initialized")
    def costFunction(self, observed, expected):
        cost = 0
        for i in range(len(observed)):
            cost += ((observed[i]-expected[i])**2)
        return cost
    def convertToRealActivation(self, weights, biases):
        for i in range(observed):
            print(i)
    def gradientDescent(self, step):
        # Take the negative gradient of the cost function
        print("hi")
    



weights = np.random.rand(1,5)
biases = np.random.rand(1,5)

activations = np.ones(5)

activationtest = np.zeros(5)


print(weights)
print(biases)
print(activations)
print(activationtest)

hi = NeuralNetworkTest()
h = hi.costFunction([3,2,1,2],[3,5,4,3])
print(h)
