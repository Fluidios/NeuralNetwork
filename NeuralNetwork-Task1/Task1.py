print('Starting...')

# This is an example of training one layer neural network 10 000 times
# Network layer contain 3 neurons

# importing libs (dot is for matrix multiplication)
import time
from numpy import exp, array, random, dot

# declaring learning set of 4 variants
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])

# declaring desired answers for learning set. array.T - matrix transpose operation
training_set_outputs = array([[0, 1, 1, 0]]).T

# setting seed for randomness generator
random.seed(1)

# generate random weights matrix[3,1] in range[-1,1]. 
# will always be same, since we set the same seed for random
synaptic_weights = 2 * random.random((3, 1)) - 1
print('Initial synaptyc_weights = {0}'.format(synaptic_weights))


# iterate form 0 to 10000
for iteration in range(10000):
    # sigmoid func where x=inputs*weights
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    # changing weights by adding the wieight_delta = error * sigmoid_derivative
    # 'training_set_outputs - output' - error
    # 'output * (1 - output)' - derivative of sigmoid func
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
    #print('Iteration({0}):\n output:\n {1}\n synaptic_weights:\n{2}'.format(iteration, output, synaptic_weights))

# test network by running input which is not been presented in learning set: [1,0,0]
# sigmoid function used as activation function
print (1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))))