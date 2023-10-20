#region Import
print('Importing...')

import pandas as pd
print('pandas import - done')
import numpy as np
print('numpy import - done')
import sympy as sp
print('sympy import - done')
#endregion

#region Constants
training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = np.array([[0, 1, 1, 0]]).T
weights = [
                [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                [[0.5, 0.5]]
          ]

x1, x2 = sp.symbols('x1 x2')
yFunc = pow(x1-x2,2)/2
epoch = 1
eta = 0.003
checker = 0.0001
#endregion

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def run_network(input_sygnals, weights):
    answer = input_sygnals
    # print(answer)
    for layer in range(len(weights)):
        answer = run_layer(answer, weights[layer])
        # print(answer)
    return answer
        
def run_layer(input_sygnals, layer_weights):
    answer = np.zeros(len(layer_weights))
    for neuron in range(len(layer_weights)):
        for i in range(len(layer_weights[neuron])):
            # print(layer_weights[neuron][i])
            # print(input_sygnals[i])
            answer[neuron] += layer_weights[neuron][i] * input_sygnals[i]
        answer[neuron] /= len(layer_weights[neuron])
        answer[neuron] = sigmoid(answer[neuron])
    return answer

def sgd(func, x1Value, x2Value):
    print(func)
    print(x1Value)
    print(x2Value)
    dx1 = sp.diff(func, x1)
    dx2 = sp.diff(func, x2)
    print(dx1)
    print(dx2)

for current_epoch in range(epoch): #for N epoch
    for i in range(training_set_inputs.size): #for every training set
        sgd(yFunc, run_network(training_set_inputs[i], weights)[0], training_set_outputs[i])

    
        

# #region Calculating derivatives
# print("Calculating derivatives...")
# dx1 = sp.diff(yFunc, x1)
# dx2 = sp.diff(yFunc, x2)
# print('Function = {0}'.format(yFunc))
# print('Derivative from this function by x1 = {0}'.format(dx1))
# print('Derivative from this function by x2 = {0}'.format(dx2))
# #endregion

# #region Searching for lowest square error value
# print('Searching for lowest square error value...')
# x1Value = initialX1
# x2Value = initialX2

# x1Diff = initialX1
# x2Diff = initialX2

# step = 0
# # while (x1Diff > checker or x2Diff > checker) and step < 100:
# while (x1Diff > checker or x2Diff > checker):
#     print('PROCESSING(step:{0}): X1={1:.5f}; X2={2:.5f}; X1D={3:.4f}; X2D={4:.4f}'.format(step,x1Value, x2Value, x1Diff, x2Diff))
#     newx1Value = x1Value - eta * dx1.evalf(subs={x1:x1Value})
#     newx2Value = x2Value - eta * dx2.evalf(subs={x2:x2Value})
#     x1Diff = sp.Abs(x1Value - newx1Value)
#     x2Diff = sp.Abs(x2Value - newx2Value)
#     x1Value = newx1Value
#     x2Value = newx2Value
#     step += 1
    
# print('FINAL(step:{0}): X1={1}; X2={2}'.format(step,x1Value,x2Value))
# #endregion