#region Import
print('Importing...')

import pandas as pd
print('pandas import - done')
import sympy as sp
print('sympy import - done')
#endregion

#region Constants
x1, x2 = sp.symbols('x1 x2')
yFunc = pow(x1,2) - 10*x1 + 2*pow(x2,2) - 20*x2 + 75
initialX1 = 10
initialX2 = 10
eta = 0.003
checker = 0.0001
#endregion

#region Calculating derivatives
print("Calculating derivatives...")
dx1 = sp.diff(yFunc, x1)
dx2 = sp.diff(yFunc, x2)
print('Function = {0}'.format(yFunc))
print('Derivative from this function by x1 = {0}'.format(dx1))
print('Derivative from this function by x2 = {0}'.format(dx2))
#endregion

#region Searching for lowest square error value
print('Searching for lowest square error value...')
x1Value = initialX1
x2Value = initialX2

x1Diff = initialX1
x2Diff = initialX2

step = 0
# while (x1Diff > checker or x2Diff > checker) and step < 100:
while (x1Diff > checker or x2Diff > checker):
    print('PROCESSING(step:{0}): X1={1:.5f}; X2={2:.5f}; X1D={3:.4f}; X2D={4:.4f}'.format(step,x1Value, x2Value, x1Diff, x2Diff))
    newx1Value = x1Value - eta * dx1.evalf(subs={x1:x1Value})
    newx2Value = x2Value - eta * dx2.evalf(subs={x2:x2Value})
    x1Diff = sp.Abs(x1Value - newx1Value)
    x2Diff = sp.Abs(x2Value - newx2Value)
    x1Value = newx1Value
    x2Value = newx2Value
    step += 1
    
print('FINAL(step:{0}): X1={1}; X2={2}'.format(step,x1Value,x2Value))
#endregion
