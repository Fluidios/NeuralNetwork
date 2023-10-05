#region Importing
print('Importing...')

import pandas as pd
print('pandas import - done')
import re # strings work
print('re import - done')
#endregion

#region Constants
SAVED_WEIGHTS = 'saved_weights.txt'
#endregion

#region Choose behaviour
user_input = input("Enter 'y' to load NN 'n' to start training: ")
if user_input.lower() == "y":
    exec(open('tester.py').read())
else:
    print('Starting learning process...')
    exec(open('learning.py').read())
#endregion

print("END")