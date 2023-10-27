import sys
import os

if len(sys.argv) != 4:
    sys.exit(1)

workers_number = int(sys.argv[1])
if(workers_number <= 0):
    sys.exit(1)
    
target_ip = sys.argv[2]
target_port = sys.argv[3]

os.environ.update({
  "DMLC_ROLE": "worker", # Could be "scheduler", "worker" or "server"
  "DMLC_PS_ROOT_URI": target_ip, # IP address of a scheduler
  "DMLC_PS_ROOT_PORT": target_port, # Port of a scheduler
  "DMLC_NUM_SERVER": "1", # Number of servers in cluster
  "DMLC_NUM_WORKER": str(workers_number), # Number of workers in cluster
  "PS_VERBOSE": "2" # Debug mode
})

import mxnet as mx
import logging
import numpy as np
import time
from sklearn.model_selection import train_test_split

print('worker launched...')

logging.getLogger().setLevel(logging.DEBUG)

# lineral equation
def f(x):
  # a = 5
  # b = 2
  return 5 * x + 2

# Data
X = np.arange(100, step=0.001)
Y = f(X)

# Split data for taining and evaluation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
kv_store = mx.kv.create('dist_async')

batch_size = 1024
train_iter = mx.io.NDArrayIter(X_train, 
                               Y_train, 
                               batch_size, 
                               shuffle=True,
                               label_name='lin_reg_label')
eval_iter = mx.io.NDArrayIter(X_test, 
                              Y_test, 
                              batch_size, 
                              shuffle=False)

X = mx.sym.Variable('data')
Y = mx.symbol.Variable('lin_reg_label')
fully_connected_layer  = mx.sym.FullyConnected(data=X, name='fc1', num_hidden = 1)
lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="lro")

model = mx.mod.Module(
    symbol = lro ,
    data_names=['data'],
    label_names = ['lin_reg_label']# network structure
)
time1 = time.time()

model.fit(train_iter, eval_iter,
            optimizer_params={
                'learning_rate':0.000000002},
            num_epoch=20,
            eval_metric='mae',
            batch_end_callback
                 = mx.callback.Speedometer(batch_size, 20),
            kvstore=kv_store)

time2 = time.time()
print('training took %0.3f ms' % ((time2 - time1) * 1000.0))
