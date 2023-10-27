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
  "DMLC_ROLE": "scheduler", # Could be "scheduler", "worker" or "server"
  "DMLC_PS_ROOT_URI": target_ip, # IP address of a scheduler
  "DMLC_PS_ROOT_PORT": target_port, # Port of a scheduler
  "DMLC_NUM_SERVER": "1", # Number of servers in cluster
  "DMLC_NUM_WORKER": str(workers_number), # Number of workers in cluster
  "PS_VERBOSE": "2" # Debug mode {0,1,2}
})

import mxnet as mx

print('scheduler launched...')
