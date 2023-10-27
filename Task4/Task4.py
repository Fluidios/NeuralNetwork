import sys

user_input = input("Enter 'l' to launch localy: ")
if user_input.lower() == "l":
    exec(open('launch_n_workers_localy.py').read())
else:
    print('Distributed process is not handled')