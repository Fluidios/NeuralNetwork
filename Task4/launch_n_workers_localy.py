import sys
import subprocess

workers_count = input("Enter workers count: ")
target_ip = "127.0.0.1"
target_port = "9000"

subprocess.call(["python", "scheduler.py", workers_count, target_ip, target_port])

subprocess.call(["python", "server.py", workers_count, target_ip, target_port])

for x in range(int(workers_count)):
    subprocess.call(["python", "worker.py", workers_count, target_ip, target_port])
