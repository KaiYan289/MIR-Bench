import multiprocessing
import time
import random, string
from typing import List, Tuple
import math
import bisect
from bisect import *
from collections import *
import heapq

# Function to execute code using exec
def exec_with_timeout(code, return_dict):
    try:
        # Using exec() to execute the code
        local_vars = {}
        global_vars = {'bisect_left': bisect_left, 'bisect_right': bisect_right, 'heapq': heapq, 'random': random, 'string': string, 'List': List, 'Tuple': Tuple, 'math': math, 'bisect': bisect, 'Counter': Counter}
        exec(code, global_vars, local_vars)
        return_dict['timeout'] = False  # Marking as success
        return_dict['result'] = local_vars['res']
    except Exception as e:
        return_dict['error'] = str(e)  # Capturing any exception during execution
        return_dict['result'] = None
    # print("return dict:", return_dict)

def run_with_timeout(code, timeout_duration):
    # Manager to hold shared data
    manager = multiprocessing.Manager()
    return_dict = manager.dict()  # Dictionary to store results across processes
    return_dict['timeout'] = True  # Default value, will be set to False if exec completes
    return_dict['result'] = None
    # Create a separate process to run the exec() call
    process = multiprocessing.Process(target=exec_with_timeout, args=(code, return_dict))

    # Start the process
    process.start()

    # Wait for the process to complete or timeout
    process.join(timeout_duration)

    # If process is still active after timeout, terminate it
    if process.is_alive():
        process.terminate()
        process.join()  # Ensuring process is completely cleaned up
    if return_dict['result'] is None: print("return dict:", return_dict)
    return return_dict