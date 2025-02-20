import torch
import sys
import json
import random, string
from typing import List, Tuple
import math
import bisect
from bisect import *
from collections import *
import heapq
def gen1(num_cases: int):
    # rationale: generate random GPAs between 0.0 and 4.0 with one decimal place
    data = []
    for _ in range(num_cases):
        N = random.randint(1, 10)  # number of GPAs in the list
        grades = [round(random.uniform(0.0, 4.0), 1) for _ in range(N)]
        data.append({'grades': grades})
    return data



def gen2(num_cases: int):
    # rationale: generate GPAs that are close to the decision boundaries
    data = []
    boundaries = [4.0, 3.7, 3.3, 3.0, 2.7, 2.3, 2.0, 1.7, 1.3, 1.0, 0.7, 0.0]
    for _ in range(num_cases):
        N = random.randint(5, 15)  # longer list of GPAs
        grades = [round(random.choice(boundaries) + random.uniform(-0.05, 0.05), 1) for _ in range(N)]
        data.append({'grades': grades})
    return data



def gen3(num_cases: int):
    # rationale: generate GPAs that are all the same or very close to each other
    data = []
    for _ in range(num_cases):
        N = random.randint(1, 10)
        base_gpa = round(random.uniform(0.0, 4.0), 1)
        grades = [round(base_gpa + random.uniform(-0.01, 0.01), 1) for _ in range(N)]
        data.append({'grades': grades})
    return data



res = gen2(10)
print('Data generation and saving complete!')
torch.save(res,'../data_genshot/shots_81_testset.pt')