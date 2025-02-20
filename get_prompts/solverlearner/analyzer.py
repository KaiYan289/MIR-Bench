import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import re
import multiprocessing
import num2words
import inflect
import ast
import time
import signal
import random, string
from typing import List, Tuple
import math
import bisect
from tqdm import tqdm
from fractions import Fraction
from collections import *
from bisect import *
from itertools import *
import heapq
from util import *
"""
This is the final step for running solverlearner experiment in our paper.
You should run build_dataset_variants_partC.py and invoke LLMs to generate output first.
It reads output from LLMs from ./res folder and generate runnable scripts for evaluation in ./code folder.
Use ./code/run.sh to run them.
"""
def evaluate():
    # to get this file, run 
    original_data = pd.read_parquet("data/build_data_code_right_random_64acc-quad-formal.parquet")
    # print("original_data:", original_data.keys())
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    # print(original_data.keys())
    prompts_idx = {}
    for i in range(len(original_data['prompt'])):
        prompts_idx[original_data['prompt'][i]] = i
    
    def judge_ans(inp, prompt, gt, answers, info):
        N, r, error = len(answers), 0, 0
        os.makedirs(os.path.dirname("code/"+str(info[0])+"/"+str(info[1])+"/"), exist_ok=True)
        print("filename:" ,os.path.dirname("code/"+str(info[0])+"/"+str(info[1])))
        for j, output in enumerate(answers):
            last_output = output.replace("code:", '').replace("Code:", '').replace('```python', "").replace("```", "").lstrip().rstrip()
            ans = gt.lstrip().rstrip()
            # print("ans:", ans)
            try:
                def_idx = last_output.find("def ")
                func_name = last_output[def_idx:].split("(")[0].split("def ")[1]
            except: # this is for more than 1 output (in the name of predict1, predict2, etc.) Have no effect if the output only contains predict0.
                error += 1
                continue
            local_vars = {}
            global_vars = {'Fraction': Fraction, 'inflect': inflect, 'num2words': num2words, 'np': np, 're': re, 'combinations': combinations, 'bisect_left': bisect_left, 'bisect_right': bisect_right, 'heapq': heapq, 'random': random, 'string': string, 'List': List, 'Tuple': Tuple, 'math': math, 'bisect': bisect, 'Counter': Counter}
            code =  last_output + "\nprint('ans:[<-[',  "+ func_name + "(" + str(inp) + "), ']->]')" # add special characters for answer extraction
            # runnable scripts will be generated into the ./code folder for running. Again, use ./code/run.sh to automatically run everything in the folder.
            f = open("code/"+str(info[0])+"/"+str(info[1])+"/"+str(j)+".py", "w")
            f.write(code)
            f.close()
            return

    def eval_single(name, threshold_shot=4096):
        res = pd.read_json(name)
        print(res.keys())
        tot_by_ER, score_by_ER = {}, {}
        tot_by_shots, score_by_shots = {}, {}
        unrunnable_by_ER, unrunnable_by_shots = {}, {}
        for i in tqdm(range(len(res['prompt']))):
            idx_in_original = prompts_idx[res['prompt'][i]]
            # extract code
            error_rate, plen, num_shots = original_data['ER_rate'][idx_in_original], original_data['plen'][idx_in_original], original_data['num_shots'][idx_in_original]
            train_idx = original_data['idx_train'][idx_in_original]
            inp_left = res['prompt'][i].rfind("Input:")
            inp = res['prompt'][i][inp_left+6:].split("\n")[0]

            if num_shots >= threshold_shot: continue

            JA = judge_ans(inp, res['prompt'][i], res['answer'][i][0], [res['predict'+str(j)][i] for j in range(1)], (name.split("/")[1], i))

    

    """
    enumerating output files for evaluation in the ./res folder. 
    The output of LLM should be put in the ./res folder, in the name of ./res/{model_name}_{suffix}/{name}.json.
    e.g. ./res/GPT4o-0806_aaaaaaaa/code.json
    """
    for root, d, files in os.walk('res', topdown=False):

        for name in files:
            abbr = root.split("/")[-1]
            rr = abbr.rfind("_")
            abbr = abbr[:rr]
            print("abbr:", abbr)
            if name.find('json') != -1: 
                eval_single(os.path.join(root, name), threshold_shot=999999)    

evaluate()