import numpy as np
import pandas as pd
import re
import random
import string
import torch
from typing import List, Tuple
import math
import time
import os
import subprocess
from copy import deepcopy
from tqdm import tqdm
np.random.seed(42)
random.seed(42)
"""
This is the fifth step, which is to generate output data based on input data.
This script will read the ground truth functions in ../get_generator and the input data in data_genshot, and put them together for a runnable script for output for each problem.
The code will be in the codes_exec_output folder, and the generated output will be in the data_exec_output folder.
Again, run each script in the codes_exec_output folder after this by go into the folder and executing ./run.sh.

We also provide the average, normalized difficulty level label in vgpt/difficulty-gpt.txt.

"""
data1_old = pd.read_parquet("../get_generator/construct_data_final.parquet")
data1 = pd.read_json("../get_generator/result/GPT4o-0806.json")
print(data1.keys())

import ast

def split_string_by_indices(s: str, indices: list) -> list:
    # Add the start and end of the string to the indices list
    indices = [0] + indices + [len(s)]
    
    # Slice the string based on the indices
    return [s[indices[i]:indices[i + 1]] for i in range(len(indices) - 1)]

def match_construct_data(data_old, data_withgen):
    q_idx = {}
    datas = [None for _ in range(len(data_old['prompt']))]
    for i in range(len(data_old['prompt'])):
        assert data_old['prompt'][i] not in data_old, "Error!"
        q_idx[data_old['prompt'][i]] = i
    for i in range(len(data_withgen['prompt'])):
        assert data_withgen['prompt'][i] in q_idx, "Error!"
        old_idx = q_idx[data_withgen['prompt'][i]]
        datas[old_idx] = data_withgen['predict0'][i]
    A = deepcopy(data_old)
    A['generators'] = datas
    return A 

data_withplan = match_construct_data(data1_old, data1)
# print(data_withplan['prompt'][0], data_withplan['generators'][0])

def find_all_occurrences(substring, string):
    matches = [match.start() for match in re.finditer(re.escape(substring), string)]
    return matches


def get_generators(data):
    codes, sp_inputs, sp_outputs, gen1s, gen2s, gen3s = [], [], [], [], [], []
    for i in range(len(data['generators'])):
        print("i:", i, len(data['generators']))
        gener = data['generators'][i]
        # print(gener)
        start_gen1 = gener.find("def gen1(")
        start_gen2 = gener.find("def gen2(")
        start_gen3 = gener.find("def gen3(")
        start_sp_case = gener.find("pecial cases")
         
        gen1 = gener[start_gen1:start_gen2].split('[[Gen2]]')[0]
        gen2 = gener[start_gen2:start_gen3].split('[[Gen3]]')[0]
        gen3 = gener[start_gen3:].split('[[')[0].split('Special cases')[0].split('special cases')[0]
        
        gen1s.append(gen1)
        gen2s.append(gen2)
        gen3s.append(gen3)

        code = (gen1 + "\n" + gen2 + "\n" + gen3).replace("```", "")
        codes.append(code)
        sp_case = gener[start_sp_case:].split(']]')[-1]
        inps, outps = find_all_occurrences('Input:', sp_case), find_all_occurrences('Output:', sp_case)
        xx, yy = [], []
        for x, y in zip(inps, outps):
            #print("sp_case:", sp_case)
            cur_input, cur_output = sp_case[x+6:].split("Output:")[0], sp_case[y+7:].split("Input:")[0].split('```')[0]
            #print("cur_input:", cur_input)
            #print(cur_output)
            xx.append(cur_input)
            yy.append(cur_output)

        sp_inputs.append(xx)
        sp_outputs.append(yy)
        # print(sp_inputs)
        # gen1 for training data, gen2 for test data, gen3 for confusing data
    data['gen1'], data['gen2'], data['gen3'] = gen1s, gen2s, gen3s
    data['gen_code'] = codes
    data['sp_input'] = sp_inputs
    data['sp_output'] = sp_outputs
    return data


data_withplan = get_generators(data_withplan)

import json

def run_shots(data, codex, idx, name):
    res, error_cnt = [], 0
    # print("codex:", codex)
    try:
        func_name = codex.split("def ")[1].split('(')[0]
    except:
        print("wrong format!")
        return None
    if func_name == "do_algebra": 
        print("skip this problem...", name)
        return None
    lef = codex.find("def "+func_name) # remove text description
    imports = "import re\nimport torch\nimport sys\nimport json\nimport random, string\nfrom typing import List, Tuple\nimport math\nimport bisect\nfrom bisect import *\nfrom collections import *\nimport heapq\n"
    codex = imports + "\nres = []\n" + codex[lef:]
    for i in range(len(data)):
        tot_len = 0
        if not isinstance(data[i], dict):
            print("data is not dict; skipping...")
            continue
        for k in data[i].keys():
            tot_len += len(str(data[i][k]))
            if tot_len >= 1000: 
                print("data is too long; skipping...")
                continue
        try:
            codex += "\nres.append("+func_name+"(*" + json.dumps(list(data[i].values())) +"))"# v3 format
        except:
            print("unrecognized data. return None...")
            return None
    codex += "\ntorch.save(res, '../data_exec_output/"+str(idx)+"_"+name+"_data.pt')"
    with open("codes_exec_output/"+str(idx)+"_"+name+".py", "w") as f:
        f.write(codex)
        # print("result:", res_now)

def generate_data(data):

    short_lst = []
    with open("shots_lim_128_4096_afterpartA.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            short_lst.append(int(line.split(" ")[0]))
    for i in tqdm(range(len(data['gen_code']))):# tqdm(range(len(data['gen_code']))):
        if i not in short_lst: continue

        trainset, testset, confusing_trainset = torch.load("data_genshot/shots_"+str(i)+"_"+"trainset.pt"), torch.load("data_genshot/shots_"+str(i)+"_"+"testset.pt"), torch.load("data_genshot/shots_"+str(i)+"_"+"confusing_trainset.pt")

        if trainset is None or testset is None or confusing_trainset is None: 
            abandon_this_prob = True
        elif len(trainset) < 20000 or len(testset) < 10 or len(confusing_trainset) < 20000:
            print("abandon!")
            exit(0)
            abandon_this_prob = True
        run_shots(trainset, data['problem'][i] + data['solution'][i], i, "trainset")
        run_shots(confusing_trainset, data['problem'][i] + data['solution'][i], i, "confusing_trainset")
        run_shots(testset, data['problem'][i] + data['solution'][i], i, "testset")


generate_data(data_withplan)

#data = pd.read_parquet("data/v1-500p/build_data_random_19.parquet")
#print(data['answer'])

