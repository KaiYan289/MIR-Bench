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
'''
This is the third script that we run.
When executing this script, it will read the data generator that GPT-4o-0806 generated in the get_generator folder (../get_generator/result/GPT4o-0806.json), and then generate runnable python scripts in the codes_genshot folder for each problem.
The next step after running this would be running each of the script in the codes_genshot folder. You don't have to do it manually; simply go into codes_genshot and execute ./run.sh.
'''

# data1 is just for getting generation data. The order always follow construct_data_final.parquet.
data1_old = pd.read_parquet("../get_generator/construct_data_final.parquet")
data1 = pd.read_json("../get_generator/result/GPT4o-0806.json")

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


def find_all_occurrences(substring, string):
    matches = [match.start() for match in re.finditer(re.escape(substring), string)]
    return matches
    
# Note: gen3 and special cases are deprecated in our work.
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

def execute_code(code, gen_code, typ, N, idx, name):
    imports = "import torch\nimport sys\nimport json\nimport random, string\nfrom typing import List, Tuple\nimport math\nimport bisect\nfrom bisect import *\nfrom collections import *\nimport heapq\n"#\nimport sys, os\nprint('Python version:', sys.version)\nprint('Current working directory:', os.getcwd())"
    if gen_code.find('yield') != -1:
        code = imports + code + "\nres = list(gen"+str(typ)+"("+str(N)+"))\nprint('Data generation and saving complete!')\ntorch.save(res,'../data_genshot/shots_"+str(idx)+"_"+name+".pt')" #+ "with open('tmp.json', 'w') as f:\n"+"    json.dump(res, f)\n"+ "print('tmp.py executed successfully\n', flush=True)\nsys.stdout.flush()"
    else:
        code = imports + code + "\nres = gen"+str(typ)+"("+str(N)+")\nprint('Data generation and saving complete!')\ntorch.save(res,'../data_genshot/shots_"+str(idx)+"_"+name+".pt')" #+ "with open('tmp.json', 'w') as f:\n"+"    json.dump(res, f)\n" + "print('tmp.py executed successfully\n', flush=True)\nsys.stdout.flush()"
    with open("codes_genshot/"+str(idx)+"_"+name+".py", "w") as f:
        f.write(code)
    print(code)


import json

def generate_data(data):

    for i in tqdm(range(len(data['gen_code']))):# tqdm(range(len(data['gen_code']))):

        t0 = time.time()
        abandon_this_prob = False
        trainset = execute_code(data['gen_code'][i], data['gen1'][i], 1, 20000, i, "trainset")
        testset = execute_code(data['gen_code'][i], data['gen2'][i], 2, 10, i, "testset")
        confusing_trainset = execute_code(data['gen_code'][i], data['gen3'][i], 3, 20000, i, "confusing_trainset")

generate_data(data_withplan)

