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
This is (one of) the final step for generating data, which should be run after ../get_shots/build_dataset_variants_partB_step4.py and with GPT-labeled difficulty level (which is in ../get_shots/vgpt for this repo).
This script will generate three files, which are unaware ('free_wrong'), aware_error ('free_aware1_wrong'), aware_ratio ('free_aware2_wrong').
The output of this script will be stored in the ./data folder as a parquet, for which "answer" is the ground truth answer, "prompt" are inputs for LLMs, and the output of LLMs are supposed to go into "predict0".
"""
data1_old = pd.read_parquet("../../get_generator/construct_data_final.parquet")
data1 = pd.read_json("../../get_generator/result/GPT4o.json")

import ast

prompts_1, plen_1, answers_1, error_rate = {}, {}, {}, {}
prompts_2, plen_2, answers_2, idx_train, idx_test, num_shots = {}, {}, {}, {}, {}, {}
for k in ["free_wrong", "free_aware1_wrong", "free_aware2_wrong"]: # aware1 = know there are some error shots, aware2 = know the ratio of error shots
    prompts_1[k], plen_1[k], answers_1[k], error_rate[k] = [], [], [], []
    prompts_2[k], plen_2[k], answers_2[k], idx_train[k], idx_test[k], num_shots[k] = [], [], [], [], [], []

def split_string_by_indices(s: str, indices: list) -> list:
    # Add the start and end of the string to the indices list
    indices = [0] + indices + [len(s)]
    
    # Slice the string based on the indices
    return [s[indices[i]:indices[i + 1]] for i in range(len(indices) - 1)]

def parse_python_object(s, flag=1, wrap_flag=False): # flag = 1: handle most outer level ","
    # print("s:", s)
    if flag == 1: 
        s = s.split('# ')[0].lstrip().rstrip()
    # print("parse:", s)
    try:
        # Safely evaluate the string
        parsed_object = eval(s) # ast.literal_eval(s)
        return parsed_object
    except NameError as e:
        if wrap_flag:
            try: # try to wrap the string in a list
                parsed_object = eval('"'+s+'"')
                print("special parsing:", s, " ", e)
                return parsed_object
            except:
                print(f"\n\nError parsing: {s} {e}")
                return None
    except (ValueError, SyntaxError) as e:
        # v3 format (dict)
        print(f"\n\nError parsing: {s} {e}")
        exit(0)
        # ends here

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


def get_generators(data):
    codes, sp_inputs, sp_outputs, gen1s, gen2s, gen3s = [], [], [], [], [], []
    for i in range(len(data['generators'])):
        print("i:", i, len(data['generators']))
        gener = data['generators'][i]
        # print(gener)
        start_gen1 = gener.find("def gen1(")
        start_gen2 = gener.find("def gen2(")
        start_gen3 = gener.find("def gen3(") # gen3 and special cases are deprecated.
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
            #xx.append(parse_python_object(cur_input))
            #yy.append(parse_python_object(cur_output))
        sp_inputs.append(xx)
        sp_outputs.append(yy)
        # print(sp_inputs)
        # gen1 for training data, gen2 for test data, gen3 for confusing data
    data['gen1'], data['gen2'], data['gen3'] = gen1s, gen2s, gen3s
    data['gen_code'] = codes
    data['sp_input'] = sp_inputs # deprecated 
    data['sp_output'] = sp_outputs # deprecated
    return data

data_withplan = get_generators(data_withplan)

def construct_string_from_shots(num_shots, input_example, output_example, special_data=None, error_rate=0): # special data is a list of tuples (input, output)
    res = ""
    data = list(zip(input_example, output_example))
    N = len(data)
    # print("data:", data)
    # print("N:", N)
    idx = np.arange(num_shots) # np.random.choice(list(filter(lambda x: len(data[x]) == 2, range(N))), num_shots - (len(special_data) if special_data is not None else 0))
    if special_data is not None: 
        special_idx = np.random.choice(np.arange(num_shots), len(special_data), replace=False)
        k = 0
        for x in special_idx:
            data[x] = special_data[k]
            k += 1
        # data += special_data 
    
    if error_rate > 0: 
        error_idx = set(np.random.choice(np.arange(num_shots), int(error_rate*num_shots), replace=False))
        # wrong_answer_idx = np.random.choice(np.arange(16384, N), int(error_rate*num_shots), replace=False)
        # we want to make sure wrong answer is different from right answer!
    for k, j in enumerate(idx):
        if error_rate == 0 or j not in error_idx:
            # print("l:", j, len(data))
            res += "Input:" + str(data[j][0]) + "\nOutput:" + str(data[j][1]) + "\n"
        else:
            wrong_answer_idx, wrong_answer = -1, None
            while wrong_answer_idx == -1:
                idx = np.random.choice(np.arange(8192, N), 1) # we only used the first 2048 shots in this variant
                wrong_answer = data[idx[0]][1]
                #print("idx:", idx, wrong_answer, data[j][1])
                if wrong_answer != data[j][1]:
                    wrong_answer_idx = idx[0]
                    #print("found!")
                    break
            # assert wrong_answer is not None, 'Error!' + str(idx[0]) + " " + str(len(data))
            res += "Input:" + str(data[j][0]) + "\nOutput:" + str(data[wrong_answer_idx][1]) + "\n"
            #if k < 2:
            #    print("input:", data[j][0], "wrong answer:", data[wrong_answer_idx][1], "right answer:", data[j][1])
    return res

import json
SUFFIX = "64acc-quad-formal"
def generate_data(data):
    
    short_lst, cnt = [], 0
    # This file is given in the repo.
    with open("../128-freecotdirect/res/64acc-final-quad.txt") as f:
        lines = f.readlines()
        for line in lines:
            short_lst.append(int(line))
    short_lst = sorted(short_lst)
    #print("short_lst:", short_lst)
    for i in tqdm(range(len(data['gen_code']))):
        if i not in short_lst: continue

        trainset, testset, confusing_trainset = torch.load("../../get_shots/data_genshot/shots_"+str(i)+"_"+"trainset.pt"), torch.load("../../get_shots/data_genshot/shots_"+str(i)+"_"+"testset.pt"), torch.load("../../get_shots/data_genshot/shots_"+str(i)+"_"+"confusing_trainset.pt")
        trainset_res, testset_res, confusing_trainset_res = torch.load("../../get_shots/data_exec_output/"+str(i)+"_"+"trainset_data.pt"), torch.load("../../get_shots/data_exec_output/"+str(i)+"_"+"testset_data.pt"), torch.load("../../get_shots/data_exec_output/"+str(i)+"_"+"confusing_trainset_data.pt")
        
        prompt_part0_free = "You are given some function that takes something as input and output something. You need to predict the output for the target input of that function. Remember always end your answer with 'Output: {your answer}', with your answer in strict python format.\n"

        def build_shots(num_shot, j, flag="direct", ER=0):
            prompt_part0 = prompt_part0_free # if flag == "free" else (prompt_part0_free_aware1 if flag == "free_aware1" else prompt_part0_free_aware2.replace(?, ?))
            
            if flag == "free_aware1": prompt_part0 += "Here are some examples. Note that not all shots are correct; there are a small portion of shots that are incorrect:"
            elif flag == "free_aware2": prompt_part0 += "Here are some examples. Note that not all shots are correct; there are " + str(int(num_shot * ER)) + " out of " + str(num_shot) + " shots that are incorrect:"
            else: prompt_part0 += "Here are some examples:"

            prompt_1 = prompt_part0 + construct_string_from_shots(num_shot, trainset, trainset_res, error_rate=ER)
            

            if flag == "free_aware1": prompt_1 += "Again, note that not all shots are correct; there are a small portion of shots that are incorrect. Use your caution and think wisely."
            elif flag == "free_aware2": prompt_1 += "Again, note that not all shots are correct; there are " + str(int(num_shot * ER)) + " out of " + str(num_shot) + " shots that are incorrect. Use your caution and think wisely."

            prompt_1 += "Input:" + str(testset[j]) + ("\nHere is your code. Again, do not output anything else; Your function name should be 'solution'. You are not allowed to write other custom functions unless it is inside 'solution'. Use imports before using package functions. You must strictly follow python format, especially input / output format (e.g., if it is a dictionary, your param should also be a dictionary). DO NOT ADD ANY STATEMENT FOR EVALUATION AFTER 'solution'.\n Code:" if flag == "code" else "")
            prompt_2 = ""

            return prompt_1, prompt_2
        TOO_LONG_FLAG = 0
        for flag in ["free", 'free_aware1', 'free_aware2']:
            print(i, "generating correct shots...")
            for num_shot in [64, 256, 1024]:
              for ER in [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 0.75]: # error rate
                for j in range(len(testset)): # differƒ
                    prompt_1, prompt_2 = build_shots(num_shot, j, flag=flag, ER=ER)

                    prompts_1[flag+"_wrong"].append(prompt_1)
                    plen_1[flag+"_wrong"].append(len(prompt_1))
                    answers_1[flag+"_wrong"].append(str(testset_res[j]))
                    prompts_2[flag+"_wrong"].append(prompt_2)
                    plen_2[flag+"_wrong"].append(len(prompt_2))
                    answers_2[flag+"_wrong"].append(str(testset_res[j]))
                    idx_train[flag+"_wrong"].append(i)
                    idx_test[flag+"_wrong"].append(j)
                    num_shots[flag+"_wrong"].append(num_shot)
                    error_rate[flag+"_wrong"].append(ER)
            if TOO_LONG_FLAG == 1: break
        if TOO_LONG_FLAG == 0:
            cnt += 1
    print("total valid number count:", cnt)
generate_data(data_withplan)
for k in prompts_1.keys():
    print("key:", k)
    print("len:", len(prompts_1[k]))
    new_data_1 = {'prompt': prompts_1[k], 'answer': answers_1[k], 'idx_train': idx_train[k], 'idx_test': idx_test[k], 'num_shots': num_shots[k], 'plen': plen_1[k], 'ER_rate': error_rate[k]}
    pd.DataFrame(new_data_1).to_parquet('data/build_data_'+k+'_random_'+SUFFIX+'.parquet', engine='pyarrow')
