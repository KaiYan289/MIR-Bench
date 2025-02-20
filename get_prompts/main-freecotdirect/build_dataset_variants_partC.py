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

MAIN_FLAG = False # set this to True for generating data on MIR-Extended; False for generating data on MIR-Core.

"""
This is the code for generating main result and CoT result in our paper.
Make sure you have run ../get_shots/build_dataset_variants_partB_step4.py first.
"""

data1_old = pd.read_parquet("../../get_generator/construct_data_final.parquet")
data1 = pd.read_json("../../get_generator/result/GPT4o-0806.json")

import ast

prompts_1, plen_1, answers_1, error_rate = {}, {}, {}, {}
prompts_2, plen_2, answers_2, idx_train, idx_test, num_shots = {}, {}, {}, {}, {}, {}
for k in ["free_right", "cot_right", "direct_right"]:
    # free_right: no specification on CoT, which is the main result
    # cot_right: must use CoT
    # direct_right: must not use CoT
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
            #xx.append(parse_python_object(cur_input))
            #yy.append(parse_python_object(cur_output))
        sp_inputs.append(xx)
        sp_outputs.append(yy)
        # print(sp_inputs)
        # gen1 for training data, gen2 for test data, gen3 for confusing data
    data['gen1'], data['gen2'], data['gen3'] = gen1s, gen2s, gen3s # gen3 deprecated
    data['gen_code'] = codes
    data['sp_input'] = sp_inputs # deprecated
    data['sp_output'] = sp_outputs # deprecated
    return data

data_withplan = get_generators(data_withplan)

def construct_string_from_shots(num_shots, input_example, output_example, special_data=None, error_rate=0): # special data is a list of tuples (input, output)
    res = ""
    data = list(zip(input_example, output_example))
    N = len(data)
    idx = np.arange(num_shots) 
    if special_data is not None: # deprecated; always set to None
        special_idx = np.random.choice(np.arange(num_shots), len(special_data), replace=False)
        k = 0
        for x in special_idx:
            data[x] = special_data[k]
            k += 1
        # data += special_data 
    assert error_rate == 0, "Error!"
    if error_rate > 0: # always set to 0 in this folder; use ../robust for error rate > 0
        error_idx = set(np.random.choice(np.arange(num_shots), int(error_rate*num_shots), replace=False))
        wrong_answer_idx = np.random.choice(np.arange(16384, N), int(error_rate*num_shots), replace=False)
    k = 0
    for j in idx:
        if error_rate == 0 or j not in error_idx: # should always use this branch
            # print("l:", j, len(data))
            res += "Input:" + str(data[j][0]) + "\nOutput:" + str(data[j][1]) + "\n"
        else:
            # print("num_shots:", num_shots, N)
            if k < 10:
                print("error data!", j,  wrong_answer_idx[k], data[j][1], data[wrong_answer_idx[k]][1])
            # print("wrong answer idx:", wrong_answer_idx)
            res += "Input:" + str(data[j][0]) + "\nOutput:" + str(data[wrong_answer_idx[k]][1]) + "\n"
            k += 1
    return res

import json
SUFFIX = "64acc-all-formal" if MAIN_FLAG else "64acc-quad-formal"
def generate_data(data):
    short_lst, cnt = [], 0
    with open(("res/64acc-final-all.txt" if MAIN_FLAG else "res/64acc-final-quad.txt")) as f:
        lines = f.readlines()
        for line in lines:
            short_lst.append(int(line))
    short_lst = sorted(short_lst)

    for i in tqdm(range(len(data['gen_code']))):# tqdm(range(len(data['gen_code']))):
        if i not in short_lst: continue

        trainset, testset, confusing_trainset = torch.load("../../get_shots/data_genshot/shots_"+str(i)+"_"+"trainset.pt"), torch.load("../../get_shots/data_genshot/shots_"+str(i)+"_"+"testset.pt"), torch.load("../../get_shots/data_genshot/shots_"+str(i)+"_"+"confusing_trainset.pt")
        trainset_res, testset_res, confusing_trainset_res = torch.load("../../get_shots/data_exec_output/"+str(i)+"_"+"trainset_data.pt"), torch.load("../../get_shots/data_exec_output/"+str(i)+"_"+"testset_data.pt"), torch.load("../../get_shots/data_exec_output/"+str(i)+"_"+"confusing_trainset_data.pt")
        
        prompt_part0_free = "You are given some function that takes something as input and output something. You need to predict the output for the target input of that function. Remember always end your answer with 'Output: {your answer}', with your answer in strict python format. Here are some examples:\n"
        prompt_part0_cot = "You are given some function that takes something as input and output something. You need to predict the output for the target input of that function. You need to first analyze it after 'Analysis:', then give your answer after 'Output:'. Remember always end your answer with 'Output: {your answer}', with your answer in strict python format. Here are some examples:\n"
        prompt_part0_direct = "You are given some function that takes something as input and output something. You need to predict the output for the target input of that function. Your answer should always be 'Output: {your answer}', with your answer in strict python format. DO NOT OUTPUT ANYTHING ELSE INCLUDING YOUR THOUGHTS. Here are some examples:\n"
        prompt_part0_silence = ""
        prompt_part0_code = "You are given some function that takes something as input and output something. You need to write a python code of the function. You need to write your rationale after # (as if it is a python comment), and give your answer after 'Code:'. DO NOT OUTPUT ANYTHING ELSE. Your function name should be 'solution'. You are not allowed to write other custom functions unless it is inside 'solution'. Use imports before using package functions. You must strictly follow python format, especially input / output format (e.g., if it is a dictionary, your param should also be a dictionary). DO NOT ADD ANY STATEMENT FOR EVALUATION AFTER 'solution'. Here are the input-output pairs for the function, with input followed by output:\n"

        def build_shots(num_shot, j, flag="direct", ER=0):
            prompt_part0 = prompt_part0_free if flag == "free" else (prompt_part0_direct if flag == "direct" else (prompt_part0_silence if flag == "silence" else (prompt_part0_cot if flag == "cot" else prompt_part0_code)))
            prompt_1 = prompt_part0 + construct_string_from_shots(num_shot, trainset, trainset_res, error_rate=ER) + "Input:" + str(testset[j]) + ("\nHere is your code. Again, do not output anything else; Your function name should be 'solution'. You are not allowed to write other custom functions unless it is inside 'solution'. Use imports before using package functions. You must strictly follow python format, especially input / output format (e.g., if it is a dictionary, your param should also be a dictionary). DO NOT ADD ANY STATEMENT FOR EVALUATION AFTER 'solution'.\n Code:" if flag == "code" else "")
            prompt_2 = ""
            return prompt_1, prompt_2
        TOO_LONG_FLAG = 0
        for flag in ["direct", "cot", 'free']: 
            print(i, "generating correct shots...")
            for num_shot in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
                for j in range(len(testset)): # differƒ
                    prompt_1, prompt_2 = build_shots(num_shot, j, flag=flag, ER=0)
                    if flag == 'cot': 
                        prompt_1 += "Analysis:"
                        prompt_2 += 'Analysis:'
                    prompts_1[flag+"_right"].append(prompt_1)
                    plen_1[flag+"_right"].append(len(prompt_1))
                    answers_1[flag+"_right"].append(str(testset_res[j]))
                    prompts_2[flag+"_right"].append(prompt_2)
                    plen_2[flag+"_right"].append(len(prompt_2))
                    answers_2[flag+"_right"].append(str(testset_res[j]))
                    idx_train[flag+"_right"].append(i)
                    idx_test[flag+"_right"].append(j)
                    num_shots[flag+"_right"].append(num_shot)
                    error_rate[flag+"_right"].append(0)
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
    # The LLM input is in 'prompt' column, ground truth answer in 'answer' column, and LLM output are supposed to go into 'predict0' column.


