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
import torch
from util import *

"""
This is the code for evaluating results for LLM with solverlearner framework.
It should be run after you finished ./analyzer.py and ./code/run.sh. 
"""

def evaluate():
    original_data = pd.read_parquet("data/build_data_code_right_random_64acc-quad-formal.parquet")
    # print("original_data:", original_data.keys())
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    # print(original_data.keys())
    prompts_idx = {}
    for i in range(len(original_data['prompt'])):
        prompts_idx[original_data['prompt'][i]] = i
    
    def judge_ans(gt, answers, info):
        N, r, error, BO5 = len(answers), 0, 0, 0
        for j, output in enumerate(answers):
            last_output = output
            ans = gt.lstrip().rstrip()
            ret_left = last_output.find("[<-[")
            ret_right = last_output.find("]->]")
            if ret_left == -1 or ret_right == -1 or ret_left + 4 >= ret_right:
                error += 1
                continue
            res = last_output[ret_left+4:ret_right].lstrip().rstrip()
            if str(res) == str(ans):
                r += 1
        return r / N, error / N
    
    all_tot_by_shots, all_score_by_shots, all_unrunnable_by_shots = {}, {}, {}

    def eval_single(prompt, answer, unrunnable_by_shots, tot_by_shots, score_by_shots, threshold_shot=4096):
        res = original_data

        tot_by_shots, score_by_shots = {}, {}
        unrunnable_by_shots = {}
        
        idx_in_original = prompts_idx[prompt]
        # print(idx_in_original)
        plen, num_shots = original_data['plen'][idx_in_original], original_data['num_shots'][idx_in_original]
        train_idx = original_data['idx_train'][idx_in_original]

        JA = judge_ans(original_data['answer'][idx_in_original], [answer], None)#, (name.split("/")[1], i))

        unrunnable_by_shots[num_shots] = unrunnable_by_shots.get(num_shots, 0) + JA[1]
        tot_by_shots[num_shots] = tot_by_shots.get(num_shots, 0) + 1
        score_by_shots[num_shots] = score_by_shots.get(num_shots, 0) + JA[0]

        return num_shots, JA[0], JA[1]
        
    cnt, t0 = 0, time.time()
    for root, d, files in os.walk('code', topdown=False):
        for name in files:
            #print("root:", root, "d:", d, "files:", files)
            if name.find(".txt") == -1 or name.find("prompt") != -1 or name.find("verdict") != -1: continue
            
            if len(root.split("/")) < 3: continue
            abbr = root.split("/")[1]
            rr = abbr.rfind("_")
            abbr = abbr[:rr]

            cnt += 1
            if cnt % 1000 == 0: 
                print("current problem count:", cnt, "/", 72000, "time:", time.time() - t0)
            
            if os.path.exists(root + "/" + name.replace(".txt", "_verdict.txt")):
                JA = [0, 0, 0]
                f = open(root + "/" + name.replace(".txt", "_verdict.txt"), "r")
                j, JA[0], JA[1] = f.read().split()
                j, JA[0], JA[1] = int(j), float(JA[0]), float(JA[1])
                f.close()
                all_tot_by_shots[abbr] = all_tot_by_shots.get(abbr,{}) 
                all_score_by_shots[abbr] = all_score_by_shots.get(abbr,{}) 
                all_unrunnable_by_shots[abbr] = all_unrunnable_by_shots.get(abbr,{}) 
                all_tot_by_shots[abbr][j] = all_tot_by_shots[abbr].get(j, 0) + 1
                all_score_by_shots[abbr][j] = all_score_by_shots[abbr].get(j, 0) + JA[0]
                all_unrunnable_by_shots[abbr][j] = all_unrunnable_by_shots[abbr].get(j, 0) + JA[1]
                continue 
            
            f = open(root + "/" + name.replace(".txt", "_prompt.txt"), "r")
            prompt = f.read()
            f.close()
            
            g = open(root + "/" + name, "r")
            answer = g.read()
            g.close()

            all_tot_by_shots[abbr] = all_tot_by_shots.get(abbr,{})
            all_score_by_shots[abbr] = all_score_by_shots.get(abbr,{})
            all_unrunnable_by_shots[abbr] = all_unrunnable_by_shots.get(abbr,{})
            JA = [0, 0, 0]
            num_shots, JA[0], JA[1] = eval_single(prompt, answer, all_unrunnable_by_shots[abbr], all_tot_by_shots[abbr], all_score_by_shots[abbr], threshold_shot=999999)
            h = open(root + "/" + name.replace(".txt", "_verdict.txt"), "w")
            h.write(str(num_shots) + ' ' + str(JA[0]) + ' '+str(JA[1]))
            h.close()               

    colors = {'Claude-35-sonnet.sdk.v2': 'red', 'Gemini-1.5_flash-002': 'darkblue', 'Gemini-1.5_pro-002': 'blue', 'GPT4o-0806': 'green', 'GPT4o-mini-0718': 'lightgreen', 'Mistral-large-2': 'purple'}                
    names = {'Claude-35-sonnet.sdk.v2': 'Claude-3.5 Sonnet', 'Gemini-1.5_flash-002': 'Gemini-1.5 Flash-002', 'Gemini-1.5_pro-002': 'Gemini-1.5 Pro-002', 'GPT4o-0806': 'GPT4o-0806', 'GPT4o-mini-0718': 'GPT4o-mini-0718', 'Mistral-large-2': 'Mistral-large-2'}
    def plot(score_by_ER, score_by_shots, tot_by_ER, tot_by_shots, abbr):
        x = list(map(lambda x: math.log(x, 2), sorted(np.array(list(score_by_shots.keys())))))
        y = []
        for k in sorted(score_by_shots.keys()):
            y.append(score_by_shots[k]/tot_by_shots[k])
        ax.plot(x, y, label=names[abbr], color=colors[abbr])

    for k in all_score_by_shots.keys():
        print("key!!:", k)
        torch.save([all_score_by_shots[k], all_tot_by_shots[k], all_unrunnable_by_shots[k]], k+".pt")
        for j in sorted(all_score_by_shots[k].keys()):
            print("shots:", j, "score:", all_score_by_shots[k][j], "/", all_tot_by_shots[k][j], "=", all_score_by_shots[k][j] / all_tot_by_shots[k][j], "unrunnable:", all_unrunnable_by_shots[k][j] / all_tot_by_shots[k][j])
        plot(None, all_unrunnable_by_shots[k], None, all_tot_by_shots[k], k)
 
    ax.set_xlabel('Number of shots')
    ax.set_ylabel('Accuracy')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig("unrunnable-coding.png")
evaluate()