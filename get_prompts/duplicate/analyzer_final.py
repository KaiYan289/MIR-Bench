import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import ast
import math
import torch

STR = "res"
assert STR in ['res-dupeven', 'res'], 'Error!'

if STR == 'res':
    merged_data = pd.read_parquet("data/build_data_free_right_random_64acc-quad-duplicate.parquet")
if STR == 'res-dupeven':
    merged_data = pd.read_parquet("data/build_data_free_right_random_64acc-quad-duplicate_dupeven.parquet")

idx_merged_data = {}
for i in range(len(merged_data['prompt'])):
    idx_merged_data[merged_data['prompt'][i]] = i

def evaluate(flag, error, data_type, standard_file=None, suffix=""):
    fig, ax = plt.subplots(1, 2, figsize=(10*0.8, 4.5*0.8))
    def judge_ans(gt, answers):
        N, res = len(answers), 0
        for output in answers:
            last_output = output.rfind("utput:")
            if output[last_output+6:last_output+8] == "**": # remove bold
                output = output[:last_output+6] + output[last_output+8:]
            verdict = output[last_output+6:].replace("```python", "").replace("```", "").lstrip().rstrip()
            ans = gt.lstrip().rstrip()

            flag = 0
            if verdict == ans: flag = 1
            
            else:
                try: 
                    v1 = ast.literal_eval(verdict)
                    if isinstance(v1, dict) and len(v1.keys()) == 1:
                        k = list(v1.keys())[0]
                        v1 = v1[k]
                    if isinstance(v1, set) and len(v1) == 1:
                        v1 = list(v1)[0]

                    if str(v1) == ans: flag = 1
                except:
                    pass
            
            res += flag
        return res / N

    def eval_single(name, abbr, threshold_shot=4096, standard_file=None, suffix=""):

        res = pd.read_json(name)
        print(res.keys())
        lst_short_gpt4 = []
        tot_by_ER, score_by_ER = {}, {}
        tot_by_shots, score_by_shots = {}, {}
        chelect = []
        if standard_file is not None:
            for file in standard_file:
                print("file:", file)
                chelect.append([])
                with open(file, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        chelect[-1].append(int(line))
                print("chelect:", chelect)

        else:
            print("standard_file is None")
        
        for i in range(len(res['prompt'])):
          for seed in range(1):
            idx_in_merged = idx_merged_data[res['prompt'][i]]
            error_rate, plen, num_shots = merged_data['ER_rate'][idx_in_merged], merged_data['plen'][idx_in_merged], merged_data['prompt'][idx_in_merged].count('Input:') - 1
            train_idx = merged_data['idx_train'][idx_in_merged]

            print("num_shots:", num_shots)

            if len(chelect) > 0:
                flag = 0
                for lst in chelect:
                    if train_idx not in lst:
                        flag = 1
                        break
                if flag == 1: continue  

            tot_by_ER[error_rate] = tot_by_ER.get(error_rate, np.zeros(1))
            score_by_ER[error_rate] = score_by_ER.get(error_rate, np.zeros(1))
            tot_by_shots[num_shots] = tot_by_shots.get(num_shots, np.zeros(1))
            score_by_shots[num_shots] = score_by_shots.get(num_shots, np.zeros(1))

            tot_by_ER[error_rate][seed] += 1
            score_by_ER[error_rate][seed] += judge_ans(res['answer'][i][0], [res['predict'+str(j)][i] for j in range(seed, seed+1)])

            tot_by_shots[num_shots][seed] += 1
            score_by_shots[num_shots][seed] += judge_ans(res['answer'][i][0], [res['predict'+str(j)][i] for j in range(seed, seed+1)])
        
        for k in sorted(tot_by_ER.keys()):
            print("seed:", seed, "ER:", k, "score:", score_by_ER[k][seed], "/", tot_by_ER[k][seed], "=", score_by_ER[k][seed] / tot_by_ER[k][seed])
        for k in sorted(tot_by_shots.keys()):
            print("seed:", seed, "shots:", k, "score:", score_by_shots[k][seed], "/", tot_by_shots[k][seed], "=", score_by_shots[k][seed] / tot_by_shots[k][seed])

        return score_by_ER, score_by_shots, tot_by_ER, tot_by_shots
    

    sbyer, sbyshots, totbyer, totbyshots, colors = {}, {}, {}, {}, {}


    for root, d, files in os.walk(STR, topdown=False):
        for file_name in files:
            print("file name:", root, d, file_name)
            # exit(0)
            abbr = root.split("/")[-1]
            loc = abbr.rfind("_")
            abbr = abbr[:loc]
            print("abbr:", abbr)
            if file_name.find('json') != -1: 

                print("name:", file_name)
                # if file_name.find(STR) == -1: continue
                score_by_ER, score_by_shots, tot_by_ER, tot_by_shots = eval_single(os.path.join(root, file_name), abbr, threshold_shot=114514, standard_file=standard_file, suffix=suffix)
                if abbr.find("Claude-35-sonnet") != -1: name, color = 'Claude-35-sonnet', 'red'
                elif abbr.find("Claude-3-sonnet") != -1: name, color = 'Claude-35-sonnet', 'gold'
                elif abbr.find("Gemini-1.5_flash-002") != -1: name, color = 'Gemini-1.5 Flash-002', 'darkblue'
                elif abbr.find("Gemini-1.5_pro-002") != -1: name, color = 'Gemini-1.5 Pro-002', 'blue'
                elif abbr.find("Gemini-2.0") != -1: name, color = 'Gemini-2.0 Flash', 'lightblue'
                elif abbr.find("GPT4o-0806") != -1: name, color = 'GPT4o-0806', 'green'
                elif abbr.find("GPT4o-mini-0718") != -1: name, color = 'GPT4o-mini-0718', 'lightgreen'
                elif abbr.find("Mistral-large-2") != -1: name, color = 'Mistral-large-2', 'purple'
                elif abbr.find("OpenAI-o1-mini") != -1: name, color = 'OpenAI-o1-mini', 'black'
                elif abbr.find("OpenAI-o1-preview") != -1: name, color = 'OpenAI-o1-preview', 'grey'
                elif abbr.find("Claude-3-haiku") != -1: name, color = 'Claude-3-haiku', 'orange'
                elif abbr.find("Claude-35-haiku") != -1: name, color = 'Claude-35-haiku', 'salmon'
                elif abbr.find('Moonshot-128k') != -1: name, color = 'Moonshot-128k', 'brown'
                elif abbr.find("Qwen2-72B") != -1: name, color = 'Qwen2-72B-Instruct', 'pink'
                else: continue
                if name not in sbyer: sbyer[name], sbyshots[name], totbyer[name], totbyshots[name] = score_by_ER, score_by_shots, tot_by_ER, tot_by_shots
                else: 
                    sbyer[name].update(score_by_ER)
                    sbyshots[name].update(score_by_shots)
                    totbyer[name].update(tot_by_ER)
                    totbyshots[name].update(tot_by_shots)
                colors[name] = color
                print("abbr:", abbr)
    def plot(score_by_ER, score_by_shots, tot_by_ER, tot_by_shots, abbr, color):
        print("score_by_shots:", score_by_shots)
        x = list(map(lambda x: math.log(x, 2), sorted(np.array(list(score_by_shots.keys())))))
        y = []
        std = []
        for k in sorted(score_by_shots.keys()):
            arr = score_by_shots[k]/tot_by_shots[k]
            y.append(arr.mean())
            std.append(arr.std())
        x, y, std = np.array(x), np.array(y), np.array(std)
        if len(x) > 1: 
            ax[0].plot(x, y, label=abbr, color=color)
            # ax[0].fill_between(x, y-std, y+std, color=color, alpha=0.5)
        else: 
            ax[0].plot(x, y)

        ax[1].plot([0], [0], label=abbr, color=color)
    
    torch.save([sbyer, sbyshots, totbyer, totbyshots, colors], STR+".pt")
    exit(0)
    for k in sbyer.keys():
        print("key:", k)
        plot(sbyer[k], sbyshots[k], totbyer[k], totbyshots[k], k, colors[k])    

    ax[0].set_xlabel('Number of shots (in power of 2)')
    ax[0].set_ylabel('Accuracy')    
    ax[0].legend()
    plt.savefig("selected_"+suffix+".png", bbox_inches='tight')

evaluate('free', 'right', 'random', standard_file=None, suffix=STR)
