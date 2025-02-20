import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import ast
import math
import torch
STR = "all"

assert STR in ["all", "quad"], "Error!"

merged_data = pd.read_json("../result/merge/merge_gpt4o-0806.json")
idx_merged_data = {}
for i in range(len(merged_data['prompt'])):
    idx_merged_data[merged_data['prompt'][i]] = i

def evaluate(flag, error, data_type, standard_file=None, suffix=""):
    fig, ax = plt.subplots()
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
                    ## print("v1:", type(v1), "ans:", type(ans))
                    if str(v1) == ans: flag = 1
                except:
                    pass
            
            res += flag
        return res / N

    def eval_single(name, abbr, threshold_shot=4096, standard_file=None, suffix=""):
        # print(flag, error, data_type, "name for eval:", name)
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
                # exit(0)
        else:
            print("standard_file is None")
        
        for i in range(len(res['prompt'])):
            # print(idx_in_original)
            idx_in_merged = idx_merged_data[res['prompt'][i]]
            error_rate, plen, num_shots = merged_data['ER_rate'][idx_in_merged], merged_data['plen'][idx_in_merged], merged_data['num_shots'][idx_in_merged]
            train_idx = merged_data['idx_train'][idx_in_merged]
            # if train_idx > 178: continue
            # print(error_rate, plen, num_shots)
            if len(chelect) > 0:
                flag = 0
                for lst in chelect:
                    if train_idx not in lst:
                        flag = 1
                        break
                if flag == 1: continue  

            tot_by_ER[error_rate] = tot_by_ER.get(error_rate, 0) + 1
            score_by_ER[error_rate] = score_by_ER.get(error_rate, 0) + judge_ans(res['answer'][i][0], [res['predict'+str(j)][i] for j in range(1)])

            tot_by_shots[num_shots] = tot_by_shots.get(num_shots, 0) + 1
            score_by_shots[num_shots] = score_by_shots.get(num_shots, 0) + judge_ans(res['answer'][i][0], [res['predict'+str(j)][i] for j in range(1)])
        
        for k in sorted(tot_by_ER.keys()):
            print("ER:", k, "score:", score_by_ER[k], "/", tot_by_ER[k], "=", score_by_ER[k] / tot_by_ER[k])
        for k in sorted(tot_by_shots.keys()):
            print("shots:", k, "score:", score_by_shots[k], "/", tot_by_shots[k], "=", score_by_shots[k] / tot_by_shots[k])

        return score_by_ER, score_by_shots, tot_by_ER, tot_by_shots
    

    sbyer, sbyshots, totbyer, totbyshots, colors = {}, {}, {}, {}, {}

    REDRAW = True # draw from scratch (True) or loading data (False)
 
    for root, d, files in os.walk(".", topdown=False):
        for file_name in files:
            print("file name:", root, d, file_name)
            # exit(0)
            abbr = root.split("/")[-1]
            loc = abbr.rfind("_")
            abbr = abbr[:loc]
            print("abbr:", abbr)
            # new coloring plan
            if abbr.find("Claude-35-sonnet") != -1: name, color = 'Claude-3.5-Sonnet', "#FF9800" # 'red'
            elif abbr.find("Gemini-1.5_flash-002") != -1: name, color = 'Gemini-1.5 Flash-002', 'lightgreen' # "#81C784"
            elif abbr.find("Gemini-1.5_pro-002") != -1: name, color = 'Gemini-1.5 Pro-002', 'green' # "#66BB6A"
            elif abbr.find("GPT4o-0806") != -1: name, color = 'GPT4o-0806', 'red' # "#FFC107" # 'green'
            elif abbr.find("GPT4o-mini-0718") != -1: name, color = 'GPT4o-mini-0718', 'salmon' # "#FFE082" # 'lightgreen'
            elif abbr.find("Mistral-large-2") != -1: name, color = 'Mistral-Large-2', 'purple' # "#00BCD4"
            elif abbr.find("OpenAI-o1-mini") != -1: name, color = 'OpenAI-o1-mini', 'black'# "#F44336"
            elif abbr.find("OpenAI-o1-preview") != -1: name, color = 'OpenAI-o1-preview', 'grey' # "#EF5350"
            ###############
            elif abbr.find("Gemini-2.0") != -1: name, color = 'Gemini-2.0 Flash', 'darkgreen' # "#4CAF50"
            elif abbr.find("Claude-3-sonnet") != -1: name, color = 'Claude-3-Sonnet', "#FFA726" # 'gold'
            elif abbr.find("Claude-3-haiku") != -1: name, color = 'Claude-3-Haiku', "#FFB74D" # 'orange'
            elif abbr.find("Claude-35-haiku") != -1: name, color = 'Claude-3.5-Haiku', "#FFE0B2"
            elif abbr.find('Moonshot-128k') != -1: name, color = 'Moonshot-128k', "#9C27B0"
            elif abbr.find("Qwen2-72B") != -1: name, color = 'Qwen2-72B-Instruct', "#2196F3"
            elif abbr.find("GLM-4-plus") != -1: name, color = 'GLM-4-Plus', 'teal' # "#03A9F4"
            ##############
            else: continue
            
            colors[name] = color
            if REDRAW and os.path.exists("saved_data/"+name+"_"+STR+"_saved_data.pt"):
                print("skipped!", abbr)
                continue

            if file_name.find('json') != -1: 
                #eval_single(os.path.join(root, name), threshold_shot=114514, standard_file=standard_file, suffix=suffix) # for debugging
                #continue
                print("name:", file_name)
                if file_name.find(STR) == -1: continue

                score_by_ER, score_by_shots, tot_by_ER, tot_by_shots = eval_single(os.path.join(root, file_name), abbr, threshold_shot=114514, standard_file=standard_file, suffix=suffix)
                if name not in sbyer: sbyer[name], sbyshots[name], totbyer[name], totbyshots[name] = score_by_ER, score_by_shots, tot_by_ER, tot_by_shots
                else: 
                    sbyer[name].update(score_by_ER) # add to dict
                    sbyshots[name].update(score_by_shots)
                    totbyer[name].update(tot_by_ER)
                    totbyshots[name].update(tot_by_shots)
                colors[name] = color
            
                print("abbr:", abbr)
    
    if REDRAW:
        for file_name in os.listdir("saved_data"):
            if file_name.find(".pt") != -1 and file_name.find(STR) != -1:
        
                sby, sbs, tby, tbs, abbr = torch.load("saved_data/"+file_name)
                sbyer[abbr], sbyshots[abbr], totbyer[abbr], totbyshots[abbr] = sby, sbs, tby, tbs
       
    def plot(score_by_ER, score_by_shots, tot_by_ER, tot_by_shots, abbr, color):

        x = list(map(lambda x: math.log(x, 2), sorted(np.array(list(score_by_shots.keys())))))
        y = []
        for k in sorted(score_by_shots.keys()):
            y.append(score_by_shots[k]/tot_by_shots[k])
        if len(x) > 1: ax.plot(x, y, label=abbr, color=color)
        else: ax.plot(y)
        #torch.save([score_by_ER, score_by_shots, tot_by_ER, tot_by_shots, abbr], "saved_data/"+abbr+"_"+STR+"_saved_data.pt")

    for k in sbyer.keys():
        print("key:", k)
        plot(sbyer[k], sbyshots[k], totbyer[k], totbyshots[k], k, colors[k])    
        for j in sorted(sbyshots[k].keys()):
            print(k, j, ":", sbyshots[k][j], "/", totbyshots[k][j], "=", sbyshots[k][j] / totbyshots[k][j])
 
    ax.set_xlabel('Number of shots (in power of 2)', fontsize=18)
    ax.set_ylabel('Accuracy', fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # Get the legend handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Sort the handles and labels by label
    sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i].lower())  # Sort alphabetically (case-insensitive)
    handles = [handles[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1.0),prop=dict(size=13))
    plt.savefig("selected_"+suffix+"_v2color.png", bbox_inches='tight')

evaluate('free', 'right', 'random', standard_file=None, suffix=STR)
