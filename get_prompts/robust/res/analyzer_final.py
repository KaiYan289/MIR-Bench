import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import ast
import math
import torch
"""
This is the code for evaluating results for LLM many-shot robustness part.

"""
STR = "aware1"
FILTER = "64shot"

assert STR in ['aware1', 'aware2', 'unaware'], "Error!"
assert FILTER in "64shot", "256shot", "1024shot", "Error!"

merged_data = pd.read_parquet("../data/build_data_free_"+STR+"_wrong_random_64acc-quad-formal.parquet")
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
            # print("verdict:", verdict, "ans:", ans)
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
            # print(idx_in_original)
            idx_in_merged = idx_merged_data[res['prompt'][i]]
            error_rate, plen, num_shots = merged_data['ER_rate'][idx_in_merged], merged_data['plen'][idx_in_merged], merged_data['num_shots'][idx_in_merged]
            
            if FILTER == "64shot" and num_shots != 64: continue
            elif FILTER == "256shot" and num_shots != 256: continue
            elif FILTER == "1024shot" and num_shots != 1024: continue
            
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
            #if num_shots == 4096 and res['predict0'][i].find("Error code") == -1 and abbr.find("gpt4o-0806") != -1:
            #    lst_short_gpt4.append(train_idx)

            #if num_shots >= threshold_shot or res['predict0'][i].find("Error code") != -1: 
                
                #if num_shots < 1024 and res['predict0'][i].find("Error code") != -1: 
                #    print("error!", res['predict0'][i])
            #    continue
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

    REDRAW = True # draw from scratch
    
    for root, d, files in os.walk(STR, topdown=False):
        for file_name in files:
            print("file name:", root, d, file_name)
            # exit(0)
            abbr = root.split("/")[-1]
            loc = abbr.rfind("_")
            abbr = abbr[:loc]
            print("abbr:", abbr)
            if abbr.find("Claude-35-sonnet") != -1: name, color = 'Claude-3.5-Sonnet', 'green' # 'red'
            elif abbr.find("Claude-3-sonnet") != -1: name, color = 'Claude-3-Sonnet', 'lightgreen' # 'gold'
            elif abbr.find("Gemini-1.5_flash-002") != -1: name, color = 'Gemini-1.5 Flash-002', 'lightblue'
            elif abbr.find("Gemini-1.5_pro-002") != -1: name, color = 'Gemini-1.5 Pro-002', 'blue'
            elif abbr.find("Gemini-2.0") != -1: name, color = 'Gemini-2.0 Flash', 'darkblue'
            elif abbr.find("GPT4o-0806") != -1: name, color = 'GPT4o-0806', 'red' # 'green'
            elif abbr.find("GPT4o-mini-0718") != -1: name, color = 'GPT4o-mini-0718', 'orange' # 'lightgreen'
            elif abbr.find("Mistral-large-2") != -1: name, color = 'Mistral-Large-2', 'purple'
            elif abbr.find("OpenAI-o1-mini") != -1: name, color = 'OpenAI-o1-mini', 'black'
            elif abbr.find("OpenAI-o1-preview") != -1: name, color = 'OpenAI-o1-preview', 'grey'
            elif abbr.find("Claude-3-haiku") != -1: name, color = 'Claude-3-Haiku', 'gold' # 'orange'
            elif abbr.find("Claude-35-haiku") != -1: name, color = 'Claude-3.5-Haiku', 'salmon'
            elif abbr.find('Moonshot-128k') != -1: name, color = 'Moonshot-128k', 'brown'
            elif abbr.find("Qwen2-72B") != -1: name, color = 'Qwen2-72B-Instruct', 'pink'
            elif abbr.find("GLM-4-plus") != -1: name, color = 'GLM-4-Plus', 'teal'
            else: continue
            colors[name] = color
            colors['Mistral-Large-2'] = 'purple'
            if (not REDRAW) and os.path.exists("saved_data_"+FILTER+"/"+name+"_"+STR+"_saved_data.pt"):
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
    
    if not REDRAW:
        for file_name in os.listdir("saved_data_"+FILTER):
            if file_name.find(".pt") != -1 and file_name.find(STR) != -1:
                sby, sbs, tby, tbs, abbr = torch.load("saved_data_"+FILTER+"/"+file_name)
                print("abbr:::", abbr)
                sbyer[abbr], sbyshots[abbr], totbyer[abbr], totbyshots[abbr] = sby, sbs, tby, tbs
    def plot(score_by_ER, score_by_shots, tot_by_ER, tot_by_shots, abbr, color):
        
        x = list(map(lambda x: math.log(x, 2), sorted(np.array(list(score_by_ER.keys())))))
        y = []
        for k in sorted(score_by_ER.keys()):
            y.append(score_by_ER[k]/tot_by_ER[k])
        if len(x) > 1: 
            ax.plot(x, y, label=abbr, color=color)
            print("draw!", abbr, color)
        else: ax.plot(x, y)
        # ax[1].plot([0], [0], label=abbr, color=color)
        torch.save([score_by_ER, score_by_shots, tot_by_ER, tot_by_shots, abbr], "saved_data_"+FILTER+"/"+abbr+"_"+STR+"_saved_data.pt")

    for k1 in sbyer.keys():
        k = k1.replace('large', 'Large')
        print("key:", k, 'colors:', colors[k])
        plot(sbyer[k], sbyshots[k], totbyer[k], totbyshots[k], k, colors[k])    

    ax.set_xlabel('Number of shots (in power of 2)', fontsize=18)
    ax.set_ylabel('Accuracy', fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # Get the legend handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Sort the handles and labels by label
    sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i].lower())  # Sort alphabetically (case-insensitive)
    handles = [handles[i] for i in sorted_indices]
    labels = [labels[i].replace('large', 'Large') for i in sorted_indices]

    if STR == 'aware2': 
        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1.0),prop=dict(size=13))
    plt.savefig("selected_"+suffix+"_"+FILTER+".pdf", bbox_inches='tight')

evaluate('free', 'right', 'random', standard_file=None, suffix=STR)
