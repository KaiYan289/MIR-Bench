import pandas as pd
import numpy as np
# This is the script for evaluating any dataset in our benchmark with our rule of exact match.

# This is an provided example
DATA_NAME = "data/MIR-Core.parquet" # your name of dataset. It should be ready in the same folder ./eval.
OUTPUT_FILE = "raw_output/GPT4o-0806_lliwt2drxg676152ba/manyshot_benchmark#64acc_quad_512.json" # your output file name. It should also be put in the same folder ./eval.

merged_data = pd.read_parquet(DATA_NAME)
# This script assumes the order of data in merged_data, which is your dataset, is different from your output file. If they are the same, nothing else needs to be done. 
idx_merged_data = {}
for i in range(len(merged_data['prompt'])):
    idx_merged_data[merged_data['prompt'][i]] = i
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
        else: # cope with {"ans": 3} vs. 3
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

def eval_single(name):

    res = pd.read_json(name)
    print(res.keys())
    lst_short_gpt4 = []
    tot_by_ER, score_by_ER = {}, {}
    tot_by_shots, score_by_shots = {}, {}
    chelect = []
        
    for i in range(len(res['prompt'])):
        for seed in range(1):
            idx_in_merged = idx_merged_data[res['prompt'][i]]
            error_rate, plen, num_shots = merged_data['ER_rate'][idx_in_merged], merged_data['plen'][idx_in_merged], merged_data['prompt'][idx_in_merged].count('Input:') - 1
            train_idx = merged_data['idx_train'][idx_in_merged]

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
        print("seed:", seed, "error rate:", k, "score:", score_by_ER[k][seed], "/", tot_by_ER[k][seed], "=", score_by_ER[k][seed] / tot_by_ER[k][seed])
    for k in sorted(tot_by_shots.keys()):
        print("seed:", seed, "shots:", k, "score:", score_by_shots[k][seed], "/", tot_by_shots[k][seed], "=", score_by_shots[k][seed] / tot_by_shots[k][seed])

    return score_by_ER, score_by_shots, tot_by_ER, tot_by_shots
    

score_by_ER, score_by_shots, tot_by_ER, tot_by_shots = eval_single(OUTPUT_FILE)
acc_by_ER, acc_by_shots = {}, {}
for k in score_by_ER.keys():
    acc_by_ER[k] = score_by_ER[k] / tot_by_ER[k]
for k in score_by_shots.keys():
    acc_by_shots[k] = score_by_shots[k] / tot_by_shots[k]
print("accuracy by error rate:", acc_by_ER)
print("accuacy by shots:", acc_by_shots)