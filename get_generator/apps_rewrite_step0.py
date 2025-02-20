# Note: This is the first script that we run.

import json
import pandas as pd
def read_apps():
    df = pd.read_json("data/apps/train.jsonl", lines=True)
    df = df[df['difficulty']=='introductory'].reset_index(drop=True)
    #lens = [len(json.loads(df['solutions'][i])) for i in range(len(df['question']))]
    #print("len:", lens)
    solutions = [json.loads(df['solutions'][i])[0] for i in range(len(df['question']))]
    idxs, stop_words = [], ['test cases', 'number of queries', 'Each input data', 'each input data']
    for i in range(len(df['question'])):
        flag = 1
        for stop_word in stop_words:
            if df['question'][i].find(stop_word) != -1: 
                flag = 0
                break
        if solutions[i].find("input(") != -1:
            flag = 0
        if flag:
            idxs.append(i)
    df = df.iloc[idxs].reset_index(drop=True)
    solutions = [solutions[i] for i in idxs]
    print(len(solutions))
    #for i in range(len(df['question'])): 
    #    print(df['question'][i], solutions[i])
    df['new_solution'] = solutions
    df['source'] = ['apps' for i in range(len(df['question']))]
    # print("solution len:", [type(df['solutions'][i]) for i in range(len(df['question']))])
    return {'problem': df['question'], 'solution': df['new_solution'], 'source': df['source']}

data = read_apps()
prompts = []
for i in range(len(data['problem'])):
    prompt = "You are a coding expert. You will be given a problem and corresponding solution. Rewrite the solution such that:\n"
    prompt += "1. It becomes a single function named 'solution', which takes parameters as input instead of reading from input() function if there is any;\n"
    prompt += "2. There is no code out of the solution function and no solution class. All auxiliary functions should be defined inside the solution function, and all imports should also be in the function.\n"
    prompt += "3. The solution function should not have any print() function. Instead, it should return the result of the function. If you need to output any rationale, leave them in comments. Your output must be directly runnable without any change.\n"
    prompt += "4. Just output the rewritten function; do not test it with extra statements."
    prompt += "Here is an example:"
    with open("rewrite_part0.txt", "r") as f:
        lines = f.readlines()
        for line in lines: prompt += line
    prompt += "[[Problem]]\n" + data['problem'][i] + '\n[[Solution]]\n' + data['solution'][i] + '\n[[Rewrite]]'
    prompts.append(prompt)
data['prompt'] = prompts
pd.DataFrame(data).to_parquet("apps_rewrite.parquet", engine='pyarrow')

# note: the output of this parquet goes into data/apps_rewritten/rewrite.csv.