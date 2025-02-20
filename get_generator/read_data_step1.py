# note: this is the second script that we run.

import json
import pandas as pd
from copy import deepcopy
def read_humanevalplus():
    df = pd.read_parquet("data/humaneval+/test-00000-of-00001-5973903632b82d40.parquet")
    df['source'] = ['humaneval+' for i in range(len(df['prompt']))]
    return {"problem": df['prompt'], "solution": df['canonical_solution'], 'source': df['source']}

def read_apps_new():
    df = pd.read_csv("data/apps_rewritten/rewrite.csv")
    for i in range(len(df['predict0'])):
        df['predict0'][i] = df['predict0'][i].replace("```python", "").replace("```", "")

    return {'problem': df['problem'], 'solution': df['predict0'], 'source': df['source']}

def read_mbppplus():
    df = pd.read_parquet("data/MBPP+/test-00000-of-00001-d5781c9c51e02795.parquet")
    print(df['prompt'])
    print(df['code'])
    df['source'] = ['mbpp+' for i in range(len(df['prompt']))]
    return {'problem': df['prompt'], 'solution': df['code'], 'source': df['source']}

def merge(a, b, c):
    print("len:", len(a['problem']), len(b['problem']), len(c['problem']))
    return {'problem': pd.concat([a['problem'], b['problem'], c['problem']]).reset_index(drop=True), 'solution': pd.concat([a['solution'], b['solution'], c['solution']]).reset_index(drop=True)}

data1 = read_humanevalplus()
data2 = read_mbppplus()
data3 = read_apps_new()

data = merge(data1, data2, data3)

print(len(data['problem']))

def construct_data(data):
    # Note: gen3 and special cases are deprecated in our work.
    prompt_part0 = "You are a coding expert. You will be provided a coding question and corresponding solution. Please write three python function that randomly generates test case for the question and 3 special test cases. Specifically:"
    prompt_part0 += "The first function's name is gen1, which generates random data (should be able to generate VERY DIVERSE, i.e., at least 1000 different data points)."
    prompt_part0 += "The second function's name is gen2, which generates data that is slightly harder than those generated in gen1. (should be able to generate at least 100 different data points)."
    prompt_part0 += "The third function's name is gen3, which generates data that does not reveal the underlying function when looking at the input and corresponding output from the solution."
    prompt_part0 += "You shall not define any function outside gen1, gen2 or gen3. Should you use any helper function, make them inner functions inside gen1, gen2 or gen3. You gen1, gen2 and gen3 function should have and only have one int parameter, which is the number of cases."
    prompt_part0 += 'Finally, the special cases should be designed as informative as possible that reveals the underlying function when looking at the input and corresponding output from the solution.'
    prompt_part0 += 'Here is an example. Note the output of gen1, gen2, gen3 should be a list of dicts describing the parameters, and your special case input should be a dict describing the parameters. Please follow the format, and do not generate cases that are too long. Do not output any other text; put all your thoughts after "# rationale:" as shown in the example.\n'
    with open("prompt_part0.txt", "r") as f:
        lines = f.readlines()
        for line in lines: prompt_part0 += line
    prompts = []
    for i in range(len(data['problem'])):
        prompt = prompt_part0 + "\n[[Problem]]" + data['problem'][i] + "\n[[Solution]]" + data['solution'][i] + "\n"
        prompts.append(prompt)
    new_data = deepcopy(data)
    new_data['prompt'] = prompts
    #print(new_data['prompt'][0])
    return new_data

data1 = pd.DataFrame(construct_data(data))
print(len(data1['prompt']))
data1.to_parquet("construct_data_final.parquet", engine='pyarrow')
# result goes into construct_data_final.parquet and then GPT-4o-0806 is called. The result will then go into 'predict0', and the output file is in result/GPT-4o-0806.json