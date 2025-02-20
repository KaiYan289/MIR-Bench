import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

"""
This is the fourth step, right after running each of the script in the codes_genshot folder (there is a script run.sh for this; you do not have to do it manually). 
Before that you should run build_dataset_variants_partA_step2.py first.
This script will read the data that each of the script in the data_genshot folder filled by scripts in codes_genshot, and then filter out the data that is too long or too short or too duplicated.
The list for next step is stored in shots_lim_128_4096_afterpartA.txt.
"""

tot_cnt = 0
def count_unique_dicts(dict_list):
    new_list = [str(x) for x in dict_list]
    return len(list(set(new_list)))

data_old = pd.read_parquet("../get_generator/construct_data_final.parquet")

def get_source(x): # 164, 378, 2456
    if x < 164:
        return "humaneval+"
    elif x < 164 + 378:
        return "mbpp+"
    else:
        return "apps"

LIM, VAR_ANS = 128, 4096
f = open("shots_lim_"+str(LIM)+"_"+str(VAR_ANS)+"_afterpartA.txt", "w")
for i in tqdm(range(2998)):
    if i == 2047: continue
    try:
        data1 = torch.load("data_genshot/shots_"+str(i)+"_trainset.pt")
        data2 = torch.load("data_genshot/shots_"+str(i)+"_testset.pt")
        data3 = torch.load("data_genshot/shots_"+str(i)+"_confusing_trainset.pt") # deprecated
    except:
        print(i, "read error!")
        continue
    if data1 is None or data2 is None or data3 is None:
        print(i, "abandon for none dataset!")
        continue
    if len(data1) < 20000 or len(data2) < 10 or len(data3) < 20000:
        print(i)
        print(len(data1), len(data2), len(data3))
        print(i, "abandon for incomplete data!")
        continue
    data1_unq = count_unique_dicts(data1)
    data2_unq = count_unique_dicts(data2)
    data12_unq = count_unique_dicts(data1 + data2)
    if not (data1_unq >= VAR_ANS and data2_unq == len(data2) and data12_unq == data1_unq + data2_unq):
        print(i, "abandon for duplicated data!")
        continue
    mx_len1 = max(np.array([len(str(data1[i])) for i in range(len(data1))]))
    mx_len2 = max(np.array([len(str(data2[i])) for i in range(len(data2))]))
    if max(mx_len1, mx_len2) > LIM:
        print(i, "abandon for too long data!")
        continue
    # print(list(set(data1)))
    print("selected!", i, get_source(i), tot_cnt)
    tot_cnt += 1
    # exit(0)
    f.write(str(i)+" "+get_source(i)+"\n")
f.close()