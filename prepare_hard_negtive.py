# %%
import pandas as pd
from tqdm import tqdm
raw_data = pd.read_csv('/home/yxlu/LearningEquality/train_top100_pos_0.99081.csv')
data = {}
for topics_ids, content_ids, topics_info, content_info, target, fold in tqdm(raw_data[['topics_ids', 'content_ids', 'topics_info', 'content_info', 'target', 'fold']].values, total=len(raw_data['topics_ids'])):
    if fold != 0:
        continue
    if topics_info not in data:
        data[topics_info] = {
            'pos':[], 
            'neg':[]
        }
    if target != 0:
        if content_info in data[topics_info]['pos']:
            continue
        if len(data[topics_info]['pos']) >= 80:
            data[topics_info]['pos'].pop(0)
        data[topics_info]['pos'].append(content_info)
    else:
        if len(data[topics_info]['neg']) + len(data[topics_info]['pos']) >= 100:
            continue
        if content_info not in data[topics_info]['neg']:
            data[topics_info]['neg'].append(content_info)

rand_content = []
import random
for content_info in tqdm(raw_data['content_info'].values):
    if random.random() < 4e-06:
        rand_content.append(content_info)
print(len(rand_content))

import numpy as np
for k in data.keys():
    if len(data[k]['neg']) < 20:
        sup = rand_content
        np.random.shuffle(sup)
        data[k]['neg'] += sup[:20 - len(data[k]['neg'])]
    if len(data[k]['pos']) + len(data[k]['neg']) > 100:
        data[k]['neg'] = data[k]['neg'][:100 - len(data[k]['pos'])]
count = 0
for k in data.keys():
    if len(data[k]['neg'])  < 20:
        # print(len(data[k]['pos']))
        count += 1
        # break
count
import json 
with open('all_listwise_test.data', 'w') as f:
    save = {}
    for k in data.keys():
        save[k] = data[k]
        save = json.dumps(save)
        f.write(save + '\n')
        save = {}




