import json
from tqdm import tqdm, trange
import evaluate
import pandas as pd

def readjson(filedir):
    queries, answers = [], []
    with open(filedir, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            queries.append(json.loads(line)['text'].strip())
            answers.append(json.loads(line)['answer'].strip())
    return queries, answers

filedir = '/data/wusiyao/chatglm/chatglm/sourcecode/AFAC/AFAC_LLM/AFAC/train_val.json'
queries, answers = readjson(filedir)
QA = pd.DataFrame([queries, answers]).T
QA.columns = ['Query', 'Answer']

from bert_score import score# data
P, R, F1 = score(queries, answers, lang="zh")

import numpy as np
F1 = np.array(F1)
ind_dirty = np.where(F1 < 0.58)[0]
dirty = QA.loc[ind_dirty]

ind_clean= np.where(F1 >= 0.58)[0]
clean = QA.loc[ind_clean]

#保存
with open('/data/wusiyao/chatglm/chatglm/sourcecode/AFAC/AFAC_LLM/AFAC/clean/train_val.json', 'w+', encoding='utf-8') as f:
    for i in trange(len(clean)):
        text = clean.iloc[i, 0]
        answer = clean.iloc[i, 1]
        f.write(json.dumps({'text':text, 'answer':answer}, ensure_ascii=False) + '\n')
# import matplotlib.pyplot as plt
# plt.figure()
# plt.boxplot(F1)
# plt.show()
# print('f')



