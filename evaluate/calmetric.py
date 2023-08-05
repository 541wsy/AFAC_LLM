# Metric

import jieba, json 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_metrics(file):
    labels, preds, queries = [], [], []
    with open(file, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            labels.append(json.loads(line)['label'])
            preds.append(json.loads(line)['pred'])
            queries.append(json.loads(line)['query'])
    rouges, bleus = [], []
    for pred, label in zip(preds, labels):
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
        rouges.append(scores[0]['rouge-1']['r'])

        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3, weights=[(1.)])
        bleus.append(bleu_score)
    return rouges, bleus

valdir = '/data/wusiyao/chatglm/chatglm/sourcecode/ptuning/output/afac-chatglm-6b-pt-128-2e-2/know/pt250_val.json'
compute_metrics(valdir)