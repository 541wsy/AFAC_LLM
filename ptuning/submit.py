import os
import pickle
import sys
import json
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig
)

from prompt_knowledge import DocumentService, PROMPT, DOCS_PATH, VS_PATH
from tqdm import trange
def knowledge_prompt(query, topk=1):
    #根据query召回
    related_text = VS.similarity_search_with_score(query, k=topk)[0][0].page_content
    prompt_know = PROMPT.replace('{knowledge}', related_text).replace("{info}", query)
    return prompt_know

#定义全局的本地知识库
KNOW = DocumentService(DOCS_PATH, VS_PATH)
#KNOW.init_source_vector()
KNOW.load_vector_store() #加载向量库
VS = KNOW.vector_store

CKP = '/data/wusiyao/chatglm/chatglm/sourcecode/ptuning/output/afac-chatglm-6b-pt-128-2e-2/know/checkpoint-50'
MODELDIR = '/data/wusiyao/chatglm/chatglm/chatglm-6b'
config = AutoConfig.from_pretrained(MODELDIR, trust_remote_code=True)
config.pre_seq_len = 128
model = AutoModel.from_pretrained(MODELDIR, config=config, trust_remote_code=True).cuda().half()
prefix_state_dict = torch.load(os.path.join(CKP, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained(MODELDIR, trust_remote_code=True)
model = model.eval()


queries = []

with open('/data/wusiyao/chatglm/chatglm/sourcecode/AFAC/AFAC_LLM/AFAC/test.json', 'rb') as f:
    while True:
        line = f.readline()
        if not line:
            break
        queries.append(json.loads(line)['text'])

with open('/data/wusiyao/chatglm/chatglm/sourcecode/ptuning/output/afac-chatglm-6b-pt-128-2e-2/know/pt50_temp05.json', 'w+', encoding='utf-8') as f:

    for i in trange(600):
        zhi_q = queries[i*2]
        lihao_q = queries[i*2 + 1]

        prompt_zhi = knowledge_prompt(zhi_q)
        prompt_lihao = knowledge_prompt(lihao_q)
        result_info, result_event = [], []
        print('inforesult----\n')
        for _ in range(5):
            response, _ = model.chat(tokenizer, prompt_zhi, history=[], temperature=0.5)
            print(response + '\n')
            result_info.append(response)
        print('eventresult----\n')
        for _ in range(5):
            response, _ = model.chat(tokenizer, prompt_lihao, history=[], temperature=0.5)
            print(response)
            result_event.append(response)

        row = {'id':str(i), "info_result":result_info, "event_result":result_event}
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


