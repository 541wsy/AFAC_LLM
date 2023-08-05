import os
import pickle
import sys
import json
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig
)

from prompt_knowledge import DocumentService, PROMPT, DOCS_PATH, VS_PATH
from tqdm import trange, tqdm
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

CKP = '/data/wusiyao/chatglm/chatglm/sourcecode/ptuning/output/afac-chatglm-6b-pt-128-2e-2/know/checkpoint-250'
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


datas = []

with open('/data/wusiyao/chatglm/chatglm/sourcecode/AFAC/AFAC_LLM/AFAC/val.json', 'rb') as f:
    while True:
        line = f.readline()
        if not line:
            break
        datas.append(json.loads(line))

with open('/data/wusiyao/chatglm/chatglm/sourcecode/ptuning/output/afac-chatglm-6b-pt-128-2e-2/know/pt250_val.json', 'w+', encoding='utf-8') as f:

    for data in tqdm(datas):
        query = data['text']
        label = data['answer']

        prompt = knowledge_prompt(query)
        response, _ = model.chat(tokenizer, prompt, history=[], temperature=1.0)
        row = {'query':query, 'label':label, "pred":response.strip()}
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


