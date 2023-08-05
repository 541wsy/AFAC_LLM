import transformers, torch, os
from transformers import AutoModel, AutoTokenizer, AutoConfig, DataCollatorForSeq2Seq
#加载ptuning后的模型
CKP = '/data/wusiyao/chatglm/chatglm/sourcecode/ptuning/output/afac-chatglm-6b-pt-128-2e-2/know/checkpoint-50'
MODELDIR = '/data/wusiyao/chatglm/chatglm/chatglm-6b'
config = AutoConfig.from_pretrained(MODELDIR, trust_remote_code=True)
config.pre_seq_len = 128

device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained(MODELDIR, trust_remote_code=True)

model = AutoModel.from_pretrained(MODELDIR, config=config, trust_remote_code=True).cuda().half()
prefix_state_dict = torch.load(os.path.join(CKP, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

from peft import PeftModel
ckpt_path = '/data/wusiyao/chatglm/chatglm/sourcecode/LORA/output/pt50'

peft_loaded = PeftModel.from_pretrained(model,ckpt_path).cuda()
model_new = peft_loaded.merge_and_unload() #合并lora权重
response, _ = model.chat(tokenizer, '估值是什么？')
response_new, _ = model_new.chat(tokenizer, '估值是什么？')
print(response)
print('\n')
print(response_new)
print('f')