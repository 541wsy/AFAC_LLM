# 导入常用模块
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import json, os
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig, DataCollatorForSeq2Seq
# 配置参数
from argparse import Namespace
cfg = Namespace()

#dataset
cfg.prompt_column = 'prompt'
cfg.response_column = 'response'
cfg.history_column = None
cfg.source_prefix = '' #添加到每个prompt开头的前缀引导语

cfg.max_source_length = 128
cfg.max_target_length = 1280

#model
cfg.model_name_or_path = '/data/wusiyao/chatglm/chatglm/chatglm-6b'  #远程'THUDM/chatglm-6b'
cfg.quantization_bit = None #仅仅预测时可以选 4 or 8


#train
cfg.epochs = 100
cfg.lr = 5e-3
cfg.batch_size = 1
cfg.gradient_accumulation_steps = 16 #梯度累积


#加载ptuning后的模型
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



#定义知识样本~
keywords, descrips = [], []
with open('/data/wusiyao/chatglm/chatglm/sourcecode/AFAC/AFAC_LLM/rawdata/知识库.json', 'rb') as f:
    while True:
        line = f.readline()
        if not line:
            break
        row = json.loads(line)
        keywords.append(list(row.keys())[0])
        descrips.append('\n'.join(list(row.values())[0]))



#对prompt使用一些简单的数据增强的方法，以便更好地收敛。
def get_prompt_list(keyword):
    return [f'你知道{keyword}是什么吗？',
            f'你知道{keyword}对于基金投资意味着什么吗?',
            f'你知道如何根据{keyword}，给出基金投资意见吗？',
            f'介绍一下{keyword}',
            f'你知道{keyword}和基金投资的关系吗？',
           ]
datas = []
for keyword, descrip in zip(keywords, descrips):
    data = [{'prompt': x, 'response': descrip} for x in get_prompt_list(keyword)]
    datas.append(data)

dfdata = pd.DataFrame(data)

def preprocess(examples):
    max_seq_length = cfg.max_source_length + cfg.max_target_length
    model_inputs = {
        "input_ids": [],
        "labels": [],
    }
    for i in range(len(examples[cfg.prompt_column])):
        if examples[cfg.prompt_column][i] and examples[cfg.response_column][i]:
            query, answer = examples[cfg.prompt_column][i], examples[cfg.response_column][i]

            #history = examples[cfg.history_column][i] if cfg.history_column is not None else None
            prompt = query

            prompt = cfg.source_prefix + prompt
            a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
                                     max_length=cfg.max_source_length)
            b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,
                                     max_length=cfg.max_target_length)

            context_length = len(a_ids)
            input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
            labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]

            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
    return model_inputs
import datasets
#训练集和验证集一样
ds_train_raw = ds_val_raw = datasets.Dataset.from_pandas(dfdata)

ds_train = ds_train_raw.map(
    preprocess,
    batched=True,
    remove_columns=ds_train_raw.column_names
)

ds_val = ds_val_raw.map(
    preprocess,
    batched=True,
    remove_columns=ds_val_raw.column_names
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=None,
    label_pad_token_id=-100,
    pad_to_multiple_of=None,
    padding=False
)

dl_train = DataLoader(ds_train,batch_size = cfg.batch_size,
                      num_workers = 2, shuffle = True, collate_fn = data_collator
                     )
dl_val = DataLoader(ds_val,batch_size = cfg.batch_size,
                      num_workers = 2, shuffle = False, collate_fn = data_collator
                     )

from peft import get_peft_model, AdaLoraConfig, TaskType

#训练时节约GPU占用
model.config.use_cache=False
model.supports_gradient_checkpointing = True  #
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

peft_config = AdaLoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False,
    r=8,
    lora_alpha=32, lora_dropout=0.1,
    target_modules=["query", "value"]
)

peft_model = get_peft_model(model, peft_config)

peft_model.is_parallelizable = True
peft_model.model_parallel = True
peft_model.print_trainable_parameters()


from torchkeras import KerasModel
from accelerate import Accelerator


class StepRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage="train", metrics_dict=None,
                 optimizer=None, lr_scheduler=None
                 ):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.accelerator = accelerator if accelerator is not None else Accelerator()
        if self.stage == 'train':
            self.net.train()
        else:
            self.net.eval()

    def __call__(self, batch):

        # loss
        with self.accelerator.autocast():
            loss = self.net(input_ids=batch["input_ids"], labels=batch["labels"]).loss

        # backward()
        if self.optimizer is not None and self.stage == "train":
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        all_loss = self.accelerator.gather(loss).sum()

        # losses (or plain metrics that can be averaged)
        step_losses = {self.stage + "_loss": all_loss.item()}

        # metrics (stateful metrics)
        step_metrics = {}

        if self.stage == "train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses, step_metrics


KerasModel.StepRunner = StepRunner


# 仅仅保存lora相关的可训练参数
def save_ckpt(self, ckpt_path='/data/wusiyao/chatglm/chatglm/sourcecode/LORA/output/checkpoint', accelerator=None):
    unwrap_net = accelerator.unwrap_model(self.net)
    unwrap_net.save_pretrained(ckpt_path)


def load_ckpt(self, ckpt_path='checkpoint'):
    self.net = self.net.from_pretrained(self.net.base_model.model, ckpt_path)
    self.from_scratch = False


KerasModel.save_ckpt = save_ckpt
KerasModel.load_ckpt = load_ckpt

optimizer = torch.optim.AdamW(peft_model.parameters(), lr=cfg.lr)
keras_model = KerasModel(peft_model, loss_fn=None,
                         optimizer=optimizer)
ckpt_path = 'output/pt50'

keras_model.fit(train_data=dl_train,
                val_data=dl_val,
                epochs=30,
                patience=20,
                monitor='val_loss',
                mode='min',
                ckpt_path=ckpt_path,
                mixed_precision='fp16',
                gradient_accumulation_steps=cfg.gradient_accumulation_steps
                )
