PRE_SEQ_LEN=128
LR=2e-2

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --train_file /data/wusiyao/chatglm/chatglm/sourcecode/ptuning/AFAC/clean/train.json \
    --validation_file /data/wusiyao/chatglm/chatglm/sourcecode/ptuning/AFAC/val.json \
    --prompt_column text \
    --response_column answer \
    --overwrite_cache \
    --model_name_or_path /data/wusiyao/chatglm/chatglm/chatglm-6b \
    --output_dir /data/wusiyao/chatglm/chatglm/sourcecode/ptuning/output/afac-chatglm-6b-pt-128-2e-2 \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 70 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 2e-2 \
    --pre_seq_len $PRE_SEQ_LEN \
