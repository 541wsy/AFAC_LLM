

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_predict \
    --test_file /data/wusiyao/chatglm/chatglm/sourcecode/ptuning/AFAC/test.json \
    --overwrite_cache \
    --prompt_column text \
    --response_column answer \
    --model_name_or_path /data/wusiyao/chatglm/chatglm/chatglm-6b \
    --ptuning_checkpoint /data/wusiyao/chatglm/chatglm/sourcecode/ptuning/output/afac-chatglm-6b-pt-128-2e-2/checkpoint-3000 \
    --output_dir /data/wusiyao/chatglm/chatglm/sourcecode/ptuning/output/afac-chatglm-6b-pt-128-2e-2/checkpoint-3000-result \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 70\
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len 128\
