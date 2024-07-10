#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
# export PYTHONPATH=/root/fairseq:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=6,7
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

run_dir=/root/autodl-tmp/SLAM-LLM
cd $run_dir
code_dir=examples/asr_aishell

speech_encoder_path=/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/models/chinese-hubert-large/chinese-hubert-large-fairseq-ckpt.pt
#/root/autodl-tmp/hubert_xtralarge_ll60k_finetune_ls960.pt
#/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/models/chinese-hubert-large/chinese-hubert-large-fairseq-ckpt.pt
#/root/autodl-tmp/hubert_xtralarge_ll60k_finetune_ls960.pt

llm_path=/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/models/MiniCPM-2B-dpo-fp16/
#/root/autodl-tmp/Vicuna-7B
#/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/models/Qwen2-7B/ 3584

output_dir=/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/exp/minicpm-2b-dpo-fp16-aishell-linear-steplrwarmupkeep1e-4-hubert-$(date +"%Y%m%d")-deepspeed

mkdir -p $output_dir/tensorboard

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=minicpm-2b-dpo-fp16 \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=2304 \
++model_config.encoder_name=hubert \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1024 \
++model_config.encoder_projector=linear \
++model_config.encoder_type=pretrain \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/data/train_data.jsonl \
++dataset_config.val_data_path=/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/data/dev_data.jsonl \
++dataset_config.input_type=raw \
++train_config.model_name=asr \
++train_config.num_epochs=3 \
++train_config.enable_deepspeed=true \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=2000 \
++train_config.batch_size_training=6 \
++train_config.val_batch_size=4 \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++log_config.tensorboard_log_dir=$output_dir/tensorboard \
++metric=acc \
"
# ++train_config.use_peft=true \
# ++train_config.peft_config.r=32 \
# ++model_config.encoder_projector=linear \
# ++model_config.encoder_projector_ds_rate=5 \
# ++train_config.peft_config.peft_method=lora \
# --peft_ckpt "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-prompt-padding30-20231228/asr/4" \
# --ckpt_path "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-prompt-padding30-20231228/asr/4/model.pt" \
#++log_config.log_file=/$output_dir/train.log \
#++log_config.use_wandb=true \
#++log_config.wandb_dir=$output_dir \
#++log_config.wandb_entity_name=zym22 \
#++log_config.wandb_project_name=slam-llm \
#++log_config.wandb_exp_name=${0##*/%.*} \
#++log_config.log_interval 5 \

# -m debugpy --listen 5678 --wait-for-client
# if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
#     python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_asr.py \
#         --config-path "conf" \
#         --config-name "prompt.yaml" \
#         $hydra_args
# else
# torchrun \
#     --nnodes 1 \
#     --nproc_per_node 1 \
#     --master_port=29503 \
#     $code_dir/finetune_asr.py \
#     --config-path "conf" \
#     --config-name "prompt.yaml" \
#     ++train_config.enable_fsdp=true \
#     ++train_config.enable_ddp=true \
#     ++train_config.use_fp16=true \
#     $hydra_args
# fi

deepspeed \
    --include localhost:0 \
    $code_dir/deepspeed_finetune_asr.py \
    $hydra_args
    # --num_gpus=1 \
    # --num_nodes=1 \

# -m debugpy --listen 5678 --wait-for-client
# if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
#     python -m debugpy --listen 5678 --wait-for-client finetune_asr.py \
#         $hydra_args
# else
#     deepspeed \
#         --num_nodes=1 \
#         --include localhost:6,7 \
#         --master_port=29502 \
#         $code_dir/deepspeed_finetune_asr.py \
#         $hydra_args
#         # --num_gpus=2 \
# fi
