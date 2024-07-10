#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
# export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

run_dir=/root/autodl-tmp/SLAM-LLM
cd $run_dir
code_dir=examples/asr_aishell

speech_encoder_path=/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/models/chinese-hubert-large/chinese-hubert-large-fairseq-ckpt.pt
llm_path=/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/models/Qwen2-7B

output_dir=/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/exp/qwen2-7b-aishell-linear-steplrwarmupkeep1e-4-hubert-20240709-deepspeed
ckpt_path=$output_dir/asr_epoch_1_step_8000
split=test_data
val_data_path=/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/data/${split}.jsonl
decode_log=$ckpt_path/decode_${split}_beam4

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/inference_asr_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name="qwen2-7b" \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=3584 \
        ++model_config.encoder_name=hubert \
        ++model_config.normalize=true \
        ++dataset_config.normalize=true \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=1024 \
        ++model_config.encoder_type=pretrain \
        ++model_config.encoder_projector=linear \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=raw \
        ++dataset_config.inference_mode=true \
        ++dataset_config.prompt=" speech transcribe. " \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=0 \
        ++train_config.output_dir=$output_dir \
        ++train_config.quantization=true \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt \
        # ++peft_ckpt=$ckpt_path \
        # ++train_config.use_peft=true \
        # ++train_config.peft_config.r=32 \
        # ++dataset_config.normalize=true \
        # ++model_config.encoder_projector=q-former \
        # ++dataset_config.fix_length_audio=64 \
