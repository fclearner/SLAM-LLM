import torch

model_path='/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/models/chinese-hubert-large/chinese-hubert-large-fairseq-ckpt.pt'
# 加载模型
# model = torch.load(model_path)


# import torch
# import torch.nn.functional as F
# import soundfile as sf

# from transformers import (
#     Wav2Vec2FeatureExtractor,
#     HubertModel,
# )


# # model_path=""
# # wav_path=""

# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
# model = HubertModel.from_pretrained(model_path)

from fairseq import checkpoint_utils

models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
print("loaded model(s) from {}".format(model_path))
model = models[0]
# # 打印模型参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")