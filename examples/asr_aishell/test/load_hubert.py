from fairseq.models.hubert import HubertModel
import fairseq


encoder_path = "/root/autodl-tmp/hubert_xtralarge_ll60k_finetune_ls960.pt"
#"/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/models/chinese-hubert-large/chinese-hubert-large-fairseq-ckpt.pt"
#"/root/autodl-tmp/hubert_xtralarge_ll60k_finetune_ls960.pt"
#"/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/models/chinese-hubert-large/chinese-hubert-large-fairseq-ckpt.pt"
# 加载预训练的 Hubert 模型
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([encoder_path])
cfg.skip_nomask=True
print("cfg.skip_nomask:", cfg.skip_nomask)
model = models[0]
model.skip_nomask = True
# print(model.skip_nomask)
# 确保模型正确加载并检查其属性
