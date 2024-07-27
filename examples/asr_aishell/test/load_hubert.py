from fairseq.models.hubert import HubertModel
import fairseq
import torch


encoder_path = "/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/models/simpleoier_librispeech_hubert_iter0_train_ssl_torchaudiohubert_base_960h_pretrain_it0_raw/exp/hubert_iter0_train_ssl_torchaudiohubert_base_960h_pretrain_it0_raw/valid.loss.ave.pth"
#"/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/models/chinese-hubert-large/chinese-hubert-large-fairseq-ckpt.pt"
#"/root/autodl-tmp/hubert_xtralarge_ll60k_finetune_ls960.pt"
#"/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/models/chinese-hubert-large/chinese-hubert-large-fairseq-ckpt.pt"
# 加载预训练的 Hubert 模型
model = torch.load(encoder_path)

for key, value in model.items():
  print(key, "\n")
# models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([encoder_path])
# cfg.skip_nomask=True
# print("cfg.skip_nomask:", cfg.skip_nomask)
# model = models[0]
# model.skip_nomask = True
# print(model.skip_nomask)
# 确保模型正确加载并检查其属性
