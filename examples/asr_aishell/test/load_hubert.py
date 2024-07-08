from fairseq.models.hubert import HubertModel

# 加载预训练的 Hubert 模型
model = HubertModel.from_pretrained('/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/models/chinese-hubert-large.pt')

# 确保模型正确加载并检查其属性
print(dir(model))
