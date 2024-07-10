from transformers import AutoModelForCausalLM

# 加载模型
model = AutoModelForCausalLM.from_pretrained("your_model_name", load_in_8bit=True)

# 打印模型的总参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
