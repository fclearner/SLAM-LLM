from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel, AutoModelForSeq2SeqLM, T5ForConditionalGeneration


llm_path = '/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/models/Qwen2-7B'
tokenizer = AutoTokenizer.from_pretrained(llm_path)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.bos_token_id = tokenizer.eos_token_id
print(tokenizer)
print(tokenizer.bos_token_id)