import safetensors.torch
import numpy as np

import json
import struct
import numpy as np

def read_safetensors(filename):
    with open(filename, 'rb') as f:
        # 读取头部长度
        header_length = struct.unpack('I', f.read(4))[0]
        
        # 读取头部数据
        header_data = f.read(header_length)
        header = json.loads(header_data)
        
        # 读取张量数据
        tensors = {}
        for name, meta in header.items():
            dtype = np.dtype(meta['dtype'])
            shape = meta['shape']
            offset = meta['data_offsets'][0]
            length = meta['data_offsets'][1] - offset
            
            f.seek(offset)
            tensor_data = f.read(length)
            tensor = np.frombuffer(tensor_data, dtype=dtype).reshape(shape)
            tensors[name] = tensor
        
        return tensors


if __name__ == "__main__":
    file_path = "/root/autodl-tmp/SLAM-LLM/examples/asr_aishell/models/Qwen2-7B/model-00001-of-00004.safetensors"  # 替换为你的 safetensors 文件路径
    # 使用示例
    filename = 'example.safetensors'
    tensors = read_safetensors(file_path)
    for name, tensor in tensors.items():
        print(f"Tensor name: {name}, shape: {tensor.shape}, dtype: {tensor.dtype}")
