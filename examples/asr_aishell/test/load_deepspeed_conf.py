import base64
import json


config = "conf/ds_config.json"
# 读取文件内容
with open(config, "r", encoding="utf-8") as file:
    config_content = file.read()

# 解析 JSON 内容
config_json = json.loads(config_content)
print(config_json)
if isinstance(config_json, dict):
  param_dict = config_json
else:
  param_dict = base64.urlsafe_b64decode(config_json).decode('utf-8')
print(param_dict)