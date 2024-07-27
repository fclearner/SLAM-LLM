import argparse
import torch
import json


def main():
    parser = argparse.ArgumentParser(description='filter out unused module')
    parser.add_argument('input_ckpt',
                        type=str,
                        help='original checkpoint')
    parser.add_argument('output_ckpt',
                        type=str,
                        help='modified checkpoint')
    args = parser.parse_args()

    state = torch.load(args.input_ckpt, map_location="cpu")
    new_state = {}

    for key, value in state.items():
      # 如果key以'encoder.pretrain_hubert_model.w2vmodel.'开头，则替换前缀
      if key.startswith('encoder.hubert_pretrain_model.'):
          new_key = key.replace('encoder.hubert_pretrain_model.', 'hubert_pretrain_model.')
      elif key.startswith('encoder.global_step'):
          new_key = key.replace('encoder.global_step', 'global_step')
      else:
          new_key = key
      # 将新的key-value对添加到新字典中
      new_state[new_key] = value

    torch.save(new_state, args.output_ckpt)


if __name__ == '__main__':
    main()