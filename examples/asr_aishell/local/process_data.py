import argparse
import json
import logging

from pathlib import Path


def process_data(data_folder, json_out_path):
  data_dict = {}

  with open(data_folder / "text", encoding="utf-8") as f_txt:
    for l in f_txt.readlines():
      key, text = l.strip().split(" ", maxsplit = 1)
      if key not in data_dict.keys():
        data_dict[key] = [text]

  with open(data_folder / "wav.scp", encoding="utf-8") as f_scp:
    for l in f_scp.readlines():
      key, wav_path = l.strip().split(" ", maxsplit = 1)
      if key not in data_dict.keys():
        logging.warning(f"{key} not in wav.scp")
      data_dict[key].append(wav_path)

  with open(json_out_path, 'w', encoding="utf-8") as f:
    for key in data_dict:
        data = {
            "key": f"{key}_ASR",
            "source": data_dict[key][1],
            "target": data_dict[key][0].replace(" ", ""),
        }
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process kaldi-format data to jsonl')
    parser.add_argument('kaldi_folder',
                        type=Path,
                        default=None,
                        help='kaldi-format_data_folder')
    parser.add_argument('json_out_path',
                        type=str,
                        default="data.jsonl",
                        help='json data out file path')
    args = parser.parse_args()
    process_data(args.kaldi_folder, args.json_out_path)
    print(f"{args.kaldi_folder} json convert finished, out to {args.json_out_path}")
