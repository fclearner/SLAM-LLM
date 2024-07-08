## data prep
## 只是做了个整合，建议分开运行

# tar -xvzf data_aishell.gz

cd aishell/data_aishell/wav
for wav in ./*.tar.gz; do
    echo "Extracting wav from $wav"
    tar -zxf $wav && rm $wav
done

local/aishell_data_prep.sh wav/ transcript/

python3 local/process_data.py data/dev data/dev_data.jsonl
python3 local/process_data.py data/test data/test_data.jsonl
python3 local/process_data.py data/train data/train_data.jsonl
