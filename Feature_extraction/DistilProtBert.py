from Bio import SeqIO
import torch
from multiprocessing import cpu_count
torch.set_num_threads(cpu_count())
import gc
import re
import numpy as np

from transformers import BertModel, BertTokenizer
import re

# 指定本地模型的路径
local_model_path = "your pretrain model path"  # 将此路径修改为您本地模型文件所在的实际路径

# 加载本地的分词器和模型
tokenizer = BertTokenizer.from_pretrained(local_model_path, do_lower_case=False)
model = BertModel.from_pretrained(local_model_path)

model = model.cuda()

def infer(seq):
    sequences_Example = [" ".join(list(seq))]
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
    ids = tokenizer.batch_encode_plus(sequences_Example,
                                      add_special_tokens=True,
                                      padding=False)

    input_ids = torch.tensor(ids['input_ids'])
    attention_mask = torch.tensor(ids['attention_mask'])

    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()

    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)

    # For feature extraction we recommend using the encoder embedding
    encoder_embedding = embedding.last_hidden_state[0, :-1].detach().cpu().numpy().mean(axis=0).tolist()
    res = encoder_embedding
    return res

dim = 780

def read_fa(path):
    res = []
    rx = SeqIO.parse(path, format="fasta")
    for x in list(rx):
        seq = str(x.seq)
        res.append(seq)
    return res
#
# 处理负样本
fa_path = "../neg.fa"
seq_fa = read_fa(fa_path)
res = []
for seq in seq_fa:
    print(seq)
    esm_vec = infer(seq)
    res.append(esm_vec)
res = np.array(res)
print(res.shape)
np.save(f"DistilProtBert_neg.npy", res)

# 处理正样本
fa_path = "../PubChem10M/pos_test.fa"
seq_fa = read_fa(fa_path)
res = []
for seq in seq_fa:
    esm_vec = infer(seq)
    res.append(esm_vec)
res = np.array(res)
print(res.shape)
np.save(f"DistilProtBert_pos.npy", res)
