from Bio import SeqIO
import torch
from multiprocessing import cpu_count
torch.set_num_threads(cpu_count())
import gc
import re
import numpy as np
from transformers import T5Tokenizer, T5Model,T5EncoderModel

tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50',do_lower_case=False)
model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50')

model = model.cuda()

def infer(seq):
    seq = seq
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
        # embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=None)
        embedding = model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          # decoder_input_ids=input_ids,
                          )

    # For feature extraction we recommend to use the encoder embedding
    encoder_embedding = embedding.last_hidden_state[0, :-1].detach().cpu().numpy().mean(axis=0).tolist()
    res = encoder_embedding
    return res
dim = 780
from Bio import SeqIO
def read_fa(path):
    res = []
    rx = SeqIO.parse(path,format="fasta")
    for x in list(rx):
        seq = str(x.seq)
        res.append(seq)
    return res


fa_path = "/data/neg.fa"
seq_fa = read_fa(fa_path)
res = []
for seq in seq_fa:
    print(seq)
    esm_vec = infer(seq)
    res.append(esm_vec)
res = np.array(res)
print(res.shape)
np.save("t5_neg.npy".format(dim=str(dim)),res)

fa_path = "/data/pos.fa"
seq_fa = read_fa(fa_path)
res = []
for seq in seq_fa:
    esm_vec = infer(seq)
    res.append(esm_vec)
res = np.array(res)
print(res.shape)
np.save("t5_pos.npy".format(dim=str(dim)),res)

