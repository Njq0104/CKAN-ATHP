from Bio import SeqIO
from rdkit import Chem
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer, FeatureExtractionPipeline
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import lightgbm as lgb

torch.set_num_threads(40)


def read_fa(path):
    res = []
    rx = SeqIO.parse(path, format="fasta")
    for x in list(rx):
        seq = str(x.seq)
        res.append(seq)
    return res


model = AutoModel.from_pretrained("D:\pretrain\PubChem10M_SMILES_BPE_450k")
tokenizer = AutoTokenizer.from_pretrained("D:\pretrain\PubChem10M_SMILES_BPE_450k")
bert_model = FeatureExtractionPipeline(model=model, tokenizer=tokenizer, return_tensors=True, device=-1)

# encoder_embedding = bert_model(smile).detach().cpu().numpy()[0].sum(axis=0)

fa_path = "../neg.fa"
seq_fa = read_fa(fa_path)
neg_data = []
for seq in seq_fa:
    m = Chem.MolFromFASTA(seq)
    ss = Chem.MolToSmiles(m)[0:512]
    smiles = [ss]
    res = bert_model(smiles)[0].cpu().numpy().squeeze().mean(axis=0)
    print(res.shape)
    neg_data.append(res)
neg_data = np.array(neg_data)
print("neg data", neg_data.shape)
np.save("PubChem10M_negative.npy", neg_data)

fa_path = "../PubChem10M/pos_test.fa"
seq_fa = read_fa(fa_path)
pos_data = []
for seq in seq_fa:
    m = Chem.MolFromFASTA(seq)
    ss = Chem.MolToSmiles(m)[0:512]
    smiles = [ss]
    res = bert_model(smiles)[0].cpu().numpy().squeeze().mean(axis=0)
    pos_data.append(res)
pos_data = np.array(pos_data)
print("pos data", pos_data.shape)
np.save("PubChem10M_test.npy", pos_data)
