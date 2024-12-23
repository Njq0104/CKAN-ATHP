# CKAN-ATHP
A KAN predictor based on feature augmentation and loss improvement for identifying antihypertensive peptides
# Requirements
Python >= 3.8.6

torch = 2.1.2

pandas = 2.1.4

scikit-learn = 11.0.2

ProtT5-XL-UniRef50 model, it can be downloaded at: https://huggingface.co/Rostlab/prot_t5_xl_uniref50

DistilProtBert model, it can be downloaded at: https://huggingface.co/yarongef/DistilProtBert

PubChem10M model, it can be downloaded at: https://huggingface.co/seyonec/PubChem10M_SMILES_BPE_450k
# Description
We present a model that uses three pre-trained models—DistilProtBert, PubChem10M, and Prot-t5—to encode sequences. While constructing features, we also employ a feature enhancement algorithm that expands the training samples in the feature space, thereby increasing the sample size and improving model performance in cases of insufficient samples. A Convolutional Kolmogorov-Arnold Network (CKAN) is used to build the prediction model. We also improve the loss function by combining KL divergence loss with cross-entropy loss, to prevent the model from becoming overly confident. 
# Training a new model for your needs
1. put the negative samples and positive samples with fasta format into neg.fa and pos.fa at data directory
2. Extract  feature: cd to the CKAN-ATHP/Feature_extraction, and run "python DistilProtBert.py ", "python prot-t5.py ", "python PubChem10M.py ".
3. If you need to enhance the features, cd to the CKAN-ATHP/Feature_augmentation, replace your npy file features, and modify the coefficients you want
4. running the command by "python train.py"
# Predict
Replace your dataset and feature file in the predcit.py file, then run "python predict.py"
