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
# Docker Image Usage Guide
This project is packaged as a Docker image and published on Docker Hub, allowing users to run predictions without manually configuring the environment.
## Pull the Image
docker pull njq0104/ckan-athp_predictor:latest
## Using the Prediction Function
To run the prediction interactively, use the following command:  
docker run -it --rm njq0104/ckan-athp_predictor:latest
## You will be prompted to enter peptide sequences, one per line. For example:
```
Enter multiple amino acid sequences (one per line). Type 'exit' and press Enter to finish:
KLLKELKKLLKLLK
VDHDHPE
DIGYY
FACRRWQWRMAALGA
exit  
Prediction Results:
Sequence: KLLKELKKLLKLLK  | Predicted Label: 0 | Probability: [1.0, 0.0]
Sequence: VDHDHPE         | Predicted Label: 0 | Probability: [0.95, 0.05]
Sequence: DIGYY           | Predicted Label: 1 | Probability: [0.35, 0.65]
Sequence: FACRRWQWRMAALGA | Predicted Label: 0 | Probability: [0.91, 0.09]
```
The model will process the input sequences and return predicted results (such as labels or probabilities).
