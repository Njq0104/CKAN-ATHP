
import numpy as np
pos_data = np.load("/home/ys/nijiaqi/features/Prot-t5_xl_uniref50/t5_data_pos_780.npy")
neg_data = np.load("/home/ys/nijiaqi/features/Prot-t5_xl_uniref50/t5_data_neg_780.npy")

combined_data = np.vstack((pos_data, neg_data))

np.save("/home/ys/nijiaqi/Convolutional-KANs-master/features/Prot-t5-merged.npy", combined_data)


print(pos_data.shape)
print(neg_data.shape)
print("combined data shape:", combined_data.shape)
