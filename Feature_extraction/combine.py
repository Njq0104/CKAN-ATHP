
import numpy as np
pos_data = np.load("feature.npy")
neg_data = np.load("feature.npy")

combined_data = np.vstack((pos_data, neg_data))

np.save("feature.npy", combined_data)

