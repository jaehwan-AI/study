import numpy as np
def split_mm(dataset, history_size, target_size):
    features = []
    labels = []
    
    for i in range(len(dataset)):
        x_index = i + history_size
        y_index = x_index + target_size
        if y_index > len(dataset):
            break
        x_tmp = dataset[i:x_index]
        y_tmp = dataset[x_index:y_index]
        features.append(x_tmp)
        labels.append(y_tmp)
    return np.array(features), np.array(labels)

def split_many_mm(dataset, history_size, history_size_dim, target_size):
    features = []
    labels = []

    for i in range(len(dataset)):
        x_index = i + history_size + history_size_dim
        y_index = x_index + target_size
        if y_index > len(dataset):
            break
        x_tmp = dataset[i:x_index]
        x_dim = []
        for j in range(history_size_dim):
            x_dim.append(x_tmp[j:-(history_size_dim-j)])
        features.append(x_dim)
        y_tmp = dataset[x_index-1:y_index-1]
        labels.append(y_tmp)
    return np.array(features), np.array(labels)