# import library
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM

# seed
import os
seed = 123
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# dataset
data = np.array([1,2,3,4,5,6,7,8,9,10])

# data preprocess
def split_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
        
    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)

x_train, y_train = split_data(data, 0, 8, 4, 0)
x_val, y_val = split_data(data, 4, 9, 4, 0)
x_test, y_test = split_data(data, 5, None, 4, 0)

print(x_train)
print(y_train)
print(x_val)
print(y_val)
print(x_test)
print(y_test)

# model structure
model = Sequential()
model.add(LSTM(100, input_shape=(4, 1)))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# model training
model.compile(loss='mse', optimizer='adam')
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1000, batch_size=1, verbose=1)

# result
prediction = model.predict(x_test)
print(prediction)

print("실제값: {:.3f}, 예측값: {:.3f}".format(y_test[0], prediction[0][0]))
