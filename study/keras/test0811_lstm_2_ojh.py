# library
import numpy as np
import tensorflow as tf
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import LSTM

# seed
import os
seed = 123
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)

# data
data = np.array(range(1, 118))
print(data)

# data preprocess
def split_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
        
    for i in range(start_index, end_index):
        indices1 = range(i - history_size, i)
        indices2 = range(i+1 - history_size, i+1)
        indices3 = range(i+2 - history_size, i+2)
        array=np.concatenate([dataset[indices1],dataset[indices2],dataset[indices3]])
        data.append(np.reshape(array, (3, history_size)))
        labels.append(dataset[i+2:i+2+target_size])
    return np.array(data), np.array(labels)

x_train, y_train = split_data(data, 0, 103, 7, 3)
print(x_train)
print(y_train)

x_test, y_test = split_data(data, 105, 113, 7, 3)
print(x_test)
print(y_test)

x_pred, _ = split_data(data, 108, 116, 7, 3)
x_pred = tf.cast(x_pred, tf.float32)
print(x_pred)

# model structure
model = Sequential()
model.add(LSTM(150, input_shape=(3,7), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(3))

model.summary()

# compile
model.compile(loss='mse', optimizer='adam')

# EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience = 30, mode='min')

# fit
hist = model.fit(x_train, y_train, epochs= 300, batch_size= 3, verbose = 1,
                 validation_split=0.2, callbacks = [es])

# evaluate
evaluate = model.evaluate(x_test, y_test, batch_size=3, verbose=2)
print(evaluate)

# predict
y_pred = model.predict(x_pred)
print(y_pred)