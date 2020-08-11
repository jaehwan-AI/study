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
data = np.array(range(1, 101))
print(data)

# data split
def split_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
        
    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i:i+target_size])
    return np.array(data), np.array(labels)

x_train, y_train = split_data(data, 0, 98, 7, 3)
print(x_train)
print(y_train)

x_test, y_test = split_data(data, 90, 98, 7, 3)
print(x_test)
print(y_test)

x_predict = np.array(list(range(111,118)))
x_pred, _ = split_data(x_predict, 0, 8, 7, 3)
x_pred = tf.cast(x_pred, tf.float32)
print(x_pred)

# model structure
model = Sequential()
model.add(LSTM(150, input_shape=(7,1), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3))

model.summary()

# compile
model.compile(loss='mse', optimizer='adam')

# EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience = 30, mode='min')

# fit
hist = model.fit(x_train, y_train, epochs= 200, batch_size= 1, verbose = 1,
                 validation_split=0.2, callbacks = [es])

# evaluate
evaluate = model.evaluate(x_test, y_test, batch_size=1, verbose=2)
print(evaluate)

# predict
y_pred = model.predict(x_pred)
print(y_pred)