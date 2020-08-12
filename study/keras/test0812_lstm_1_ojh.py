# library
import numpy as np
import tensorflow as tf
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import train_test_split

# seed
import os
seed = 123
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)

# data
data = np.array(range(1, 101))
print(data)

# data slice
from slice_1_ojh import split_mm
x, y = split_mm(data, 7, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print(x_train)
print(y_train)

x_predict = np.array(list(range(111,118)))
x_predict = np.reshape(x_predict, (-1, 7, 1))
x_pred = tf.cast(x_predict, tf.float32)
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