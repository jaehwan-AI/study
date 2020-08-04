# import library
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from sklearn.model_selection import train_test_split

# seed
import os
seed = 123
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# dataset
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
print(x)
print(y)

# split dataset
x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=seed)

x_train = x_train[:,np.newaxis]
x_val = x_val[:,np.newaxis]
x_test = x_test[:,np.newaxis]
y_train = y_train[:,np.newaxis]
y_val = y_val[:,np.newaxis]
y_test = y_test[:,np.newaxis]

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)

# model structure
model = Sequential()
model.add(Dense(10, input_dim = x_train.shape[1], activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()

# model training
model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=500, batch_size=1, verbose=1)

# result
prediction = model.predict(x_test)
print(prediction)

error_mae = np.mean(abs(y_test - prediction))
error_mse = np.mean((y_test - prediction) ** 2)
print(error_mae)
print(error_mse)
for i in range(len(y_test)):
    label = y_test[i][0]
    predict = prediction[i][0]
    print("실제값: {:.3f}, 예측값: {:.3f}".format(label, predict))