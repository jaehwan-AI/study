# import library
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# seed
import os
seed = 123
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# dataset
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,1,2,3,4,5])
print(x)
print(y)

# split dataset
x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=seed)

x_train = x_train[:,np.newaxis]
x_val = x_val[:,np.newaxis]
x_test = x_test[:,np.newaxis]

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)

# model structure
model = Sequential()
model.add(Dense(15, input_dim = x_train.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.summary()

# model training
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1000, batch_size=1, verbose=1)

# result
prediction = model.predict(x_test)
for i in range(len(y_test)):
    label = np.argmax(y_test[i])
    predict = np.argmax(prediction[i])
    print("실제값: {:.3f}, 예측값: {:.3f}".format(label, predict))