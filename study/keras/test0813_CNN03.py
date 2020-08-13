from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()

model.add(Conv2D(5, (2,2), strides=1, input_shape=(10, 10, 1)))
model.add(Conv2D(7, (3,3)))
model.add(Flatten())
model.add(Dense(1))

model.summary()