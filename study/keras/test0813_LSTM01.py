from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(10, input_shape=(3,1)))
model.add(Dense(5))
model.add(Dense(1))

model.summary()