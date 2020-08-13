from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(10, input_shape=(3,1), return_sequences=True))
model.add(LSTM(9, return_sequences=True))
model.add(LSTM(8, return_sequences=True))
model.add(LSTM(7, return_sequences=True))
model.add(LSTM(6, return_sequences=True))
model.add(LSTM(5, return_sequences=True))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.summary()