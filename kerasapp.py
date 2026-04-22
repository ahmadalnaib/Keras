from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, LSTM
import numpy as np

model = Sequential()
model.add(Input(shape=(1,)))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mae', 'mape'])

x_train = np.array([1, 2, 3, 4])
y_train = np.array([2, 4, 6, 8])

epochs = 1000
batch_size = 2
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
weights,bias = model.get_weights()
prediction = model.predict(np.array([5,6,7]))
print(np.round(prediction))