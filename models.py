from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Flatten, Dropout, Conv1D, MaxPooling1D

class RNNModel:
    def __init__(self, shape):
        self.shape = shape
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.shape, 1)))
        model.add(LSTM(64, activation='tanh', return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(64, activation='tanh', return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(64, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(60, activation='tanh'))
        model.add(Dense(1))


        return model


class CNNModel:
    def __init__(self, shape):
        self.shape = shape
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.shape, 1)))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))


        return model
