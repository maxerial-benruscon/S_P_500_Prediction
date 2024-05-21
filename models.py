import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, TimeDistributed, BatchNormalization

class RNNModel:
    def __init__(self, shape):
        self.shape = shape
        self.model = self.build_model()
    def build_model(self):
        model = Sequential()
        model.add(LSTM(64, activation='tanh', input_shape=(self.shape, 1), return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(64, activation='tanh', return_sequences=True))
        model.add(Dropout(0.5))
        model.add(Dense(60, activation='tanh'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        return model
    
class CNNModel:
    def __init__(self, shape):
        self.shape = shape
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # Adding Conv1D layers
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(self.input_shape, 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))

        model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))
        model.add(Conv1D(filters=512, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))

        # Applying TimeDistributed Flatten layer
        model.add(TimeDistributed(Flatten()))

        # Adding LSTM layers
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.5))
        # Adding Dense layers
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        # Compile the model
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])

        return model