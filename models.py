#ridge regression model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
#randome forest regressor

#Baseline Model

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
        model.add(LSTM(64, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(60, activation='tanh'))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        return model


