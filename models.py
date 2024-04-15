import tensorflow as tf


class ReferenceModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(ReferenceModel, self).__init__()

        # Define LSTM layers with clear variable naming
        self.lstm1 = tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape, name='lstm1')
        self.lstm2 = tf.keras.layers.LSTM(64, return_sequences=False, name='lstm2')

        # Dense layers with activation for output scaling
        self.dense1 = tf.keras.layers.Dense(25, activation='relu', name='dense1')  # ReLU for hidden layer
        self.dense2 = tf.keras.layers.Dense(1, name='output')  # No activation for final regression output

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.dense1(x)
        return self.dense2(x)

    def train(self, x_train, y_train, batch_size, epochs, validation_data=None):
        self.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data)  # Consistent fit usage

    def predict(self, x_test):
        return self.predict(x_test)  # Already in parent Model class

