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


class DeepModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(DeepModel, self).__init__()

        # Define LSTM layers with clear variable naming
        self.lstm1 = tf.keras.layers.LSTM(256, return_sequences=True, input_shape=input_shape, name='lstm1')
        self.lstm2 = tf.keras.layers.LSTM(128, return_sequences=True, name='lstm2')
        self.lstm3 = tf.keras.layers.LSTM(64, return_sequences=True, name='lstm3')
        self.lstm4 = tf.keras.layers.LSTM(32, return_sequences=False, name='lstm4')

        # Dense layers with activation for output scaling
        self.dense1 = tf.keras.layers.Dense(48, activation='relu', name='dense1')  # ReLU for hidden layer
        self.dense2 = tf.keras.layers.Dense(1, name='output')  # No activation for final regression output
        
        
    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.lstm3(x)
        x = self.lstm4(x)
        x = self.dense1(x)
        return self.dense2(x)
    
    def train(self, x_train, y_train, batch_size, epochs, validation_data=None):
        self.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data)  # Consistent fit usage

class CNNModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(CNNModel, self).__init__()

        # Define LSTM layers with clear variable naming
        self.conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=input_shape, name='conv1')
        self.pool1 = tf.keras.layers.MaxPooling1D(2, name='pool1')
        self.conv2 = tf.keras.layers.Conv1D(32, 3, activation='relu', name='conv2')
        self.pool2 = tf.keras.layers.MaxPooling1D(2, name='pool2')
        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.dense1 = tf.keras.layers.Dense(128, activation='relu', name='dense1')
        self.dense2 = tf.keras.layers.Dense(1, name='output')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

    def train(self, x_train, y_train, batch_size, epochs, validation_data=None):
        self.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data)  # Consistent fit usage