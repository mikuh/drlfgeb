import tensorflow as tf


class CnnEmbedding(tf.keras.layers.Layer):
    def __init__(self, state_shape):
        super().__init__()
        self.conv0 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu', input_shape=state_shape)
        self.pool0 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv1 = tf.keras.layers.Conv2D(32, 5, activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, 4, activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(64, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(512)
        self.prelu = tf.keras.layers.PReLU()

    def call(self, inputs):
        inputs = self.conv0(inputs)
        inputs = self.pool0(inputs)
        inputs = self.conv1(inputs)
        inputs = self.pool1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.pool2(inputs)
        inputs = self.conv3(inputs)
        inputs = self.flatten(inputs)
        inputs = self.fc(inputs)
        inputs = self.prelu(inputs)
        return inputs


class DenseEmbedding(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.layer1 = tf.keras.layers.Dense(256, activation='relu')
        self.layer2 = tf.keras.layers.Dense(128, activation='relu')

    def call(self, inputs):
        output = self.layer1(inputs)
        output = self.layer2(output)
        return output
