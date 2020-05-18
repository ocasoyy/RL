# Stacked AutoEncoder

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.losses import mean_squared_error


# Data Load
BATCH_SIZE = 256
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
train_images /= 255.
test_images /= 255.
train_ds = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(BATCH_SIZE)
test_ds= tf.data.Dataset.from_tensor_slices(test_images).shuffle(10000).batch(BATCH_SIZE)


class AutoEncoder(Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.layers1 = Dense(units=128, activation='relu')
        self.layers2 = Dense(units=64, activation='sigmoid')
        self.layers3 = Dense(units=28*28, activation='sigmoid')

    def call(self, inputs):
        batch_size = inputs.shape[0]
        inputs = tf.reshape(inputs, [batch_size, -1])
        output = self.layers1(inputs)
        output = self.layers2(output)
        output = self.layers3(output)

        return output


def train_on_batch(model, optimizer, inputs):
    with tf.GradientTape() as tape:
        output = model(inputs)
        loss = mean_squared_error(output, inputs)

    grads = tape.gradient(target=loss, sources=model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def train(epochs):
    model = AutoEncoder()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    for i in range(epochs):
        loss_history = []

        for x in train_ds:
            loss = train_on_batch(model, optimizer, tf.reshape(x, [-1, 784]))
            loss_history.append(tf.reduce_sum(loss))

        print("Epoch {:03d}: 누적 Loss: {:.4f}".format(i, np.mean(loss_history)))

    return model


if __name__ == '__main__':
    model = train(30)
    
    x = next(iter(test_ds))
    pred = model(tf.reshape(x[0], [1, 28, 28]))

#plt.imshow(tf.reshape(pred, [28, 28]))
#plt.imshow(tf.reshape(x[0], [28, 28]))
