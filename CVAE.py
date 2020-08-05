# Conditional Variational AutoEncoder

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from time import perf_counter

# Data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

train_size = 60000
batch_size = 32
test_size = 10000

train_dataset = (tf.data.Dataset.from_tensor_slices(
    (tf.cast(train_images, tf.float32), tf.cast(train_labels, tf.float32)))
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(
    (tf.cast(test_images, tf.float32), tf.cast(test_labels, tf.float32)))
                .shuffle(test_size).batch(batch_size))

# x, y = next(iter(train_dataset))

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28*28 + 1)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=512, activation='relu'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(units=256, activation='relu'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dense(units=latent_dim + latent_dim),
            ])

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim+1)),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(units=256, activation='relu'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(units=512, activation='relu'),
                tf.keras.layers.Dense(units=784),
            ])

    @tf.function
    def encode(self, x, y):
        inputs = tf.concat([x, y], 1)
        mean, logvar = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        stddev = 1e-8 + tf.nn.softplus(logvar)
        return mean, stddev

    def reparameterize(self, mean, stddev):
        eps = tf.random.normal(shape=mean.shape)
        z = mean + eps * stddev
        return z

    @tf.function
    def decode(self, z, y, apply_sigmoid=False):
        inputs = tf.concat([z, y], 1)
        logits = self.decoder(inputs)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

optimizer = tf.keras.optimizers.Adam(1e-4)


def compute_loss(model, x, y):
    x = tf.reshape(x, [-1, 784])
    y = tf.reshape(y, [-1, 1])
    mean, stddev = model.encode(x, y)
    z = model.reparameterize(mean, stddev)
    x_logit = model.decode(z, y, True)
    x_logit = tf.clip_by_value(x_logit, 1e-8, 1-1e-8)

    # Loss
    marginal_likelihood = tf.reduce_sum(x * tf.math.log(x_logit) + (1 - x) * tf.math.log(1 - x_logit), axis=[1])
    loglikelihood = tf.reduce_mean(marginal_likelihood)

    kl_divergence = -0.5 * tf.reduce_sum(1 + tf.math.log(1e-8 + tf.square(stddev)) - tf.square(mean) - tf.square(stddev),
                                         axis=[1])
    kl_divergence = tf.reduce_mean(kl_divergence)

    ELBO = loglikelihood - kl_divergence
    loss = -ELBO

    return loss


@tf.function
def train_step(model, x, y, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


epochs = 30
latent_dim = 2
model = CVAE(latent_dim)

# Train
for epoch in range(1, epochs + 1):
    train_losses = []
    for x, y in train_dataset:
        loss = train_step(model, x, y, optimizer)
        train_losses.append(loss)

    print('Epoch: {}, Loss: {:.2f}'.format(epoch, np.mean(train_losses)))


# Test
def generate_images(model, test_x, test_y):
    test_x = tf.reshape(test_x, [-1, 784])
    test_y = tf.reshape(test_y, [-1, 1])
    mean, stddev = model.encode(test_x, test_y)
    z = model.reparameterize(mean, stddev)

    predictions = model.decode(z, test_y, True)
    predictions = tf.clip_by_value(predictions, 1e-8, 1 - 1e-8)
    predictions = tf.reshape(predictions, [-1, 28, 28, 1])

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.show()


num_examples_to_generate = 16
random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])
test_x, test_y = next(iter(test_dataset))
test_x, test_y = test_x[0:num_examples_to_generate, :, :, :], test_y[0:num_examples_to_generate, ]

for i in range(test_x.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(test_x[i, :, :, 0], cmap='gray')
    plt.axis('off')

plt.show()

generate_images(model, test_x, test_y)
