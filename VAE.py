import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from time import perf_counter

# Data
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

train_size = 60000
batch_size = 32
test_size = 10000

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ])

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim, )),
                tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'),
            ])

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        stddev = 1e-8 + tf.nn.softplus(logvar)
        return mean, stddev

    def reparameterize(self, mean, stddev):
        eps = tf.random.normal(shape=mean.shape)
        z = mean + eps * stddev
        return z

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

optimizer = tf.keras.optimizers.Adam(1e-4)


def compute_loss(model, x):
    mean, stddev = model.encode(x)
    z = model.reparameterize(mean, stddev)
    x_logit = model.decode(z, True)
    x_logit = tf.clip_by_value(x_logit, 1e-8, 1-1e-8)

    # Loss
    marginal_likelihood = tf.reduce_sum(x * tf.math.log(x_logit) + (1 - x) * tf.math.log(1 - x_logit), axis=[1, 2])
    loglikelihood = tf.reduce_mean(marginal_likelihood)

    kl_divergence = -0.5 * tf.reduce_sum(1 + tf.math.log(1e-8 + tf.square(stddev)) - tf.square(mean) - tf.square(stddev),
                                         axis=[1])
    kl_divergence = tf.reduce_mean(kl_divergence)

    ELBO = loglikelihood - kl_divergence
    loss = -ELBO

    return loss


@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


epochs = 10
latent_dim = 2
model = CVAE(latent_dim)

# Train
for epoch in range(1, epochs + 1):
    train_losses = []
    for train_x in train_dataset:
        loss = train_step(model, train_x, optimizer)
        train_losses.append(loss)

    print('Epoch: {}, Loss: {:.2f}'.format(epoch, np.mean(train_losses)))

    metric = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        metric(compute_loss(model, test_x))
    elbo = -metric.result()


# Test
def generate_images(model, test_sample, random_sample=False):
    mean, stddev = model.encode(test_sample)
    z = model.reparameterize(mean, stddev)

    if random_sample:
        predictions = model.sample(z)
    else:
        predictions = model.decode(z, True)

    predictions = tf.clip_by_value(predictions, 1e-8, 1 - 1e-8)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.show()


num_examples_to_generate = 16
random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])
test_sample = next(iter(test_dataset))[0:num_examples_to_generate, :, :, :]

for i in range(test_sample.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(test_sample[i, :, :, 0], cmap='gray')
    plt.axis('off')

plt.show()

generate_images(model, test_sample, False)
