# FM

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer

file = load_breast_cancer()

X, Y = file['data'], file['target']

n = X.shape[0]
p = X.shape[1]
k = 5
batch_size = 8

x = X[0:batch_size].astype('float32')
y = Y[0:batch_size]

w_0 = tf.Variable([0.0])
W = tf.Variable(tf.random.normal(shape=(p, p)))
V = tf.Variable(tf.random.normal(shape=(p, k)))

linear_terms = tf.reduce_sum(tf.math.multiply(w_0, x))
interactions = 0.5 * tf.reduce_sum(
    tf.math.pow(tf.matmul(x, V), 2) - tf.matmul(tf.math.pow(x, 2), tf.math.pow(V, 2)),
    1,
    keepdims=False
)

y_hat = w_0 + linear_terms + interactions
output = tf.math.sigmoid(y_hat)
loss = tf.keras.losses.binary_crossentropy(from_logits=False, y_true=y, y_pred=output)

# 헙...
class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.w_0 = tf.Variable([0.0])
    self.W = tf.Variable(tf.random.normal(shape=(p, p)))
    self.V = tf.Variable(tf.random.normal(shape=(p, k)))

  def call(self, inputs):
    linear_terms = tf.reduce_sum(tf.math.multiply(self.w_0, inputs))
    interactions = 0.5 * tf.reduce_sum(
        tf.math.pow(tf.matmul(inputs, self.V), 2)
        - tf.matmul(tf.math.pow(inputs, 2), tf.math.pow(self.V, 2)),
        1,
        keepdims=False
    )

    y_hat = tf.math.sigmoid(self.w_0 + linear_terms + interactions)

    return y_hat

# Forward
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss = tf.keras.losses.binary_crossentropy(from_logits=False,
                                               y_true=y,
                                               y_pred=model(inputs))
  return tape.gradient(target=loss, sources=model.trainable_variables)

model = Model()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
print("초기 손실: {:.3f}".format(tf.keras.losses.binary_crossentropy(from_logits=False,
                                               y_true=y,
                                               y_pred=model(x))))

train_ds = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(600).batch(8)

# 반복 훈련
for i in range(300):
  for x, y in train_ds:
      grads = grad(model, x, y)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      
      if i % 10 == 0:
          print("스텝 {:03d}에서 손실: {:.3f}".format(i, loss(model, x, y)))

print("최종 손실: {:.3f}".format(loss(model, x, y)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))





