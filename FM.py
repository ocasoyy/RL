# FM
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

file = load_breast_cancer()
scaler = MinMaxScaler()

X, Y = file['data'], file['target']
X = scaler.fit_transform(X)

n = X.shape[0]
p = X.shape[1]
k = 5
batch_size = 8

x = X[0:batch_size].astype('float32')
y = Y[0:batch_size]

"""
w_0 = tf.Variable([0.0])
w = tf.Variable(tf.zeros([p]))
V = tf.Variable(tf.random.normal(shape=(p, k)))

linear_terms = tf.reduce_sum(tf.math.multiply(w, x), axis=1)
interactions = 0.5 * tf.reduce_sum(
    tf.math.pow(tf.matmul(x, V), 2) - tf.matmul(tf.math.pow(x, 2), tf.math.pow(V, 2)),
    1,
    keepdims=False
)

y_hat = w_0 + linear_terms + interactions
loss = tf.keras.losses.binary_crossentropy(from_logits=True, y_true=y, y_pred=y_hat)

with tf.GradientTape() as tape:
    grads = tape.gradient(target=loss, sources=[w_0, w, V])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
optimizer.apply_gradients(zip(grads, [w_0, w, V]))
"""


# 헙...
class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.w_0 = tf.Variable([0.0])
    self.w = tf.Variable(tf.zeros([p]))
    self.V = tf.Variable(tf.random.normal(shape=(p, k)))

  def call(self, inputs):
    linear_terms = tf.reduce_sum(tf.math.multiply(self.w, inputs))
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
    loss = tf.keras.losses.binary_crossentropy(from_logits=True,
                                               y_true=y,
                                               y_pred=model(inputs))
  return loss, tape.gradient(target=loss, sources=model.trainable_variables)

model = Model()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(X, Y, batch_size=8, epochs=10, verbose=1, validation_split=0.2)

# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# y_pred = model(x)

print("초기 손실: {:.3f}".format(tf.keras.losses.binary_crossentropy(from_logits=False,
                                               y_true=y,
                                               y_pred=model(x))))

train_ds = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(600).batch(8)

# 반복 훈련
for i in range(50):
  for x, y in train_ds:
      loss, grads = grad(model, x, y)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      
  if i % 10 == 0:
      print("스텝 {:03d}에서 손실: {:.3f}".format(i, loss))

print("최종 손실: {:.3f}".format(loss(model, x, y)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))

model.predict()



