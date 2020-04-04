# FM
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# GPU 확인
tf.config.list_physical_devices('GPU')

# 자료형 선언
# tf.debugging.assert_equal(y_pred.shape, (batch_size, ))
tf.keras.backend.set_floatx('float32')

# 데이터 로드
scaler = MinMaxScaler()
file = pd.read_csv('data/banknote.txt', header=None)
X, Y = file.loc[:, 0:3], file.loc[:, 4]
X = scaler.fit_transform(X)

n = X.shape[0]
p = X.shape[1]
k = 3
batch_size = 8
epochs = 20

# 참고: 한 batch만 불러오고 싶으면
# x, y = next(iter(train_ds))

class FM(tf.keras.Model):
    def __init__(self):
        super(FM, self).__init__()

        # 모델의 파라미터 정의
        self.w_0 = tf.Variable([0.0])
        self.w = tf.Variable(tf.zeros([p]))
        self.V = tf.Variable(tf.random.normal(shape=(p, k)))

    def call(self, inputs):
        linear_terms = tf.reduce_sum(tf.math.multiply(self.w, inputs), axis=1)
        interactions = 0.5 * tf.reduce_sum(
            tf.math.pow(tf.matmul(inputs, self.V), 2)
            - tf.matmul(tf.math.pow(inputs, 2), tf.math.pow(self.V, 2)),
            1,
            keepdims=False
        )

        y_hat = tf.math.sigmoid(self.w_0 + linear_terms + interactions)

        return y_hat


# Forward
def train_on_batch(model, optimizer, accuracy, inputs, targets):
    with tf.GradientTape() as tape:
        y_pred = model(inputs)
        loss = tf.keras.losses.binary_crossentropy(from_logits=False, y_true=targets, y_pred=y_pred)

    grads = tape.gradient(target=loss, sources=model.trainable_variables)

    # apply_gradients()를 통해 processed gradients를 적용함
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # accuracy
    accuracy.update_state(targets, y_pred)

    return loss


# 반복 학습 함수
def train(epochs):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_train, tf.float32), tf.cast(Y_train, tf.float32))).shuffle(500).batch(8)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_test, tf.float32), tf.cast(Y_test, tf.float32))).shuffle(200).batch(8)

    model = FM()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    accuracy = BinaryAccuracy(threshold=0.5)
    loss_history = []

    for i in range(epochs):
      for x, y in train_ds:
          loss = train_on_batch(model, optimizer, accuracy, x, y)
          loss_history.append(loss)

      if i % 2== 0:
          print("스텝 {:03d}에서 누적 평균 손실: {:.4f}".format(i, np.mean(loss_history)))
          print("스텝 {:03d}에서 누적 train 정확도: {:.4f}".format(i, accuracy.result().numpy()))


    test_accuracy = BinaryAccuracy(threshold=0.5)
    for x, y in test_ds:
        y_pred = model(x)
        test_accuracy.update_state(y, y_pred)

    print("테스트 정확도: {:.4f}".format(test_accuracy.result().numpy()))


if __name__ == '__main__':
    train(epochs=epochs)

