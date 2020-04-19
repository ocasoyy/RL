# Model 정의

import config
from preprocess import get_X

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import BinaryAccuracy


tf.keras.backend.set_floatx('float32')

class DeepFM(tf.keras.Model):

    def __init__(self, embedding_size, num_feature, num_field, field_index):
        super(DeepFM, self).__init__()
        self.embedding_size = embedding_size    # k: 임베딩 벡터의 차원(크기)
        self.num_feature = num_feature          # 원래 feature 개수
        self.num_field = num_field              # m: grouped field 개수
        self.field_index = field_index          # 인코딩된 X의 칼럼들이 본래 어디 소속이었는지

        self.w = tf.Variable(tf.random.normal(shape=[num_field],
                                              mean=0.0, stddev=0.1), name='w')
        self.V = tf.Variable(tf.random.normal(shape=(num_field, embedding_size),
                                              mean=0.0, stddev=0.1), name='V')
        # --> 이걸 tf.nn.embedding_lookup으로 각각의 필드에 대해 복제한다.

        #self.layers1 = tf.keras.layers.Dense(units=32, activation='relu')
        #self.layers2 = tf.keras.layers.Dense(units=16, activation='relu')
        #self.layers3 = tf.keras.layers.Dense(units=1, activation='sigmoid')


    def __repr__(self):
        pass

    def call(self, inputs):
        # Embedding
        # embeds = (None, num_feature=len(field_index), embedding_size)
        # embeds = tf.nn.embedding_lookup(params=self.V, ids=self.field_index)
        x_batch = tf.reshape(inputs, [-1, self.num_feature , 1])
        embeddings = tf.multiply(self.V, x_batch)    # (None, 131, 5)

        # TODO: drop out or L2 Regularization 추가

        # Deep Component를 위한 Embedding inputs 생성
        # -> 0이 아닌 행들만 모아 만든 (14, 5) Dense Tensor
        mask = tf.not_equal(embeddings, 0)
        non_zero_array = tf.boolean_mask(embeddings, mask, axis=0)
        # (8, 14, 5)
        embedding_inputs = tf.reshape(non_zero_array, [-1, self.num_field, self.embedding_size])

        # 1) FM Component
        # (8, )
        linear_terms = tf.nn.embedding_lookup(params=self.w, ids=self.field_index)
        linear_terms = tf.reduce_sum(
            tf.math.multiply(linear_terms, inputs), axis=1, keepdims=False)

        # (8, )
        interactions = 0.5 * tf.reduce_sum(
            tf.subtract(
                tf.square(tf.reduce_sum(embeddings, 1)),
                tf.reduce_sum(tf.square(embeddings), 1),
            ), keepdims=False)

        # (8, )
        # y_fm = tf.concat([self.linear_terms, self.interactions], axis=1)
        y_fm = tf.math.sigmoid(linear_terms + interactions)

        # 2) Deep Component
        #y_deep = tf.reshape(self.embeddings, [-1, self.num_feature*self.embedding_size])
        #y_deep = self.layers1(inputs=y_deep)
        #y_deep = self.layers2(inputs=y_deep)

        # Final Output
        #y_pred = self.layers3(inputs=tf.concat([y_fm, y_deep], axis=1))
        #y_pred = tf.reshape(y_pred, [-1, ])

        return y_fm

"""
train_ds = tf.data.Dataset.from_tensor_slices(
    (tf.cast(X_modified.values, tf.float32), tf.cast(Y, tf.float32))).shuffle(300000).batch(8)

x, y = next(iter(train_ds))

embedding_size = 5
num_field = 14
num_feature = 131

w = tf.Variable(tf.random.normal(shape=[num_field], mean=0.0, stddev=1.0))
V = tf.Variable(tf.random.normal(shape=(num_field, embedding_size), mean=0.0, stddev=0.01))

# 요 밑의 embeds가 진정한 V이다.
embeds = tf.nn.embedding_lookup(params=V, ids=field_index) # 131, 5
x_batch = tf.reshape(x, [-1, num_feature, 1])              # 8, 131, 1
embeddings = tf.multiply(embeds, x_batch)                  # 8, 131, 5

# TODO 한 행에 하나라도 True가 있으면 그 벡터는 전부 True로 씌우는 마스크로 변형...
# 만약 임베딩된 값 중 하나라도 0.0이 있으면 에러가 발생할 수 있다.
mask = tf.not_equal(embeddings, 0)
non_zero_array = tf.boolean_mask(embeddings, mask, axis=0)
embedding_inputs = tf.reshape(non_zero_array, [-1, num_field, embedding_size])    # 8, 14, 5

x_batch = tf.reshape(x_batch, [-1, num_feature])
linear_terms = tf.nn.embedding_lookup(params=w, ids=field_index)
linear_terms = tf.reduce_sum(tf.math.multiply(linear_terms, x_batch), axis=1, keepdims=False)     # 8, 131

# (8, 5)
interactions = 0.5 * tf.reduce_sum(
   tf.subtract(
       tf.square(tf.reduce_sum(embeddings, 1)),
       tf.reduce_sum(tf.square(embeddings), 1),
), keepdims=False)

# (8, 136)
# y_fm = tf.concat([linear_terms, interactions], axis=1)
y_fm = tf.math.sigmoid(linear_terms + interactions)
"""

# Forward
#@tf.function
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
def train(epochs, batch_size):
    file = pd.read_csv('data/adult.data', header=None)
    X = file.loc[:, 0:13]
    Y = file.loc[:, 14].map({' <=50K': 0, ' >50K': 1})

    X.columns = config.ALL_FIELDS
    field_dict, field_index, X_modified = get_X(X, 5, config.ALL_FIELDS, config.CONT_FIELDS, config.CAT_FIELDS)
    X_train, X_test, Y_train, Y_test = train_test_split(X_modified, Y, test_size=0.2, stratify=Y)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_train.values, tf.float32), tf.cast(Y_train, tf.float32)))\
        .shuffle(30000).batch(batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_test.values, tf.float32), tf.cast(Y_test, tf.float32)))\
        .shuffle(10000).batch(batch_size)

    model = DeepFM(embedding_size=5, num_feature=X_train.shape[1],
                   num_field=len(field_dict), field_index=field_index)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    accuracy = BinaryAccuracy(threshold=0.5)

    for i in range(epochs):
        loss_history = []

        for x, y in train_ds:
            loss = train_on_batch(model, optimizer, accuracy, x, y)
            loss_history.append(loss)

        print("스텝 {:03d}에서 Loss: {:.4f}".format(i, np.mean(loss_history)))
        print("스텝 {:03d}에서 누적 train 정확도: {:.4f}".format(i, accuracy.result().numpy()))

    test_accuracy = BinaryAccuracy(threshold=0.5)
    for x, y in test_ds:
        y_pred = model(x)
        test_accuracy.update_state(y, y_pred)

    print("테스트 정확도: {:.4f}".format(test_accuracy.result().numpy()))


if __name__ == '__main__':
    train(epochs=100, batch_size=256)
