# Model 정의

import tensorflow as tf
from tensorflow.keras.layers import Embedding
from preprocess import get_X

num_field = 14

class DeepFM(tf.keras.Model):

    def __init__(self, embedding_size, num_feature, num_field):
        super().__init__(DeepFM, self)
        self.embedding_size = embedding_size    # k: 임베딩 벡터의 차원(크기)
        self.num_feature = num_feature
        self.num_field = num_field              # m: grouped field 개수

        # input_dim= 단어 집합의 크기. 즉, 총 단어의 개수
        # output_dim= 임베딩 벡터의 출력 차원. 결과로서 나오는 임베딩 벡터의 크기
        # input_length= 입력 시퀀스의 길이
        self.embedding = Embedding(input_dim=self.num_feature,
                                   output_dim=self.num_field * self.embedding_size,
                                   input_length=None)

        self.w_0 = tf.Variable([0.0])
        self.w = tf.Variable(tf.zeros([self.num_field]))
        self.V = tf.Variable(tf.random.normal(shape=(self.num_field, self.)))

    def call(self, inputs):
        # Embedding
        embeds = self.embedding(inputs=inputs)
        output = tf.matmul(embeds, tf.reshape(inputs, [-1, self.num_feature, 1]), 1)
        X_batch_embeds = tf.reshape(output, [self.embedding_size, self.num_field])

        # FM Component


        # Deep Component


        # Concatenation



# 모델의 파라미터 정의

linear_terms = tf.reduce_sum(tf.math.multiply(self.w, inputs), axis=1)
interactions = 0.5 * tf.reduce_sum(
    tf.math.pow(tf.matmul(inputs, self.V), 2)
    - tf.matmul(tf.math.pow(inputs, 2), tf.math.pow(self.V, 2)),
    1,
    keepdims=False
)

y_pred = tf.math.sigmoid(self.w_0 + linear_terms + interactions)




train_ds = tf.data.Dataset.from_tensor_slices(
    (tf.cast(X_modified.values, tf.float32), tf.cast(Y, tf.float32))).shuffle(300000).batch(1)

x, y = next(iter(train_ds))

embedding = Embedding(108, 14*5)
embeds = embedding(x)                                         # (batch_size, 108, 14*5)
output = tf.matmul(embeds, tf.reshape(x, [-1, 108, 1]), 1)    # (batch_size, 14*5, 1)

X_batch_embeds = tf.reshape(tensor=output, shape=[5, 14])





















