# Model 정의

import config
from preprocess import get_X

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split


file = pd.read_csv('data/adult.data', header=None)
X = file.loc[:, 0:13]
Y = file.loc[:, 14].map({' <=50K': 0, ' >50K': 1})


X.columns = config.ALL_FIELDS
field_dict, field_index, X_modified = get_X(X, config.ALL_FIELDS, config.CONT_FIELDS, config.CAT_FIELDS)
X_train, X_test, Y_train, Y_test = train_test_split(X_modified, Y, test_size=0.2, stratify=Y)


class DeepFM(tf.keras.Model):

    def __init__(self, embedding_size, num_feature, num_field, field_index):
        super().__init__(DeepFM, self)
        self.embedding_size = embedding_size    # k: 임베딩 벡터의 차원(크기)
        self.num_feature = num_feature
        self.num_field = num_field              # m: grouped field 개수

        self.field_index = field_index          # 인코딩된 X의 칼럼들이 본래 어디 소속이었는지

        # input_dim= 단어 집합의 크기. 즉, 총 단어의 개수
        # output_dim= 임베딩 벡터의 출력 차원. 결과로서 나오는 임베딩 벡터의 크기
        # input_length= 입력 시퀀스의 길이

        self.w_0 = tf.Variable([0.0])
        self.w = tf.Variable(tf.zeros([self.num_field]))
        self.V = tf.Variable(tf.random.normal(shape=(self.num_field, self.embedding_size)))
        # --> 이걸 tf.nn.embedding_lookup으로 각각의 필드에 대해 복제한다.


    def call(self, inputs, field_dict):
        # Embedding
        # embeds = (None, num_feature=len(field_index), embedding_size)
        self.embeds = tf.nn.embedding_lookup(params=self.V, ids=self.field_index)
        x_batch = tf.reshape(inputs, [-1, self.num_feature ,1])
        self.embeddings = tf.multiply(self.embeds, x_batch)

        # TODO: 연속형 변수 구간화 해서 바로 윗 단계에서 14개의 0이 아닌 숫자가 나오게 만든 후
        # 0이 아닌 행들만 모아서 진짜 임베딩된 X_batch: (14, 5)를 만들어낸다.

        # 바로 위 embeds: (None, 108, 5) -->

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
    (tf.cast(X_modified.values, tf.float32), tf.cast(Y, tf.float32))).shuffle(300000).batch(8)

x, y = next(iter(train_ds))






















