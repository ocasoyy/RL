import tensorflow as tf
import numpy as np
from itertools import combinations


class Embedding_layer(tf.keras.layers.Layer):
    def __init__(self, num_field, num_feature, num_cont, embedding_size):
        super(Embedding_layer, self).__init__()
        self.embedding_size = embedding_size    # k: 임베딩 벡터의 차원(크기)
        self.num_field = num_field              # m: 인코딩 이전 feature 수
        self.num_feature = num_feature          # p: 인코딩 이후 feature 수, m <= p
        self.num_cont = num_cont                # 연속형 field 수
        self.num_cat  = num_field - num_cont    # 범주형 field 수

        # Parameters
        self.V = tf.Variable(tf.random.normal(shape=(num_feature, embedding_size),
                                              mean=0.0, stddev=0.01), name='V')

    def call(self, inputs):
        # inputs: (None, p, k), embeds: (None, m, k)
        batch_size = inputs.shape[0]

        # 원핫인코딩으로 생성된 0을 제외한 값에 True를 부여한 mask(np.array): (None, m)
        # indices: 그 mask의 indices
        cont_mask = np.full(shape=(batch_size, self.num_cont), fill_value=True)
        cat_mask = tf.not_equal(inputs[:, self.num_cont:], 0.0).numpy()
        mask = np.concatenate([cont_mask, cat_mask], axis=1)

        _, flatten_indices = np.where(mask == True)
        indices = flatten_indices.reshape((batch_size, self.num_field))

        # embedding_matrix: (None, m, k)
        embedding_matrix = tf.nn.embedding_lookup(params=self.V, ids=indices.tolist())

        # masked_inputs: (None, m, 1)
        masked_inputs = tf.reshape(tf.boolean_mask(inputs, mask),
                                   [batch_size, self.num_field, 1])

        masked_inputs = tf.multiply(masked_inputs, embedding_matrix)    # (None, m, k)

        return masked_inputs


class Pairwise_Interaction_Layer(tf.keras.layers.Layer):
    def __init__(self, num_field, num_feature, embedding_size):
        super(Pairwise_Interaction_Layer, self).__init__()
        self.embedding_size = embedding_size    # k: 임베딩 벡터의 차원(크기)
        self.num_field = num_field              # m: 인코딩 이전 feature 수
        self.num_feature = num_feature          # p: 인코딩 이후 feature 수, m <= p
        self.comb_list = list(range(0, num_field, 1))


    def call(self, inputs):
        batch_size = inputs.shape[0]
        interactions = []

        for b in range(batch_size):
            for i, j in list(combinations(self.comb_list, 2)):
                interactions.append(tf.multiply(inputs[b, i, :], inputs[b, j, :]))

        pairwise_interactions = tf.reshape(tf.stack(interactions),
                                           (batch_size, -1, self.embedding_size))

        return pairwise_interactions


class Attention_Pooling_Layer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, hidden_size):
        super(Attention_Pooling_Layer, self).__init__()
        self.embedding_size = embedding_size    # k: 임베딩 벡터의 차원(크기)

        # Parameters
        self.h = tf.Variable(tf.random.normal(shape=(1, hidden_size),
                                              mean=0.0, stddev=0.1), name='h')
        self.W = tf.Variable(tf.random.normal(shape=(hidden_size, embedding_size),
                                              mean=0.0, stddev=0.1), name='W_attention')
        self.b = tf.Variable(tf.zeros(shape=(hidden_size, 1)))


    def call(self, inputs):
        # 조합 수 = combinations(num_feauture, 2)
        # inputs: (None, 조합 수, embedding_size)
        # --> (전치 후) (None, embedding_size, 조합 수)
        inputs = tf.transpose(inputs, [0, 2, 1])

        # e: (None, 조합 수, 1)
        e = tf.matmul(self.h, tf.nn.relu(tf.matmul(self.W, inputs) + self.b))
        e = tf.transpose(e, [0, 2, 1])

        # Attention Score 산출
        attention_score = tf.nn.softmax(e)

        return attention_score







