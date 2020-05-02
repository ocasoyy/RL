import tensorflow as tf
import config
import numpy as np

class Embedding_layer(tf.keras.layers.Layer):
    def __init__(self, num_field, num_feature, num_cont, embedding_size):
        super(Embedding_layer, self).__init__()
        self.embedding_size = embedding_size    # k: 임베딩 벡터의 차원(크기)
        self.num_field = num_field              # m: 인코딩 이전 feature 수
        self.num_feature = num_feature          # p: feature 개수

        # filed 기준
        self.num_cont = num_cont
        self.num_cat  = num_field - num_cont

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
        indices = flatten_indices.reshape((batch_size, self.num_feature))

        # embedding_matrix: (None, m, k)
        embedding_matrix = tf.nn.embedding_lookup(params=self.V, ids=indices.tolist())

        # masked_inputs: (None, m, 1)
        masked_inputs = tf.reshape(tf.boolean_mask(inputs, mask),
                                   [batch_size, self.num_field, 1])

        masked_inputs = tf.multiply(masked_inputs, embedding_matrix)    # (None, m, k)

        return masked_inputs
