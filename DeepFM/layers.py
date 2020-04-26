import tensorflow as tf


class FM_layer(tf.keras.layers.Layer):
    def __init__(self, num_feature, num_field, embedding_size, field_index):
        super(FM_layer, self).__init__()
        self.embedding_size = embedding_size    # k: 임베딩 벡터의 차원(크기)
        self.num_feature = num_feature          # f: 원래 feature 개수
        self.num_field = num_field              # m: grouped field 개수
        self.field_index = field_index          # 인코딩된 X의 칼럼들이 본래 어디 소속이었는지

        # Parameters of FM Layer
        # w: capture 1st order interactions
        # V: capture 2nd order interactions
        self.w = tf.Variable(tf.random.normal(shape=[num_feature],
                                              mean=0.0, stddev=1.0), name='w')
        self.V = tf.Variable(tf.random.normal(shape=(num_field, embedding_size),
                                              mean=0.0, stddev=0.01), name='V')

    def call(self, inputs):
        x_batch = tf.reshape(inputs, [-1, self.num_feature, 1])
        # Parameter V를 field_index에 맞게 복사하여 num_feature에 맞게 늘림
        embeds = tf.nn.embedding_lookup(params=self.V, ids=self.field_index)

        # Deep Component에서 쓸 Input
        new_inputs = tf.math.multiply(x_batch, embeds)    # (num_batch, num_feature, embedding_size)

        # (num_batch, )
        linear_terms = tf.reduce_sum(
            tf.math.multiply(self.w, inputs), axis=1, keepdims=False)

        # (num_batch, )
        interactions = 0.5 * tf.subtract(
            tf.square(tf.reduce_sum(new_inputs, [1, 2])),
            tf.reduce_sum(tf.square(new_inputs), [1, 2])
        )

        linear_terms = tf.reshape(linear_terms, [-1, 1])
        interactions = tf.reshape(interactions, [-1, 1])

        y_fm = tf.concat([linear_terms, interactions], 1)

        return y_fm, new_inputs

