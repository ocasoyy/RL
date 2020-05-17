# Model 정의
from layers import *

tf.keras.backend.set_floatx('float32')

class AFM(tf.keras.Model):

    def __init__(self, num_field, num_feature, num_cont, embedding_size, hidden_size):
        super(AFM, self).__init__()
        self.embedding_size = embedding_size    # k: 임베딩 벡터의 차원(크기)
        self.num_field = num_field              # m: 인코딩 이전 feature 수
        self.num_feature = num_feature          # p: 인코딩 이후 feature 수, m <= p
        self.num_cont = num_cont                # 연속형 field 수
        self.hidden_size = hidden_size          # Attention Pooling Layer Hidden Unit 수

        self.embedding_layer = Embedding_layer(num_field, num_feature,
                                               num_cont, embedding_size)
        self.pairwise_interaction_layer = Pairwise_Interaction_Layer(
            num_field, num_feature, embedding_size)
        self.attention_pooling_layer = Attention_Pooling_Layer(embedding_size, hidden_size)

        # Parameters
        self.w_0 = tf.Variable(tf.zeros([1]))
        self.w = tf.Variable(tf.zeros([num_feature]))
        self.p = tf.Variable(tf.random.normal(shape=(embedding_size, 1),
                                              mean=0.0, stddev=0.1))

        self.dropout = tf.keras.layers.Dropout(rate=config.DROPOUT_RATE)


    def __repr__(self):
        return "AFM Model: embedding{}, hidden{}".format(self.embedding_size, self.hidden_size)


    def call(self, inputs):
        # 1) Linear Term: (None, )
        linear_terms = self.w_0 + tf.reduce_sum(tf.multiply(self.w, inputs), 1)

        # 2) Interaction Term
        masked_inputs = self.embedding_layer(inputs)
        pairwise_interactions = self.pairwise_interaction_layer(masked_inputs)

        # Dropout and Attention Score
        pairwise_interactions = self.dropout(pairwise_interactions)
        attention_score = self.attention_pooling_layer(pairwise_interactions)

        # (None, 조합 수, embedding_size)
        attention_interactions = tf.multiply(pairwise_interactions, attention_score)

        # (None, embedding_size)
        final_interactions = tf.reduce_sum(attention_interactions, 1)

        # 3) Final: (None, )
        y_pred = linear_terms + tf.squeeze(tf.matmul(final_interactions, self.p), 1)
        y_pred = tf.nn.sigmoid(y_pred)

        return y_pred

