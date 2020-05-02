# Model 정의
import tensorflow as tf
from layers import Embedding_layer

tf.keras.backend.set_floatx('float32')

class AFM(tf.keras.Model):

    def __init__(self, num_field, num_feature, embedding_size):
        super(AFM, self).__init__()
        self.embedding_size = embedding_size    # k: 임베딩 벡터의 차원(크기)
        self.num_field = num_field              # m: 인코딩 이전 feature 수
        self.num_feature = num_feature          # p: 인코딩 이후 feature 수, m <= p

        # 1) Embedding Layer
        self.embedding_layer = Embedding_layer






        self.layers1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.2)


    def __repr__(self):
        pass


    def call(self, inputs):


        mask = tf.greater(a_tensor, 0)
        non_zero_array = tf.boolean_mask(a_tensor, mask)



        y_pred = 0

        return y_pred

