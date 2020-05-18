# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# import tensorflow.python.util.deprecation as deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False

# tf.config.list_physical_devices('GPU')

from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 초보자1: MNIST 예제
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, verbose=True)
model.evaluate(x_test, y_test, verbose=2)


#----------
# 초보자2: 자동차 연비 예측하기
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)

# 데이터 읽기
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

# 전처리
dataset.isna().sum()
dataset = dataset.dropna()
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

# 준비
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# 모델 만들기
def build_model():
  model = keras.Sequential([
    layers.Dense(units=64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(units=64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

  return model

# 조기종료 설정 & 학습
model = build_model()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(x=normed_train_data, y=train_labels, epochs=1000,
                    validation_split=0.2, verbose=True, callbacks=[early_stop])

# 모델 훈련과정 시각화
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(8,12))

  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.subplot(2,1,2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

plot_history(history)

# 테스트
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
test_predictions = model.predict(normed_test_data).flatten()

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

#----------
# 맞춤설정 - 즉시 실행
import cProfile
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf

# Version Check
print(tf.__version__)

# 텐서플로2에서는 즉시 실행이 기본으로 활성화되어 있음
tf.executing_eagerly()

# GPU 확인
tf.config.list_physical_devices('GPU')

# 연산 바로 확인
# 즉시 실행 활성화는 텐서플로 연산을 바로 평가하고 파이썬에게 알려줌
# tf.Tensor 객체는 계산 그래프에 있는 노드를 가리키는 간접 핸들 대신 구체적인 값을 참조함
# 나중에 실행하기 위해 생성된 계산 그래프가 없기 때문에 print 나 디버거를 통해 결과 검토가 쉬움
x = [[2.0]]
m = tf.matmul(x, x)
print("hello, {}".format(m))

a = tf.constant([[2, 3], [4, 5]])
a.numpy()

# 즉시훈련: 그래디언트 계산하기
# tf.GradientTape: 즉시 실행 중에 그래디언트를 계산하고 모델 훈련에 이용함
# 오직 하나의 그래디언트만 계산하고 또 호출하면 RuntimeError가 뜸
w = tf.Variable([[1.0]])

with tf.GradientTape() as tape:
    loss = w*w

grad = tape.gradient(target=loss, sources=w)
print(grad)

# 즉시훈련: 모델 훈련
# mnist 데이터 가져오기 및 포맷 맞추기
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

# tf.cast: int에서 float으로 바꾸는 등
dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
   tf.cast(mnist_labels, tf.int64)))
dataset = dataset.shuffle(1000).batch(32)

# 모델 생성
mnist_model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16,[3,3], activation='relu',
                         input_shape=(None, None, 1)),
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(10)
])

# 즉시 실행에서는 훈련을 하지 않아도 모델을 사용하고 결과를 점검할 수 있음
for images, labels in dataset.take(1):
  print("로짓: ", mnist_model(inputs=images[0:1]).numpy())

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_history = []

# y_pred가 원핫인코딩이면: CategoricalCrossentropy
# y_pred가 label이면: SparseCategoricalCrossentropy
# -- y_true.shape = [batch_size]
# -- y_pred.shape = [batch_size, num_classes]

# 그래디언트를 처리하기 전에 적용하고 싶으면 아래와 같은 방법을 취하면 된다.
# tf.GradientTape를 통해 gradients를 계산함
# gradients를 처리함
# apply_gradients()를 통해 processed gradients를 적용함
def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = mnist_model(inputs=images, training=True)

        # 결과의 형태를 확인하기 위해서 단언문 추가
        # 위에서 batch 32개로 설정함
        tf.debugging.assert_equal(logits.shape, (32, 10))

        loss_value = loss_object(y_true=labels, y_pred=logits)

    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(target=loss_value, sources=mnist_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))

def train():
  for epoch in range(3):
    for (batch, (images, labels)) in enumerate(dataset):
      train_step(images, labels)
    print ('에포크 {} 종료'.format(epoch))

train()

# 변수와 Optimizer
# tf.Variable: 자동 미분을 쉽게 하기 위해 학습하는 동안 변경된 tf.Tensor 값을 저장함
# 모델 파라미터는 class instance 변수로 캡슐화 할 수 있고,
# 이를 효과적으로 하려면 tf.Variable을 tf.GradientTape와 함께 사용하면 됨

class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.W = tf.Variable(5., name='weight')
    self.B = tf.Variable(10., name='bias')
  def call(self, inputs):
    return inputs * self.W + self.B

# 실험 데이터
NUM_EXAMPLES = 2000
training_inputs = tf.random.normal([NUM_EXAMPLES])
noise = tf.random.normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# 최적화할 손실함수
def loss(model, inputs, targets):
  error = model(inputs) - targets
  return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(target=loss_value, sources=[model.W, model.B])

# 정의:
# 1. 모델
# 2. 모델 파라미터에 대한 손실 함수의 미분
# 3. 미분에 기초한 변수 업데이트 전략
model = Model()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
print("초기 손실: {:.3f}".format(loss(model, training_inputs, training_outputs)))

# 반복 훈련
for i in range(300):
  grads = grad(model, training_inputs, training_outputs)
  optimizer.apply_gradients(zip(grads, [model.W, model.B]))
  if i % 20 == 0:
    print("스텝 {:03d}에서 손실: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

print("최종 손실: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))

#-----
# 토크 ON 세미나
# 3강
import numpy as np
import tensorflow as tf

tf.executing_eagerly()

array = np.array([[1.0, 2.0], [3.0, 4.0]])
print(tf.convert_to_tensor(array, dtype=tf.float32))

weight = tf.Variable(tf.random_normal_initializer(stddev=0.1)([5, 2]))
print(weight)

# Dataset
array2 = np.arange(10)
ten2 = tf.data.Dataset.from_tensor_slices(array2)
data = ten2.map(tf.square).shuffle(20).batch(2)

# 랜덤형 상수형 텐서
tf.random.normal(shape=(2, 2), mean=0.0, stddev=1.0)
tf.random.uniform(shape=(2, 2), minval=0, maxval=10, dtype='int32')

# Variable: 변할 수 있는 상태(예를 들어 weight)를 저장하는데 사용되는 텐서
# 초깃값을 사용해서 Variable 을 생성할 수 있음
tf.Variable(tf.random.normal(shape=(2, 2)))

# GradientTape 를 통해 경사도(Gradient)를 계산
# GradientTape 을 열게 되면, tape.watch() 를 통해 텐서를 확인하고
# 이 텐서를 입력함으로써 사용하는 미분 가능한 표현을 구성해야 함
a = tf.random.normal(shape=(2, 2))
b = tf.random.normal(shape=(2, 2))

with tf.GradientTape() as tape:
    # Start recording the history of operations applied to `a`
    tape.watch(a)

    c = tf.sqrt(tf.square(a) + tf.square(b))

    # What's the gradient of `c` with respect to `a`?
    # c를 a로 미분하라
    dc_da = tape.gradient(c, a)
    print(dc_da)

# 그런데 Variable 은 자동으로 watch 가 된다.
a = tf.Variable(a)

with tf.GradientTape() as tape:
  c = tf.sqrt(tf.square(a) + tf.square(b))
  dc_da = tape.gradient(c, a)
  print(dc_da)

#-----
# 선형 회귀 예제
import numpy as np
import tensorflow as tf
import random

input_dim = 2
output_dim = 1
learning_rate = 0.01

w = tf.Variable(tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(tf.zeros(shape=(output_dim, )))

def compute_predictions(features):
    return tf.matmul(features, w) + b

def compute_loss(labels, predictions):
    return tf.reduce_mean(tf.square(labels - predictions))

def train_on_batch(x, y):
    with tf.GradientTape() as tape:
        predictions = compute_predictions(x)
        loss = compute_loss(y, predictions)

        # loss를 [w, b]로 각각 미분하라
        dloss_dw, dloss_db = tape.gradient(loss, [w, b])

    # 변수 텐서의 값을 바꾸는 메서드
    # 그냥 바꾸면 변수형 텐서가 상수형 텐서로 바뀐다.
    # assign: 값을 완전히 할당
    # assign_add or assign_sub 값을 증가/감소시킴
    w.assign_sub(learning_rate * dloss_dw)
    b.assign_sub(learning_rate * dloss_db)
    return loss


# Prepare a dataset.
num_samples = 10000
negative_samples = np.random.multivariate_normal(
    mean=[0, 3], cov=[[1, 0.5],[0.5, 1]], size=num_samples)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0], cov=[[1, 0.5],[0.5, 1]], size=num_samples)
features = np.vstack((negative_samples, positive_samples)).astype(np.float32)
labels = np.vstack((np.zeros((num_samples, 1), dtype='float32'),
                    np.ones((num_samples, 1), dtype='float32')))

# Shuffle the data.
indices = np.random.permutation(len(features))
features = features[indices]
labels = labels[indices]

# Create a tf.data.Dataset object for easy batched iteration
# features: (20000, 2), labels: (20000, 1)
dataset = tf.data.Dataset.from_tensor_slices(tensors=(features, labels))
dataset = dataset.shuffle(buffer_size=1024).batch(256)

# 학습 진행
for epoch in range(10):
    for step, (x, y) in enumerate(dataset):
        loss = train_on_batch(x, y)
    print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))


# 학습 함수를 정적 그래프로 compile 함: tf.function 데커레이터
# 이렇게 하면 시간이 빨라짐
@tf.function
def train_on_batch(x, y):
    with tf.GradientTape() as tape:
        predictions = compute_predictions(x)
        loss = compute_loss(y, predictions)
        dloss_dw, dloss_db = tape.gradient(loss, [w, b])
    w.assign_sub(learning_rate * dloss_dw)
    b.assign_sub(learning_rate * dloss_db)
    return loss

#-----
# Keras API
from tensorflow.keras.layers import Layer

# Layer 기본 클래스
# y = wx + b
# output_dim = units
class Linear(Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()

        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype='float32'),
            trainable=True
        )

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units, ), dtype='float32'),
            trainable=True
        )

    def call(self, inputs):
        # forward 연산 식 설정
        return tf.matmul(inputs, self.w) + self.b

# Layer Instance 생성
linear_layer = Linear(units=4, input_dim=3)

y = linear_layer(inputs = tf.ones(shape=(10, 3)))
assert y.shape == (2, 4)

# 가중치 추적
print(linear_layer.weights)

# add_weight 를 통해 가중치를 생성할 수 있음 (Simple)
class Linear(Layer):
  """y = w.x + b"""

  def __init__(self, units=32):
      super(Linear, self).__init__()
      self.units = units

  def build(self, input_shape):
      self.w = self.add_weight(shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)
      self.b = self.add_weight(shape=(self.units,),
                               initializer='random_normal',
                               trainable=True)

  def call(self, inputs):
      return tf.matmul(inputs, self.w) + self.b


# Layer 조합
class MLP(Layer):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear_1 = Linear(units=32)
        self.linear_2 = Linear(units=32)
        self.linear_3 = Linear(units=10)

    # 이어 붙이기
    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)


#-----
# 미리 정의된 Layer

# call 메서드에서 training 인자에 대해
# BatchNormalization과 Dropout Layer는 학습과 추론 단계에서 다른 동작방식을 가짐
# 이런 Layer에 대해서는 call 메서드를 구현할 때 training 인자를 만들어주어야 한다.
class Dropout(Layer):
    def __init__(self, rate):
        super(Dropout, self).__init__()
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)


# Loss 클래스
import tensorflow as tf

bce = tf.keras.losses.BinaryCrossentropy()
y_true = [0.0, 1.0]
y_pred = [1.0, 0.0]

loss = bce(y_true=y_true, y_pred=y_pred)
print(loss.numpy())


# Metric 클래스
# Loss와 다르게 State를 가지기에 update_state 메서드를 사용해 상태를 갱신하고
# result를 사용해 scalar형 결과값을 요청할 수 있음
# 초기화하고자 할 때는 reset_state 메서드를 사용
# 사용자 평가 지표 함수를 만들기 위해서는 Metric 클래스의 하위 클래스를 만들면 됨
metric = tf.keras.metrics.AUC()
metric.update_state([0, 1, 1, 1], [0, 1, 0, 0])
print("중간 결과: ", metric.result().numpy())

metric.update_state([1, 1, 1, 1], [0, 0, 0, 0])
print("최종 결과: ", metric.result().numpy())


# Simple 학습 과정
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[:].reshape(60000, 784).astype('float32') / 255
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(buffer_size=1024).batch(64)

# Instantiate a simple classification model
model = tf.keras.Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(10)
])

# Instantiate a logistic loss function that expects integer targets.
# Sparse: 원핫이 아닌 정수형 레이블 대상
# from_logits: Softmax 를 거치지 않은 결과 값을 대상으로 한다는 뜻
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
optimizer = tf.keras.optimizers.Adam()

# Iterate over the batches of the dataset.
for step, (x, y) in enumerate(dataset):
    # Open a GradientTape.
    with tf.GradientTape() as tape:

        # Forward pass.
        logits = model(x)

        # Loss value for this batch.
        loss_value = loss(y, logits)

    # Get gradients of loss wrt the weights.
    gradients = tape.gradient(target=loss_value, sources=model.trainable_weights)

    # Update the weights of our linear layer.
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    # Update the running accuracy.
    accuracy.update_state(y, logits)

    # Logging.
    if step % 100 == 0:
        print('Step:', step)
        print('Loss from last step: %.3f' % loss_value)
        print('Total running accuracy so far: %.3f' % accuracy.result())

x_test = x_test[:].reshape(10000, 784).astype('float32') / 255
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(128)

accuracy.reset_states()  # This clears the internal state of the metric

for step, (x, y) in enumerate(test_dataset):
  logits = model(x)
  accuracy.update_state(y, logits)

print('Final test accuracy: %.3f' % accuracy.result())


#-------------------------------------------
# 5강 Image Segmentation / 코드가 정상적으로 작동하지 않음
# Page: https://www.tensorflow.org/tutorials/images/segmentation
# tf.data.experimental.AUTOTUNE: 텐서플로가 알아서 몇 개의 스레드를 이용할지 결정함
# model.fit을 쓸 때는 repeat을 써야 하고,
# GradientTape을 쓸 때는 repeat을 쓰면 무한 루프에 빠짐
# prefetch: 다음 배치를 디스크에서 미리 읽어와서 시간을 단축시킴
# Conv2DTranspose: 그냥 Conv2D와 다르게 stride=2를 하면 2배씩 커짐


#-------------------------------------------
# 텐서플로 튜토리얼: 정형 데이터 다루기
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# 데이터 준비
URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
dataframe.head()

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

# 입력 파이프라인: tf.data 를 사용해 데이터프레임을 감싸기
def df_to_dataset(df, shuffle=True, batch_size=32):
    df = df.copy()
    labels = df.pop('target')

    # dataset
    ds = tf.data.Dataset.from_tensor_slices(tensors=(dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds

batch_size = 5
train_ds = df_to_dataset(df=train, shuffle=True, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# 엿보기
for feature_batch, label_batch in train_ds.take(1):
    print('전체 특성:', list(feature_batch.keys()))
    print('나이 특성의 배치:', feature_batch['age'])
    print('타깃의 배치:', label_batch)


# 여러 종류의 특성 열 알아 보기
# 샘플 배치
example_batch = next(iter(train_ds))[0]

# 특성 열을 만들고 배치 데이터를 변환하는 함수
def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())


# 아래에 등장하는 feature_column은 위에서
# from tensorflow import feature_column 으로 Import 했음

#1: 실수형 열
age = feature_column.numeric_column(key="age")
demo(age)

#2: 버킷형 열
# 위에 demo(age)에서 뽑은 57, 44, 50, 51, 76을 구간화 한 것
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)

#3: 범주형 열
# 문자열 목록은 categorical_column_with_vocabulary_list를 사용해 리스트로 전달하거나
# categorical_column_with_vocabulary_file을 사용해 파일에서 읽을 수 있음
thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible']
)

thal_one_hot = feature_column.indicator_column(thal)
demo(thal_one_hot)

#4: 임베딩 열
# 범주형 열에 가능한 값이 많을 때는 임베딩 열을 사용하는 것이 최선일 수 있음
# 임베딩 열의 입력은 앞서 만든 범주형 열임
thal_embedding = feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)

#5: 해시 특성 열
# 가능한 값이 범주형 열을 표현하는 또다른 방법임
# 입력의 hash 값을 계산하여 hash_bucket_size 크기의 버킷 중 하나를 선택하여 문자열을 인코딩 함
# 이 열을 사용할 때는 어휘 목록을 제공할 필요가 없고,
# 공간을 절약하기 위해 실제 범주의 개수보다 작게 해서 해시 버킷의 크기를 정할 수 있음
# 단, 다른 문자열이 같은 버킷에 매핑될 수 있음
thal_hashed = feature_column.categorical_column_with_hash_bucket(
    'thal', hash_bucket_size=1000
)
demo(feature_column.indicator_column(thal_hashed))

#6: 교차 특성 열
# 여러 특성을 연결하여 하나의 특성으로 만듦, 모델이 특성의 조합에 대한 가중치를 학습할 수 있음
# crossed_column 은 모든 가능한 조합에 대한 해시 테이블을 만들지 않고
# hashed_column 매개 변수를 사용해 해시 테이블의 크기를 선택함
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(feature_column.indicator_column(crossed_feature))

####################
# 사용할 열 선택하기 #
feature_columns = []

# 수치형 열
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))

# 버킷형 열
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# 범주형 열
thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# 임베딩 열
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# 교차 특성 열
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)


# 특성 layer 만들기 (tf.keras.layers.DenseFeatures 사용)
feature_layer = tf.keras.layers.DenseFeatures(feature_columns=feature_columns)

# 모델
model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(units=128, activation='relu'),
    layers.Dense(units=128, activation='relu'),
    layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("정확도", accuracy)

