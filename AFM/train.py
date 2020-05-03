import config
from preprocess import get_modified_data
from AFM import AFM

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import BinaryAccuracy, AUC
from time import perf_counter


def get_data():
    file = pd.read_csv('data/adult.data', header=None)
    X = file.loc[:, 0:13]
    Y = file.loc[:, 14].map({' <=50K': 0, ' >50K': 1})

    X.columns = config.ORIGINAL_FIELDS
    X_modified, num_feature = get_modified_data(X, config.CONT_FIELDS, config.CAT_FIELDS)

    X_train, X_test, Y_train, Y_test = train_test_split(X_modified, Y, test_size=0.2, stratify=Y)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_train.values, tf.float32), tf.cast(Y_train, tf.float32))) \
        .shuffle(30000).batch(config.BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_test.values, tf.float32), tf.cast(Y_test, tf.float32))) \
        .shuffle(10000).batch(config.BATCH_SIZE)

    return train_ds, test_ds, num_feature


def train_on_batch(model, optimizer, acc, auc, inputs, targets):
    with tf.GradientTape() as tape:
        y_pred = model(inputs)
        loss = tf.keras.losses.binary_crossentropy(from_logits=False, y_true=targets, y_pred=y_pred)

    grads = tape.gradient(target=loss, sources=model.trainable_variables)

    # apply_gradients()를 통해 processed gradients를 적용함
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # accuracy & auc
    acc.update_state(targets, y_pred)
    auc.update_state(targets, y_pred)

    return loss


def train(epochs):
    train_ds, test_ds, num_feature = get_data()

    model = AFM(config.NUM_FIELD, num_feature, config.NUM_CONT,
                config.EMBEDDING_SIZE, config.HIDDEN_SIZE)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    print("Start Training: Batch Size: {}, Embedding Size: {}, Hidden Size: {}".\
        format(config.BATCH_SIZE, config.EMBEDDING_SIZE, config.HIDDEN_SIZE))
    start = perf_counter()
    for i in range(epochs):
        acc = BinaryAccuracy(threshold=0.5)
        auc = AUC()
        loss_history = []
        cnt = 0

        for x, y in train_ds:
            print(cnt)
            loss = train_on_batch(model, optimizer, acc, auc, x, y)
            loss_history.append(loss)
            cnt += 1

        print("Epoch {:03d}: 누적 Loss: {:.4f}, Acc: {:.4f}, AUC: {:.4f}".format(
            i, np.mean(loss_history), acc.result().numpy(), auc.result().numpy()))

    test_acc = BinaryAccuracy(threshold=0.5)
    test_auc = AUC()
    for x, y in test_ds:
        y_pred = model(x)
        test_acc.update_state(y, y_pred)
        test_auc.update_state(y, y_pred)

    print("테스트 ACC: {:.4f}, AUC: {:.4f}".format(test_acc.result().numpy(), test_auc.result().numpy()))
    print("Batch Size: {}, Embedding Size: {}".format(config.BATCH_SIZE, config.EMBEDDING_SIZE))
    print("걸린 시간: {:.3f}".format(perf_counter() - start))
    model.save_weights('weights/weights-epoch({})-batch({})-embedding({}).h5'.format(
        epochs, config.BATCH_SIZE, config.EMBEDDING_SIZE))


if __name__ == '__main__':
    train(epochs=100)
