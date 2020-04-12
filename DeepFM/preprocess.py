# Preprocess

import config

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

pd.options.display.max_rows = 50
pd.options.display.max_columns = 50

file = pd.read_csv('data/adult.data', header=None)
X = file.loc[:, 0:13]
Y = file.loc[:, 14].map({' <=50K': 0, ' >50K': 1})

X.columns = config.ALL_FIELDS

def get_X(X, all_fields, continuous_fields, categorical_fields):
    field_dict = dict()
    X_modified = pd.DataFrame()

    for index, col in enumerate(X.columns):
        if col not in all_fields:
            print("{} not included: Check your column list".format(col))
            raise ValueError

        if col in continuous_fields:
            scaler = MinMaxScaler()
            X_cont = pd.DataFrame(scaler.fit_transform(X[[col]]),
                                  columns=[col])

            field_dict[index] = col
            X_modified = pd.concat([X_modified, X_cont], axis=1)

        if col in categorical_fields:
            X_cat_col = pd.get_dummies(X[col], prefix=col, prefix_sep='-')
            field_dict[index] = list(X_cat_col.columns)
            X_modified = pd.concat([X_modified, X_cat_col], axis=1)

    print('X shape: {}'.format(X_modified.shape))

    return field_dict, X_modified


field_dict, X_modified = get_X(X, config.ALL_FIELDS, config.CONT_FIELDS, config.CAT_FIELDS)
X_train, X_test, Y_train, Y_test = train_test_split(X_modified, Y, test_size=0.2, stratify=Y)

