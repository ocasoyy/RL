# Preprocess
from itertools import repeat
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# pd.options.display.max_rows = 50
# pd.options.display.max_columns = 50

def get_X(X, all_fields, continuous_fields, categorical_fields):
    field_dict = dict()
    field_index = []
    X_modified = pd.DataFrame()

    for index, col in enumerate(X.columns):
        if col not in all_fields:
            print("{} not included: Check your column list".format(col))
            raise ValueError

        if col in continuous_fields:
            scaler = MinMaxScaler()
            X_cont = pd.DataFrame(scaler.fit_transform(X[[col]]),
                                  columns=[col])

            field_dict[index] = [col]
            field_index.append(index)
            X_modified = pd.concat([X_modified, X_cont], axis=1)

        if col in categorical_fields:
            X_cat_col = pd.get_dummies(X[col], prefix=col, prefix_sep='-')
            field_dict[index] = list(X_cat_col.columns)
            field_index.extend(repeat(index, X_cat_col.shape[1]))
            X_modified = pd.concat([X_modified, X_cat_col], axis=1)

    print('X shape: {}'.format(X_modified.shape))

    return field_dict, field_index, X_modified

