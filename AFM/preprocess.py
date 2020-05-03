# Preprocess
import config
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

file = pd.read_csv('data/adult.data', header=None)
X = file.loc[:, 0:13]
Y = file.loc[:, 14].map({' <=50K': 0, ' >50K': 1})

X.columns = config.ORIGINAL_FIELDS

def get_modified_data(X, continuous_fields, categorical_fields):

    X_cont = X[continuous_fields]
    X_cat = pd.DataFrame()

    scaler = MinMaxScaler()
    X_cont = pd.DataFrame(scaler.fit_transform(X_cont), columns=X_cont.columns)

    for col in categorical_fields:
        X_cat_col = pd.get_dummies(X[col], prefix=col, prefix_sep='-')
        X_cat = pd.concat([X_cat, X_cat_col], axis=1)

    X_modified = pd.concat([X_cont, X_cat], axis=1)
    num_feature = X_modified.shape[1]

    print('Data Prepared...')
    print('X shape: {}'.format(X_modified.shape))

    return X_modified, num_feature

