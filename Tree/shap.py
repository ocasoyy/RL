# Santander Customar Satisfaction
# LightGBM, SHAP

import os
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer

BASE_DIR = os.path.join(os.getcwd(), 'Tree')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

pd.options.display.max_rows= 100
pd.options.display.max_columns= 100

# Data Load
# (76020, 371), (75818, 370)
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

train['var3'].replace(-999999, 2, inplace=True)
test['var3'].replace(-999999, 2, inplace=True)

Y = train['TARGET']
X = train.iloc[:, 1:-1]
X_test = test.iloc[:, 1:]

del train

# Split
X_train, X_valid, Y_train, Y_valid = train_test_split(X.values, Y.values,
                                                      test_size=0.1, random_state=522, stratify=Y.values)

train_set = lgb.Dataset(data=X_train, label=Y_train)
valid_set = lgb.Dataset(data=X_valid, label=Y_valid)

params = {'objective': 'binary',
          'boosting': 'gbdt',
          'tree_learner': 'serial',
          'num_threads': 0,
          'device_type': 'cpu',
          'num_leaves': 30,
          'learning_rate': 0.1,
          'seed': 522,
          'metric': ['binary_logloss', 'auc']}


# Train
booster = lgb.train(params=params, train_set=train_set, valid_sets=valid_set,
                    num_boost_round=200, early_stopping_rounds=50)

best_iter = booster.best_iteration
best_logloss = np.round(booster.best_score['valid_0']['binary_logloss'], 4)
best_auc = np.round(booster.best_score['valid_0']['auc'], 4)
print("Best Iteration: {}, Logloss: {}, AUC: {}".format(best_iter, best_logloss, best_auc))

booster.save_model(os.path.join(MODEL_DIR, 'model.txt'))
# booster = lgb.Booster(model_file=os.path.join(LOCAL_DATA_DIR, 'models/model.txt'))

# Plot Importance: type: ['split', 'gain']
lgb.plot_importance(booster=booster, max_num_features=20, importance_type='gain')



# Shap Value 구하기, 마지막 값은 explainer.expected_value와 동일
booster.predict(X.iloc[0, :], pred_contrib=True)


# Shap Value
# Binary Classification: shap_values 2개 --> shap_values[1]: shap value for positive class
shap.initjs()
explainer = shap.TreeExplainer(booster)
shap_values = explainer.shap_values(X)[1]

# 첫 번째 예측 값의 설명을 시각화
# explainer.expected_value: 전체 Y 값의 평균
# shap_values: 각 feature가 Y값에 미치는 영향
# booster.predict(X.iloc[0, :]) = explainer.expected_value + np.sum(shap_values[0, :])
# 예측 값 = Y 평균 + shap_value 합
shap.force_plot(explainer.expected_value[1], shap_values[0, :], X.iloc[0, :], matplotlib=True)

# 전체 예측 값의 시각화 (Jupyter에서만 가능)
shap.force_plot(explainer.expected_value, shap_values, X)

# 한 Feature가 모델의 예측 값에 미치는 영향 이해하기
# X축: Feature의 값, Y축: Feature의 Shap value
shap.dependence_plot("RM", shap_values, X)

# Overview: 어떤 Feature가 모델에 있어 가장 중요한지
# 색깔: Feature 자체의 값을 나타냄 (Red: 큼, Blue: 작음)
shap.summary_plot(shap_values, X, max_display=15, plot_type='dot')

# Mean Absolute Value of shap value: feature magnitude
shap.summary_plot(shap_values, X, plot_type='bar', max_display=15)


# Predict
def make_submission_data(test, preds, binarizer):
    ids = test['ID'].values.reshape(-1, 1)
    preds = binarizer.fit_transform(preds)
    print("Density: {:.4f}".format(np.sum(preds) / ids.shape[0]))
    submission = np.concatenate([ids, preds], axis=1).astype('int32')
    submission = pd.DataFrame(data=submission, columns=['ID', 'TARGET'])

    return submission


preds = booster.predict(X_test).reshape(-1, 1)
binarizer = Binarizer(threshold=0.3)
submission = make_submission_data(test, preds, binarizer)

submission.to_csv(os.path.join(BASE_DIR, 'submission.csv'), index=None)


