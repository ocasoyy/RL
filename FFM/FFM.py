from config import *
import xlearn as xl

model = xl.create_ffm()

# 학습/테스트 데이터 path 연결
model.setTrain("data/train_ffm.txt")
model.setValidate("data/test_ffm.txt")

# Early Stopping 불가
model.disableEarlyStop()

# param 선언
param = {'task': 'binary', 'lr': 0.2, 'lambda': 0.00002,
         'k': 3, 'epoch': 100, 'metric': 'auc', 'opt': 'adagrad',
         'num_threads': NUM_THREADS}

# 학습
# model.fit(param=param, model_path="model/model.out")

# Cross-Validation 학습
model.cv(param)

# Predict
model.setTest("data/test_ffm.txt")
model.setSigmoid()
model.predict("model/model.out", "output/predictions.txt")



