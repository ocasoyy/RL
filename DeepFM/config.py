# Configuration File
# 모든 칼럼 이름, 연속형 변수/범주형 변수 목록을 List로 저장한다.

ALL_FIELDS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
             'marital-status', 'occupation', 'relationship', 'race',
             'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'country']
CONT_FIELDS = ['age', 'fnlwgt', 'education-num',
               'capital-gain', 'capital-loss', 'hours-per-week']
CAT_FIELDS = list(set(ALL_FIELDS).difference(CONT_FIELDS))

# Hyper-parameters for Experiment
NUM_BIN = 10
BATCH_SIZE = 256
EMBEDDING_SIZE = 15
