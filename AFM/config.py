# Configuration File
# 모든 칼럼 이름, 연속형 변수/범주형 변수 목록을 List로 저장한다.

ORIGINAL_FIELDS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                   'marital-status', 'occupation', 'relationship',
                   'race', 'sex', 'capital-gain', 'capital-loss',
                   'hours-per-week', 'country']

CONT_FIELDS = ['age', 'fnlwgt', 'education-num',
               'capital-gain', 'capital-loss', 'hours-per-week']
CAT_FIELDS = ['workclass', 'education', 'marital-status', 'occupation',
              'relationship', 'race', 'sex', 'country']

ALL_FIELDS = CONT_FIELDS + CAT_FIELDS

NUM_FIELD = len(ALL_FIELDS)
NUM_CONT = len(CONT_FIELDS)

# Hyper-parameters for Experiment
BATCH_SIZE = 256
EMBEDDING_SIZE = 5
HIDDEN_SIZE = 16
