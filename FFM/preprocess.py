import pandas as pd
from utils import _convert_to_ffm
from sklearn.model_selection import train_test_split
import config
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('data/train.csv')

cols = ['Education', 'ApplicantIncome', 'Loan_Status', 'Credit_History']
train_sub = train[cols]
train_sub['Credit_History'].fillna(0, inplace=True)
dict_ls = {'Y': 1, 'N': 0}
train_sub['Loan_Status'].replace(dict_ls, inplace=True)

train, test = train_test_split(train_sub, test_size=0.3, random_state=5)
print(f' train data: \n {train.head()}')

# Initialise fields and variables encoder
encoder = {"currentcode": len(config.NUMERICAL_FEATURES),  # Unique index for each numerical field or categorical variables
           "catdict": {},  # Dictionary that stores numerical and categorical variables
           "catcodes": {}}  # Dictionary that stores index for each categorical variables per categorical field

encoder = _convert_to_ffm('data/', train, 'train', config.GOAL[0],
                          config.NUMERICAL_FEATURES,
                          config.CATEGORICAL_FEATURES,
                          config.ALL_FEATURES,
                          encoder)

encoder = _convert_to_ffm('data/', test, 'test', config.GOAL[0],
                          config.NUMERICAL_FEATURES,
                          config.CATEGORICAL_FEATURES,
                          config.ALL_FEATURES,
                          encoder)
