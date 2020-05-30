# Train LightFM
import numpy as np
import pandas as pd

from scipy.io import mmread
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# Load Data
item_meta = pd.read_csv('data/books.csv')
item_meta = item_meta[['book_id', 'authors', 'average_rating', 'original_title']]

interactions = mmread('data/interactions.mtx')
item_features = mmread('data/item_features.mtx')
weights = mmread('data/weights.mtx')


# Split Train, Test data
train, test = random_train_test_split(interactions, test_percentage=0.1)
train, test = train.tocsr().tocoo(), test.tocsr().tocoo()
train_weights = train.multiply(weights).tocoo()


# Hyper Optimization
from collections import OrderedDict
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

# Define Search Space
space = OrderedDict(
    [('no_components', hp.choice('no_components', list(range(10, 30, 10)))),
     ('learning_schedule', hp.choice('learning_schedule', ['adagrad', 'adadelta']))
    ]
)

# Define Objective Function
def objective(params):
    no_components = params['no_components']
    learning_schedule = params['learning_schedule']

    model = LightFM(no_components=no_components,
                    learning_schedule=learning_schedule,
                    loss='warp',
                    learning_rate=0.05,
                    random_state=0)

    model.fit(interactions=train,
              item_features=item_features,
              sample_weight=train_weights,
              epochs=5,
              verbose=False)

    test_precision = precision_at_k(model, test, k=5, item_features=item_features).mean()
    # test_auc = auc_score(model, test, item_features=item_features).mean()
    output = -test_precision

    if np.abs(output+1) < 0.01 or output < -1.0:
        output = 0.0

    return output

best_model = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10)



# Find Similar Items
# def similar_items(item_id, item_features, model, N=10):
item_biases, item_embeddings = model.get_item_representations(features=item_features)

def make_best_items_report(item_embeddings, book_id, num_search_items=10):
    item_id = book_id - 1

    # Cosine similarity
    scores = item_embeddings.dot(item_embeddings[item_id])  # (10000, )
    item_norms = np.linalg.norm(item_embeddings, axis=1)    # (10000, )
    item_norms[item_norms == 0] = 1e-10
    scores /= item_norms

    # best: score가 제일 높은 item의 id를 num_search_items 개 만큼 가져온다.
    best = np.argpartition(scores, -num_search_items)[-num_search_items:]
    similar_item_id_and_scores = sorted(zip(best, scores[best] / item_norms[item_id]), key=lambda x: -x[1])

    # Report를 작성할 pandas dataframe
    best_items = pd.DataFrame(columns=['book_id', 'title', 'author', 'score'])

    for similar_item_id, score in similar_item_id_and_scores:
        book_id = similar_item_id + 1
        title = item_meta[item_meta['book_id'] == book_id].values[0][1]
        author = item_meta[item_meta['book_id'] == book_id].values[0][3]

        row = pd.Series([book_id, title, author, score], index=best_items.columns)
        best_items = best_items.append(row, ignore_index=True)

    return best_items


# book_id 2: Harry Potter and the Philosopher's Stone by J.K. Rowling, Mary GrandPré
# book_id 9: Angels & Demons by Dan Brown
report01 = make_best_items_report(item_embeddings, 2, 10)
report02 = make_best_items_report(item_embeddings, 9, 10)




