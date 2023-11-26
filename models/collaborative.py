import implicit
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import sparse


class Collaborative:
    def __init__(self, target='item_id', model_name='tfidf', k_top=50):
        self.target = target
        self.k_top = k_top
        models = {
            'tfidf': implicit.nearest_neighbours.TFIDFRecommender(K=k_top, num_threads=64),
            'bm25': implicit.nearest_neighbours.BM25Recommender(K=k_top, num_threads=64),
            'cosine': implicit.nearest_neighbours.CosineRecommender(K=k_top, num_threads=64),
        }
        self.model = models[model_name]

    def fit(self, train):
        if f'{self.target}_le' in train:
            train = train.drop(columns=f'{self.target}_le')

        le_item = LabelEncoder()
        train[f'{self.target}_le'] = le_item.fit_transform(train[self.target]).astype(np.int32)

        le_session_train = LabelEncoder()
        train['rating'] = 1
        train['user_id_le'] = le_session_train.fit_transform(train['user_id']).astype(np.int32)
        df_train = sparse.csr_matrix(
            (list(train['rating'] + 0.01), (list(train['user_id_le']), list(train[f'{self.target}_le'])))
        )

        self.model.fit(df_train.astype(np.float32), show_progress=False)

        self.item_encoding = train[[self.target, f'{self.target}_le']].drop_duplicates()

    def _predict_users(self, model, df_test, user_ids):
        predictions = []
        for i in user_ids:
            pred_user = model.recommend(
                userid=i,
                user_items=df_test[i],
                N=self.k_top,
                filter_already_liked_items=False,
                recalculate_user=True,
            )
            pred_user = np.vstack(pred_user).T
            predictions.append(np.c_[[i] * len(pred_user), pred_user])

        predictions = pd.DataFrame(np.vstack(predictions), columns=['user_id_le', f'{self.target}_le', 'score'])
        return predictions[predictions['score'] > 0]

    def predict(self, test):
        if f'{self.target}_le' in test:
            test = test.drop(columns=f'{self.target}_le')
        le_user = LabelEncoder()
        test['user_id_le'] = le_user.fit_transform(test['user_id']).astype(np.int32)
        test = test.merge(self.item_encoding, on=self.target, how='inner')
        test['rating'] = 1
        df_test = sparse.csr_matrix(
            (list(test['rating'] + 0.01), (list(test['user_id_le']), list(test[f'{self.target}_le']))),
            shape=(test['user_id_le'].max() + 1, self.item_encoding[f'{self.target}_le'].max() + 1),
        )

        test_users = le_user.transform(list(set(test['user_id'])))
        candidates = self._predict_users(self.model, df_test, test_users)
        candidates['user_id'] = le_user.inverse_transform(candidates['user_id_le'].astype(np.int32))
        candidates = candidates.merge(self.item_encoding, on=f'{self.target}_le', how='inner')
        candidates['rnk'] = candidates.groupby('user_id')['score'].rank(method='dense', ascending=False)

        candidates = candidates.drop(columns=['user_id_le', f'{self.target}_le']).reset_index(drop=True)

        return candidates
    
    def fit_predict(self, train):
        self.fit(train)
        return self.predict(train)
