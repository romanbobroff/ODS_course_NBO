import numpy as np
import pandas as pd
from scipy.stats import multinomial
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from tqdm import tqdm


class MarkovChain:
    def __init__(self, target='item_id'):
        self.target_col = target
        self.target_cols = [self.target_col[:-3] + '1_id', self.target_col[:-3] + '2_id']

    def fit(self, train_sessions: pd.DataFrame):
        self._make_item_pair_in_sessions(train_sessions)

        i2i_transitions_num = self.df_sales_pair.groupby(self.target_cols).size().reset_index(name='i2i_transitions_num')
        self.df_sales_pair = self.df_sales_pair.merge(i2i_transitions_num, on=self.target_cols, how='left').drop_duplicates()

        self.le_item = LabelEncoder()
        train_items = list(set(self.df_sales_pair[self.target_cols[0]]) | set(self.df_sales_pair[self.target_cols[1]]))
        self.n_items = len(train_items)
        self.le_item.fit(train_items)
        self.df_sales_pair[self.target_cols[0] + '_le'] = self.le_item.transform(self.df_sales_pair[self.target_cols[0]])
        self.df_sales_pair[self.target_cols[1] + '_le'] = self.le_item.transform(self.df_sales_pair[self.target_cols[1]])

        self.observed_matrix = np.array(sparse.csr_matrix(
            (list(self.df_sales_pair['i2i_transitions_num']), 
            (list(self.df_sales_pair[self.target_cols[0] + '_le']), list(self.df_sales_pair[self.target_cols[1] + '_le']))),
            shape=(self.n_items, self.n_items),
        ).todense())

        obs_row_totals = self.observed_matrix.sum(axis=1)
        self.observed_p_matrix = np.nan_to_num(self.observed_matrix / obs_row_totals[:, None])

        uniform_p = 1 / self.n_items
        zero_row = np.argwhere(self.observed_p_matrix.sum(1) == 0).ravel()
        self.observed_p_matrix[zero_row, :] = uniform_p

        self.states = np.arange(len(train_items))

        return self.observed_matrix, self.observed_p_matrix

    def _make_item_pair_in_sessions(self, events):
        events['date_rnk'] = events.groupby('user_id')['purch_date'].rank(ascending=True)

        cols = ['user_id', self.target_col, 'date_rnk']
        events = events[cols]
        self.df_sales_pair = pd.merge(events, events, on='user_id')

        cond1 = self.df_sales_pair[self.target_col + '_x'] != self.df_sales_pair[self.target_col + '_y']
        cond2 = self.df_sales_pair['date_rnk_y'] - self.df_sales_pair['date_rnk_x'] == 1
        self.df_sales_pair = self.df_sales_pair[cond1 & cond2]

        self.df_sales_pair = self.df_sales_pair.rename(columns={
            self.target_col + '_x': self.target_cols[0],
            self.target_col + '_y': self.target_cols[1],
            'user_id_x': 'user_id',
        })
            
        self.df_sales_pair = self.df_sales_pair[['user_id', *self.target_cols]]

    def simulate(self, topk: int, start=None, seed=None):
        tf = self.observed_matrix
        fp = self.observed_p_matrix

        if start is None:
            row_totals = tf.sum(axis=1)
            _start = np.argmax(row_totals / tf.sum())
        elif isinstance(start, int):
            _start = start if start < len(self.states) else len(self.states) - 1
        elif start == 'random':
            _start = np.random.randint(0, len(self.states))
        elif isinstance(start, str):
            _start = np.argwhere(self.states == start).item()

        seq = np.zeros(topk, dtype=int)
        seq[0] = _start

        r_states = np.random.randint(0, topk, topk) if seed is None else seed

        for i in range(1, topk):
            _ps = fp[seq[i - 1]]
            _sample = np.argmax(
                multinomial.rvs(1, _ps, 1, random_state=r_states[i])
            )
            seq[i] = _sample

        preds = self.le_item.inverse_transform(seq)
        return pd.DataFrame({self.target_col: preds, 'rnk': np.arange(1, len(preds) + 1)})

    def _predict_user(self, user_id, user_items, topk):
        user_items = list(set(user_items) & set(self.le_item.classes_))
        user_items_le = self.le_item.transform(user_items)
        user_items_le = [1 if i in user_items_le else 0 for i in range(self.n_items)]
        scores = np.dot(np.array(user_items_le).reshape(1, -1), self.observed_p_matrix)[0]
        ids = scores > 0
        scores = scores[ids]
        predicted_items = np.arange(self.n_items)[ids]
        predicted_items = self.le_item.inverse_transform(predicted_items)
        predicted = pd.DataFrame({'user_id': [user_id] * len(scores), self.target_col: predicted_items, 'score': scores})
        predicted['rnk'] = predicted.groupby('user_id')['score'].rank(method='dense', ascending=False)
        predicted = predicted[predicted['rnk'] <= topk]
        return predicted
    
    def predict_users(self, train_sessions, test_users, k_top):
        predicted = []
        for user_id in tqdm(test_users):
            user_items = train_sessions[train_sessions['user_id'] == user_id][self.target_col].unique()
            preds = self._predict_user(user_id, user_items, k_top)
            predicted.append(preds)

        return pd.concat(predicted, axis=0)

    def predict(self, train_sessions, test_users, k_top, shrink=True):
        test_p = train_sessions.copy()
        test_p = test_p[test_p['user_id'].isin(test_users)]
        test_items = list(set(test_p[self.target_col]) & set(self.le_item.classes_))
        test_p = test_p[test_p[self.target_col].isin(test_items)]
        test_p[self.target_col + '_le'] = self.le_item.transform(test_p[self.target_col])
        test_p['rating'] = 1
        le_user = LabelEncoder()
        test_p['user_id_le'] = le_user.fit_transform(test_p['user_id'])

        user_item_matrix = np.array(sparse.csr_matrix(
            (list(test_p['rating'] + 0.01), 
            (list(test_p['user_id_le']), list(test_p[self.target_col + '_le']))),
            shape=(test_p['user_id_le'].max() + 1, self.n_items),
        ).todense())

        predicted = np.dot(user_item_matrix, self.observed_p_matrix)

        test_users = test_p['user_id'].unique()
        users = np.array([[user_id] * predicted.shape[1] for user_id in test_users]).reshape(1, -1)[0]
        items = np.array([self.le_item.inverse_transform(np.arange(self.n_items))] * len(test_users)).reshape(1, -1)[0]
        scores = predicted.reshape(1, -1)[0]
        predicted = pd.DataFrame({'user_id': users, self.target_col: items, 'score': scores})

        predicted = predicted[predicted['score'] > 0]
        predicted['rnk'] = predicted.groupby('user_id')['score'].rank(method='dense', ascending=False)
        predicted = predicted[predicted['rnk'] <= k_top]

        if shrink:
            rnk_size = predicted.groupby(['user_id', 'rnk']).size().reset_index(name='rnk_size')
            predicted = predicted.merge(rnk_size, on=['user_id', 'rnk'], how='left')
            predicted = predicted.sort_values(by=['user_id', 'rnk'], ascending=True)
            predicted['rnk_size_cumsum'] = predicted.groupby(['user_id', 'rnk'])['rnk_size'].cumsum()
            rnk_size_cumsum_last = predicted.groupby(['user_id', 'rnk'])['rnk_size_cumsum'].last().reset_index(name='rnk_size_cumsum_last')
            predicted = predicted.merge(rnk_size_cumsum_last, on=['user_id', 'rnk'], how='left')
            predicted = predicted[predicted['rnk_size_cumsum_last'] <= k_top]

        return predicted[['user_id', self.target_col, 'score', 'rnk']]

    def fit_predict(self, train_sessions, test_sessions, k_top, shrink=True):
        test_users = test_sessions['user_id'].unique()
        self.fit(train_sessions)
        return self.predict(train_sessions, test_users, k_top, shrink=shrink)
