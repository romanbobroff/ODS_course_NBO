import numpy as np
import pandas as pd


class TopPopular:
    def __init__(self, target='item_id', user_key='user_id', k_top=50):
        self.target = target
        self.user_key = user_key
        self.k_top = k_top

    def fit(self, train):
        grouping = train.groupby(self.target).count()[self.user_key].reset_index()
        self.recs = (
            grouping.sort_values([self.user_key], ascending=False)[self.target]
            .reset_index(drop=True)
            .iloc[0 : self.k_top]
            .to_frame()
        )
    
    def predict(self, test_users):
        recs = list(self.recs[self.target].unique())
        n_users = len(test_users)
        user_ids = [[user_id] * self.k_top for user_id in test_users]
        user_ids = [user_id for user_idss in user_ids for user_id in user_idss]

        predicted = pd.DataFrame({
            self.user_key: user_ids, 
            self.target: recs * n_users, 
            'rnk': list(np.arange(1, self.k_top + 1)) * n_users,
        })
        predicted['score'] = 1 / predicted['rnk']
        return predicted
    
    def fit_predict(self, train, test):
        test_users = test['user_id'].unique()
        self.fit(train)
        return self.predict(test_users)


class UserTopPopular:
    def __init__(self, target='item_id', user_key='user_id', k_top=50):
        self.target = target
        self.user_key = user_key
        self.k_top = k_top

    def fit(self, train):
        self.recs = train.groupby([self.user_key, self.target]).size().reset_index(name='cnt')
        self.recs['rnk'] = self.recs.groupby(self.user_key)['cnt'].rank(method='dense', ascending=False)
        self.recs = self.recs.drop(columns='cnt')
    
    def predict(self, test_users):
        predicted = self.recs[self.recs[self.user_key].isin(test_users) & self.recs['rnk'] <= self.k_top]
        predicted['score'] = 1 / predicted['rnk']
        return predicted
    
    def fit_predict(self, train, test):
        test_users = test['user_id'].unique()
        self.fit(train)
        return self.predict(test_users)
