import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from scipy import sparse
from functools import reduce
import json
import sys

sys.path.append('..')

from models.boosting import Boosting


cluster_features = [
    'cluster_cart',
    # 'label_dbscan',
    # 'label_kmeans_10',
]
item_features = [
    'item_id',
    # 'category_name',
    # 'category_id',
    'category_id_le',
    # 'description',
    # 'uom',
    'uom_le',
    'packing_size',
    'price_uom',
]
user_features = [
    'user_id',
    'user_id_le',
    'ltv_turnover',
    'ltv_quantity',
    'ltv_check_count',
    'ltv_purch_date_count',
    'ltv_item_count',
    'check_av_turn',
    'check_av_quintity',
    'check_av_item', 
    'frequence_client_per_month',
    'monetary',
    'frequency',
    'elasticity_client',
] + cluster_features

model_params = {
    'item_id': {
        'max_depth': 4,
        'n_estimators': 10000,
        'thread_count': 64,
        'random_state': 42,
        'verbose': 200,
        'early_stopping_rounds': 50,
    },
    'category_id': {
        'thread_count': 64,
        'random_state': 42,
        'verbose': 200,
        'early_stopping_rounds': 50,
    }
}

target = 'target'
drop_features = {
    'item_id': ['user_id', 'user_id_le', 'item_id'],
    'category_id': ['user_id', 'user_id_le'],
}

cat_features_user = cluster_features

cat_features = {
    'item_id': ['category_id_le', 'uom_le'] + cat_features_user,
    'category_id': [],
}
selected_features = {
    'item_id': item_features + user_features,
    'category_id': user_features,
}


def preprocessing(sessions):
    sessions = sessions.rename(columns={
        'id': 'user_id', 
        'item': 'item_id',
        'id_check_unique': 'session_id',
        'category': 'category_id',
    })
    
    sessions = sessions[sessions['is_purchase'] == 1]

    cond = sessions['elasticity_client'] == 'elastic'
    sessions.loc[cond, 'elasticity_client'] = 1
    sessions.loc[~cond, 'elasticity_client'] = 0
    sessions['elasticity_client'] = sessions['elasticity_client'].astype(bool)

    le_users = LabelEncoder()
    sessions['user_id_le'] = le_users.fit_transform(sessions['user_id'])

    sessions['category_id_le'] = LabelEncoder().fit_transform(sessions['category_id'])

    sessions['uom_le'] = LabelEncoder().fit_transform(sessions['uom'])

    items = sessions.drop_duplicates(subset='item_id')[item_features].reset_index(drop=True)
    users = sessions.drop_duplicates(subset='user_id', keep='last')[user_features].reset_index(drop=True)

    users.fillna(0, inplace=True)

    return sessions, items, users, le_users

def train_test_split(df, test_months=2):
    max_date = df['purch_date'].max()
    train = df[df['purch_date'] < max_date - pd.Timedelta(weeks=test_months * 4)]
    test = df[df['purch_date'] >= max_date - pd.Timedelta(weeks=test_months * 4)]

    return train, test

def union_predictions(predictions, user_col='user_id', item_col='item_id'):
    users = reduce(np.union1d, (pred[user_col].unique() for pred in predictions.values()))
    items = reduce(np.union1d, (pred[item_col].unique() for pred in predictions.values()))
    le_user = LabelEncoder()
    le_item = LabelEncoder()
    le_user.fit(users)
    le_item.fit(items)

    user_item = []
    for preds in predictions.values():
        preds['user_idx'] = le_user.transform(pd.array(preds[user_col].values))
        preds['item_idx'] = le_item.transform(pd.array(preds[item_col].values))
        matrix = sparse.csr_matrix(
            (preds['score'], (preds['user_idx'].values, preds['item_idx'].values)),
            shape=(le_user.classes_.shape[0], le_item.classes_.shape[0]),
        )
        user_item.append(matrix)

    mask = sum([matrix for matrix in user_item])
    mask = mask.nonzero()
    predictions_all = pd.DataFrame()
    predictions_all[user_col] = le_user.inverse_transform(mask[0])
    predictions_all[item_col] = le_item.inverse_transform(mask[1])
    cols = [col for col in predictions.keys()]
    for column, ui in zip(cols, user_item):
        predictions_all[f'{column}_score'] = np.squeeze(np.array(ui[mask]))

    return predictions_all

def union_candidates(predictions, target):
    candidates = union_predictions(predictions, item_col=target)

    for column in predictions:
        candidates = candidates.merge(
            predictions[column][['user_id', target, 'rnk']].rename(columns={'rnk': f'{column}_rnk'}),
            on=['user_id', target], how='left',
        )
    candidates.fillna(10000, inplace=True)

    return candidates

def calc_metrics(test, predicted, target='item_id', rnk_col='rnk', score_col='score', use_clusters=False):
    metrics = {}
    cluster_metrics = []

    for k in [1, 5, 10, 20]:
        predicted_topk = predicted[predicted[rnk_col] <= int(k)][['user_id', target, rnk_col]]
        merged = pd.merge(test, predicted_topk, on=['user_id', target], how='left')

        merged['users_item_count'] = merged['user_id'].map(merged.groupby('user_id').size())
        merged['users_hits_count'] = merged['user_id'].map(merged[merged[rnk_col].notna()].groupby('user_id').size())
        
        # recall@k
        merged['recall_user'] = merged['users_hits_count'] / merged['users_item_count']
        merged['recall_user'].fillna(0, inplace=True)
        metrics[f'recall@{str(k).zfill(2)}'] = merged.drop_duplicates(subset='user_id')['recall_user'].mean()

        # precision@k
        merged['precision_user'] = merged['users_hits_count'] / k
        merged['precision_user'].fillna(0, inplace=True)
        metrics[f'precision@{str(k).zfill(2)}'] = merged.drop_duplicates(subset='user_id')['precision_user'].mean()

        # users_hitted@k
        metrics[f'user_hitted@{str(k).zfill(2)}'] = merged[merged['users_hits_count'].notna()]['user_id'].nunique() / merged['user_id'].nunique()
    
    print (json.dumps(metrics, sort_keys=True, indent=4))

    return metrics, cluster_metrics

def label_candidates(candidates, test, target):
    positives = candidates.merge(test[['user_id', target]], on=['user_id', target], how='inner')
    positives['target'] = 1
    test['is_pos'] = 1
    negatives = candidates.merge(test, on=['user_id', target], how='left')
    gp = negatives.groupby('user_id')['is_pos'].sum()
    users_nonzero = gp[gp.values > 0].keys()
    negatives = negatives[negatives['is_pos'].isna() & negatives['user_id'].isin(users_nonzero)]
    negatives['target'] = 0
    negatives = negatives.drop(columns=['is_pos'])

    rnk_cols = candidates.columns[candidates.columns.str.contains('rnk')].values.tolist()
    score_cols = candidates.columns[candidates.columns.str.contains('score')].values.tolist()
    select_col = ['user_id', target] + rnk_cols + score_cols + ['target']
    return shuffle(pd.concat([positives, negatives])[select_col], random_state=2)

def add_features(candidates, users, items, target):
    candidates = candidates.merge(users, on='user_id', how='left')
    if target == 'item_id':
        candidates = candidates.merge(items, on='item_id', how='left')
    return candidates

#--------------------------------------------------------
# Boosting
#--------------------------------------------------------
def fit_boosting(df_train, target):
    rnk_cols = df_train.columns[df_train.columns.str.contains('rnk')].values.tolist()
    score_cols = df_train.columns[df_train.columns.str.contains('score')].values.tolist()

    ctb_model = Boosting(
        model_params[target], 
        'target', 
        drop_features[target], 
        cat_features[target], 
        selected_features[target] + rnk_cols + score_cols,
    )
    return ctb_model.fit(df_train)

def predict_boosting(model, df_test, candidates_test, target):
    rnk_cols = df_test.columns[df_test.columns.str.contains('rnk')].values.tolist()
    score_cols = df_test.columns[df_test.columns.str.contains('score')].values.tolist()

    boosting_model = Boosting(
        model_params[target], 
        target, 
        drop_features[target], 
        cat_features[target], 
        selected_features[target] + rnk_cols + score_cols,
    )
    boosting_predictions = boosting_model.predict(model, df_test)
    predicted = candidates_test.copy()
    predicted['boosting_score'] = boosting_predictions[:, 1]
    predicted['boosting_rnk'] = predicted.groupby('user_id')['boosting_score'].rank(ascending=False)
    return predicted

#--------------------------------------------------------
# Merge item and catetory predictions, calculate metrics
#--------------------------------------------------------
def merge_item_cat_preds(predicted_item, predicted_cat, test, items):
    predicted_cat = predicted_cat.drop_duplicates(subset=['user_id', 'category_id'])

    test = test[['user_id', 'item_id', 'category_id']]

    items = items.rename(columns={'item': 'item_id'})[['item_id', 'description']].drop_duplicates()
    predicted_item = predicted_item.merge(items, on='item_id', how='left')
    predicted_item['is_item']  = 1
    predicted_cat['is_cat'] = 1

    cols = ['user_id', 'item_id', 'category_id', 'description', 'boosting_score', 'boosting_rnk', 'is_item', 'is_cat']
    predicted_merge = pd.concat((predicted_item, predicted_cat), axis=0)[cols]
    predicted_merge['merge_rnk'] = predicted_merge.groupby('user_id')['boosting_score'].rank(method='dense', ascending=False)
    predicted_merge['target'] = predicted_merge['item_id'].fillna(predicted_merge['category_id'])

    return predicted_merge

def calc_merged_metrics(predicted_merge, test, items):
    items = items.rename(columns={'item': 'item_id'})[['item_id', 'description']].drop_duplicates()

    test = test.merge(items, on='item_id', how='left')

    test = test.merge(
        predicted_merge[['user_id', 'item_id', 'is_item', 'boosting_rnk']], 
        on=['user_id', 'item_id'], how='left',
    )
    test = test.merge(
        predicted_merge[['user_id', 'category_id', 'is_cat', 'boosting_rnk']], 
        on=['user_id', 'category_id'], how='left',
    )

    test['boosting_rnk'] = test['boosting_rnk_x'].fillna(test['boosting_rnk_y'])

    test.loc[(test['is_item'] == 1) | (test['is_cat'] == 1), 'is_hit'] = 1

    metrics = {}
    for k in [1, 5, 10, 20]:
        predicted_topk = test[test['boosting_rnk'] <= k]

        # recall@k
        metrics[f'recall@{str(k).zfill(2)}'] = (predicted_topk.groupby('user_id')['is_hit'].sum() / test.groupby('user_id').size()).mean()

        # metrics[f'precision@{str(k).zfill(2)}'] = (predicted_topk.groupby('user_id')['is_hit'].sum() / k).mean()

        # metrics[f'user_hitted@{str(k).zfill(2)}'] = predicted_topk[predicted_topk['users_hits_count'].notna()]['user_id'].nunique() / merged['user_id'].nunique()

    print (json.dumps(metrics, sort_keys=True, indent=4))

#--------------------------------------------------------
# Clustering
#--------------------------------------------------------
def clustering_info(users, labels, label_col):
    print (np.unique(labels))
    print (len([x for x in labels if x != -1]) / len(users))
    print (users[label_col].value_counts())
    # print ('silhouette_score:', silhouette_score(users[num_columns], labels, metric='euclidean'))

def clustering_viz(users, label_col):
    plt.figure(figsize=(10, 10))
    for label in users[label_col].unique():
        if label != -1:
            tmp = users[users[label_col] == label]
            plt.scatter(tmp['tsne_cat_1'], tmp['tsne_cat_2'], label=label)
