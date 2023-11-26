import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from scipy import sparse
from functools import reduce

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
    'ltv_quantity_cumul',
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

def calc_metrics(test_sessions, predicted, target='item_id', rnk_col='rnk', score_col='score', use_clusters=False):
    metrics = {}
    cluster_metrics = []
    for k in [1, 5, 10, 20]:
        predicted_topk = predicted[predicted[rnk_col] <= int(k)][['user_id', target, rnk_col]]
        merged = pd.merge(test_sessions, predicted_topk, on=['user_id', target], how='left')
        
        # recall@k
        metrics[f'recall@{k}'] = sum(merged[rnk_col].notna()) / len(merged)

        # recall@k_users
        merged['users_item_count'] = merged['user_id'].map(merged.groupby('user_id').size())
        merged['users_hits_count'] = merged['user_id'].map(merged[merged[rnk_col].notna()].groupby('user_id').size())
        merged['recall_user'] = merged['users_hits_count'] / merged['users_item_count']
        merged['recall_user'].fillna(0, inplace=True)
        metrics[f'recall@{k}_users'] = merged.drop_duplicates(subset='user_id')['recall_user'].mean()

        # users_hitted@k
        metrics[f'users_hitted@{k}'] = merged[merged['users_hits_count'].notna()]['user_id'].nunique() / merged['user_id'].nunique()

        if use_clusters:
            merged['clusters_item_count'] = merged[cluster_features[0]].map(merged.groupby(cluster_features[0]).size())
            merged['clusters_hits_count'] = merged[cluster_features[0]].map(merged[merged[rnk_col].notna()].groupby(cluster_features[0]).size())
            merged[f'recall@{k}_cluster'] = merged['clusters_hits_count'] / merged['clusters_item_count']
            merged[f'recall@{k}_cluster'].fillna(0, inplace=True)

            cluster_metrics.append(merged.drop_duplicates(subset=[cluster_features[0]])[[cluster_features[0], f'recall@{k}_cluster']])

    print (metrics)
    if use_clusters and (cluster_features[0] is not None):
        cluster_metrics = cluster_metrics[0].merge(cluster_metrics[1], on=cluster_features[0], how='left')\
            .merge(cluster_metrics[2], on=cluster_features[0], how='left')\
            .merge(cluster_metrics[3], on=cluster_features[0], how='left')
        # print (cluster_metrics)

    # if rnk_col == 'boosting_rnk':
    #     for k in [0.5, 0.7, 0.9]:
    #         hit_k = f'hit@{k}'
    #         df_merged[hit_k] = df_merged[score_col] >= k
    #         metrics[f'recall@{k}'] = (df_merged[hit_k] / df_merged['users_item_count']).sum() / df_merged['user_id'].nunique()
    #     print (metrics)

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
