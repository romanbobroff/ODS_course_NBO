import pandas as pd
import numpy as np
from replay.models.association_rules import AssociationRulesItemRec
from replay.utils import convert2spark


K_TOP = 50

def time_test_train_split(df, test_months=2):
    max_date = df['purch_date'].max()
    train = df[df['purch_date'] < max_date - pd.Timedelta(weeks=test_months * 4)]
    test = df[df['purch_date'] >= max_date - pd.Timedelta(weeks=test_months * 4)]

    return train, test

# min_item_count (int) – items with fewer sessions will be filtered out
# min_pair_count (int) – pairs with fewer sessions will be filtered out
def association_rules_fit_predict(
    train_sessions, test_sessions,
    session_col='id_check_unique_le', target='item_id', 
    min_item_count=1, min_pair_count=1,
):
    # Train
    train_copy = train_sessions[['user_id_le', target, session_col]].rename(columns={'user_id_le': 'user_idx', target: 'item_idx'})
    train_spark = convert2spark(train_copy)

    model = AssociationRulesItemRec(
        session_col=session_col, min_item_count=min_item_count, 
        min_pair_count=min_pair_count, num_neighbours=K_TOP,
    )
    model.fit(train_spark)

    # Predict on test
    test_copy = test_sessions[['user_id_le', target, session_col]].rename(columns={'user_id_le': 'user_idx', target: 'item_idx'})
    test_spark = convert2spark(test_copy)

    predicted = model.get_nearest_items(test_spark, k=K_TOP, metric='confidence') # output: ['item_idx', 'neighbour_item_idx', 'confidence', 'lift', 'confidence_gain']
    predicted = predicted.toPandas()
    predicted = predicted.rename(columns={'item_idx': target, 'neighbour_item_idx': 'item2_id', 'confidence': 'score'})
    predicted['rnk'] = predicted.groupby([target])['score'].rank(method='dense', ascending=False)

    # [item_id, item2_id, score, rnk]
    return predicted

def calc_metrics(test_sessions, predicted, target='item_id'):
    df_merged = test_sessions[['user_id', target]].merge(predicted, on=['user_id', target], how='left')
    df_merged['users_item_count'] = df_merged.groupby('user_id')['rnk'].transform(np.size)
    
    metrics = {}
    for k in [5, 10, 20]:
        hit_k = f'hit@{k}'
        df_merged[hit_k] = df_merged['rnk'] <= k
        metrics[f'recall@{k}'] = (df_merged[hit_k] / df_merged['users_item_count']).sum() / df_merged['user_id'].nunique()
    print (metrics)

    metrics = {}
    for k in [0.5, 0.7, 0.9]:
        hit_k = f'hit@{k}'
        df_merged[hit_k] = df_merged['score'] >= k
        metrics[f'recall@{k}'] = (df_merged[hit_k] / df_merged['users_item_count']).sum() / df_merged['user_id'].nunique()
    print (metrics)

    return metrics

def get_basket_predicted(sessions, i2i_recommendation):
    candidates = sessions.merge(
        i2i_recommendation[['item_id', 'item2_id', 'score']], on='item_id', how='left',
    )
    candidates_tmp = candidates.drop_duplicates(subset=['user_id', 'item_id', 'item2_id'])
    df_rank = candidates_tmp.groupby(['user_id', 'item2_id']).sum()['score'].reset_index()
    candidates = candidates.drop(columns=['score'])
    candidates = candidates.merge(df_rank, on=['user_id', 'item2_id']).reset_index()
    candidates['rnk'] = candidates.groupby('user_id')['score'].rank(method='first', ascending=False)

    return candidates[['user_id', 'item2_id', 'item_id', 'score', 'rnk']]