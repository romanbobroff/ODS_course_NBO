import pandas as pd


def calc_median_time_intervals(events, months=None, merge_cols_list=['user_id'], target='item_id'):
    max_date = events['purch_date'].max()
    if months:
        events = events[events['purch_date'] >= max_date - pd.Timedelta(weeks=months * 4)]

    cols = ['user_id', 'purch_date', target]
    targets = [target[:-3] + '1_id', target[:-3] + '2_id']   
    events = events[cols]
    pairs = pd.merge(events, events, on=merge_cols_list)

    cond1 = pairs[target + '_x'] != pairs[target + '_y']
    cond2 = pairs['purch_date_x'] < pairs['purch_date_y']
    pairs = pairs[cond1 & cond2]

    pairs = pairs.rename(columns={
        target + '_x': targets[0],
        target + '_y': targets[1],
        'purch_date_x': 'purch_date1',
        'purch_date_y': 'purch_date2',
        'user_id_x': 'user_id',
    })

    pairs['delta_t'] = pairs['purch_date2'] - pairs['purch_date1']
    pairs = pairs.sort_values(by=['user_id', *targets, 'delta_t'], ascending=True)
    pairs = pairs.drop_duplicates(subset=['user_id', *targets, 'delta_t'], keep='first')

    median_durations = pairs.groupby(targets)['delta_t'].median().reset_index()
    median_durations['rnk'] = median_durations.groupby(targets[0])['delta_t'].rank(method='first', ascending=True)
        
    return pairs, median_durations

def predict_durations(train, test, predictions, target='item_id', months_dur=2):
    _, median_durations = calc_median_time_intervals(train, months=months_dur, merge_cols_list=['user_id'], target=target)

    targets = [target[:-3] + '1_id', target[:-3] + '2_id']

    purchs_last = train[['user_id', 'purch_date', target]].drop_duplicates().sort_values(by=['user_id', 'purch_date'], ascending=True)
    purchs_last['last_purch_date'] = purchs_last['user_id'].map(purchs_last.groupby('user_id')['purch_date'].max())
    purchs_last = purchs_last[purchs_last['purch_date'] == purchs_last['last_purch_date']].drop(columns=['last_purch_date'])

    pred_durs = purchs_last.merge(predictions.rename(columns={target: targets[1]})[['user_id', targets[1]]], on='user_id', how='left')
    pred_durs = pred_durs.merge(median_durations.rename(columns={targets[0]: target})[[target, targets[1], 'delta_t']], how='left')
    pred_durs = pred_durs[pred_durs['delta_t'].notna()]
    pred_durs = pred_durs[['user_id', 'delta_t']].drop_duplicates()
    pred_durs['rnk'] = pred_durs.groupby('user_id')['delta_t'].rank(method='dense', ascending=True)
    purchs_last = purchs_last[['user_id', 'purch_date']].drop_duplicates()

    purchs_gt = test[['user_id', 'purch_date']].drop_duplicates()
    purchs_gt['rnk'] = purchs_gt.groupby('user_id').rank(method='dense', ascending=True)
    purchs_gt = purchs_gt.merge(purchs_last.rename(columns={'purch_date': 'last_purch_date'}), on='user_id', how='left')
    purchs_gt['delta_t'] = purchs_gt['purch_date'] - purchs_gt['last_purch_date']
    purchs_gt = purchs_gt[['user_id', 'delta_t', 'rnk']]

    res = pred_durs.merge(purchs_gt.rename(columns={'delta_t': 'delta_t_gt'}), on=['user_id', 'rnk'], how='inner')
    metrics = {}
    metrics['MAE'] = (res['delta_t'] - res['delta_t_gt']).abs().mean()
    print (metrics)

    return pred_durs