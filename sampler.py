import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from functools import reduce


def merge_predictions(predictions, user_key='user_id', item_key='item_id'):
    le_user = LabelEncoder()
    le_item = LabelEncoder()
    # объединение сессий генераторов - для последующего label encoding'а
    union_ar = reduce(np.union1d, (pred[user_key].unique()
                        for pred in predictions.values()))
    le_user.fit(union_ar)
    # объединение айтемов генераторов - для последующего label encoding'а
    union_ar = reduce(np.union1d, (pred[item_key].unique()
                        for pred in predictions.values()))
    le_item.fit(union_ar)

    # для каждого из генераторов составляем спарс-матрицу одного и того же размера
    # со скорами и общими энкодингами сессий и айтемов, закидываем её в список
    user2score_matrices = []
    for pred_df in predictions.values():
        pred_df['item_index'] = le_item.transform(
            pd.array(pred_df[item_key].values))
        pred_df['user_index'] = le_user.transform(
            pd.array(pred_df[user_key].values))
        sparce_matrix = sparse.csr_matrix(
            (pred_df.score,
                (pred_df['user_index'].values,
                pred_df['item_index'].values)),
            shape=(le_user.classes_.shape[0],
                    le_item.classes_.shape[0]),
        )
        user2score_matrices.append(sparce_matrix)

    # суммируем матрицы поэлементно
    mut_matrix = sum([matrix for matrix in user2score_matrices])
    final_mask = mut_matrix.nonzero()
    # формируем итоговую таблицу
    merged_df = pd.DataFrame()
    # берём только те взаимодействия, у которых скор ненулевой, и закидываем
    # в таблицу соотв. сессии и айтемы
    merged_df[user_key] = le_user.inverse_transform(final_mask[0])
    merged_df[item_key] = le_item.inverse_transform(final_mask[1])
    columns = [col for col in predictions.keys()]
    # а скоры от каждого генератора - в отдельный столбец
    for column, u2s_matrix in zip(columns, user2score_matrices):
        merged_df[f'{column}_score'] = np.squeeze(
            np.array(u2s_matrix[final_mask]))

    return merged_df
