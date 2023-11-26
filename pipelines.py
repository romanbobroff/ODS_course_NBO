import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lib.utils import preprocessing, train_test_split, union_candidates, \
    label_candidates, add_features, fit_boosting, predict_boosting, calc_metrics
from models.collaborative import Collaborative
from models.markov_chain import MarkovChain
from models.toppopular import TopPopular, UserTopPopular
from lib.durations import predict_durations


def toppopular_pipeline(k_top, target, model_name, test_months):
    sessions = pd.read_pickle('data/transaction_and_features_2.pkl')
    sessions, _, _, _ = preprocessing(sessions)

    train, test = train_test_split(sessions, test_months=test_months)

    if model_name == 'toppop':
        candidates = TopPopular(k_top=k_top, target=target).fit_predict(train, test)
    else:
        candidates = UserTopPopular(k_top=k_top, target=target).fit_predict(train, test)

    metrics, _ = calc_metrics(test, candidates, target=target)

    predicted_durs = predict_durations(train, test, candidates, target=target)

    return candidates, predicted_durs

def collab_pipeline(k_top, target, model_name, test_months):
    sessions = pd.read_pickle('data/transaction_and_features_2.pkl')
    sessions, _, _, _ = preprocessing(sessions)

    train, test = train_test_split(sessions, test_months=test_months)

    candidates = Collaborative(target=target, k_top=k_top, model_name=model_name).fit_predict(train)

    metrics, _ = calc_metrics(test, candidates, target=target)

    predicted_durs = predict_durations(train, test, candidates, target=target)

    return candidates, predicted_durs

def markov_pipeline(k_top, target, test_months=2, shrink=True):
    sessions = pd.read_pickle('data/transaction_and_features_2.pkl')
    sessions, _, _, _ = preprocessing(sessions)

    train, test = train_test_split(sessions, test_months=test_months)

    candidates = MarkovChain(target=target).fit_predict(train, test, k_top=k_top, shrink=shrink)

    metrics, _ = calc_metrics(test, candidates, target=target)

    predicted_durs = predict_durations(train, test, candidates, target=target)

    return candidates, predicted_durs

def boosting_pipeline(k_top, target, model_names, test_months_collab=2, test_months_global=2):
    sessions = pd.read_pickle('data/transaction_and_features_2.pkl')
    sessions, items, users, _ = preprocessing(sessions)

    train, test = train_test_split(sessions, test_months=test_months_global)
    generator_train_data, generator_test_data = train_test_split(train, test_months=test_months_collab)

    # TF-IDF
    candidates_dict = {}
    candidates_test_dict = {}

    print ('Candidates generation...')
    for model_name in model_names:
        if model_name == 'tfidf':
            candidates = Collaborative(target=target, k_top=k_top, model_name='tfidf').fit_predict(generator_train_data)
            candidates_test = Collaborative(target=target, k_top=k_top, model_name='tfidf').fit_predict(train)
        elif model_name == 'cosine':
            candidates = Collaborative(target=target, k_top=k_top, model_name='cosine').fit_predict(generator_train_data)
            candidates_test = Collaborative(target=target, k_top=k_top, model_name='cosine').fit_predict(train)
        elif model_name == 'bm25':
            candidates = Collaborative(target=target, k_top=k_top, model_name='bm25').fit_predict(generator_train_data)
            candidates_test = Collaborative(target=target, k_top=k_top, model_name='bm25').fit_predict(train)
        elif model_name == 'markov':
            candidates = MarkovChain(target=target).fit_predict(generator_train_data, generator_test_data, k_top=k_top)
            candidates_test = MarkovChain(target=target).fit_predict(train, test, k_top=k_top)
        elif model_name == 'toppop':
            candidates = TopPopular(k_top=k_top, target=target).fit_predict(generator_train_data, generator_test_data)
            candidates_test = TopPopular(k_top=k_top, target=target).fit_predict(train, test)
        elif model_name == 'usertoppop':
            candidates = UserTopPopular(k_top=k_top, target=target).fit_predict(generator_train_data, generator_test_data)
            candidates_test = UserTopPopular(k_top=k_top, target=target).fit_predict(train, test)
        elif model_name == 'apriori':
            candidates = pd.read_parquet(f'target_{target}_{test_months_global}.par')
            candidates = candidates.drop_duplicates(subset=['user_id', target])
            candidates_test = candidates
        elif model_name == 'apriori_lastnext':
            candidates = pd.read_parquet(f'target_{target}_lastnext_{test_months_global}.par')
            candidates = candidates.drop_duplicates(subset=['user_id', target])
            candidates_test = candidates

        candidates_dict[model_name] = candidates
        candidates_test_dict[model_name] = candidates_test

    print ('Candidates sampling...')
    candidates = union_candidates(candidates_dict, target)

    print ('Pos and neg candidates generation...')
    boosting_df_train = label_candidates(candidates, generator_test_data, target)
    print ('Adding features...')
    boosting_df_train = add_features(boosting_df_train, users, items, target)

    print ('Fitting boosting...')
    model_boosting, feature_importances = fit_boosting(boosting_df_train, target)

    print ('Candidates sampling 1...')
    candidates_test = union_candidates(candidates_test_dict, target)

    print ('Add features 1...')
    boosting_df_test = add_features(candidates_test, users, items, target)
    print ('Predicting boosting...')
    predicted = predict_boosting(model_boosting, boosting_df_test, candidates_test, target)

    for gen in model_names:
        print (f'{gen} metrics:')
        metrics_collab, _ = calc_metrics(test, candidates_test_dict[gen], target=target, rnk_col='rnk', score_col='score')
        predicted_durs = predict_durations(train, test, candidates_test_dict[gen], target=target)

    print ('Boosting metrics:')
    metrics_boosting, cluster_metrics = calc_metrics(test, predicted, target=target, rnk_col='boosting_rnk', score_col='boosting_score')
    predicted_durs = predict_durations(train, test, predicted, target=target)

    df = pd.DataFrame({'feature_names': feature_importances.keys(), 'feature_importance': feature_importances.values()})
    df = df.sort_values(by=['feature_importance'], ascending=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=df['feature_importance'] / df['feature_importance'].sum() * 100, y=df['feature_names'])
    plt.title('Feature importance')
    plt.xlabel('Feature importance, %')
    plt.show();

    return predicted, predicted_durs, test
