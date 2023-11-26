from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier


class Boosting():
    def __init__(self, params, target, drop_features, cat_features, selected_features):
        self.params = params
        self.target = target
        self.drop_features = drop_features
        self.cat_features = cat_features
        self.selected_features = selected_features

    def _preprocess_categories(self, df, is_train=True):
        for col in self.cat_features:
            df[col] = df[col].astype(int)

        if is_train:
            return df.drop(self.drop_features + [self.target], axis=1), df[self.target]
        else:
            return df.drop([*self.drop_features], axis=1), None

    def predict(self, model, df_test):
        X_test, _ = self._preprocess_categories(df_test[self.selected_features], is_train=False)
        return model.predict_proba(X_test)

    def fit(self, df_train):
        X, y = self._preprocess_categories(df_train[self.selected_features + [self.target]])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

        print(f'{X_train.shape[1]} features')
        print(X_train.columns)
        model = CatBoostClassifier(**self.params)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=self.cat_features, verbose=False)

        feature_importances = dict(zip(X_train.columns, model.feature_importances_))
        return model, feature_importances
