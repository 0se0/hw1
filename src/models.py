import scipy.stats as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from .evaluation import evaluate_model, time_series_cv


def get_base_models():
    return {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }


def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """Fit every base model; the best one is chosen by TimeSeriesSplit CV MSE, and the test
    set is used for reporting only."""
    results = {}
    best_model = None
    best_cv = float('inf')

    for name, model in get_base_models().items():
        print(f"\nTraining {name}...")

        cv_score = time_series_cv(model, X_train, y_train)
        print(f"CV MSE: {cv_score:.2f}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        eval_results = evaluate_model(y_test, y_pred, name)
        eval_results['cv_mse'] = cv_score
        results[name] = eval_results

        if cv_score < best_cv:
            best_cv = cv_score
            best_model = (name, model)

    return results, best_model


PART3_SEARCH_SPACE = {
    'Ridge Regression': (Ridge(), {'alpha': st.loguniform(1e-2, 1e3)}),
    'Lasso Regression': (Lasso(max_iter=20000), {'alpha': st.loguniform(1e-4, 1e1)}),
    'Elastic Net': (ElasticNet(max_iter=20000), {'alpha': st.loguniform(1e-4, 1e1), 'l1_ratio': st.uniform(0.0, 1.0)}),
    'Random Forest': (RandomForestRegressor(random_state=42), {
        'n_estimators': st.randint(100, 400),
        'max_depth': st.randint(3, 15),
        'min_samples_leaf': st.randint(1, 10),
    }),
    'Gradient Boosting': (GradientBoostingRegressor(random_state=42), {
        'n_estimators': st.randint(100, 400),
        'max_depth': st.randint(2, 6),
        'learning_rate': st.loguniform(1e-2, 3e-1),
    }),
    'XGBoost': (XGBRegressor(random_state=42), {
        'n_estimators': st.randint(100, 400),
        'max_depth': st.randint(2, 8),
        'learning_rate': st.loguniform(1e-2, 3e-1),
        'subsample': st.uniform(0.6, 0.4),
        'colsample_bytree': st.uniform(0.6, 0.4),
    }),
}


def tune_models(search_space, X_train, y_train, n_iter=20, n_splits=5, verbose=True):
    """RandomizedSearchCV per model with TimeSeriesSplit. Returns ({name: estimator}, {name: cv_mse})."""
    estimators, cv_mse = {}, {}
    for name, (estimator, distributions) in search_space.items():
        if verbose:
            print(f"\nTuning {name}...")
        search = RandomizedSearchCV(
            estimator, distributions, n_iter=n_iter, cv=TimeSeriesSplit(n_splits=n_splits),
            scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
        )
        search.fit(X_train, y_train)
        estimators[name] = search.best_estimator_
        cv_mse[name] = -search.best_score_
        if verbose:
            print(f"Best parameters: {search.best_params_}")
            print(f"CV MSE: {-search.best_score_:.2f}")
    return estimators, cv_mse


# Ranges rescaled to next-day-return magnitudes (targets are ~1e-2, not ~1e3)
PART4_SEARCH_SPACE = {
    'Ridge': (Ridge(), {'alpha': st.loguniform(1e-1, 1e4)}),
    'Lasso': (Lasso(max_iter=50000), {'alpha': st.loguniform(1e-6, 1e-2)}),
    'Elastic Net': (ElasticNet(max_iter=50000), {'alpha': st.loguniform(1e-6, 1e-2), 'l1_ratio': st.uniform(0.05, 0.9)}),
    'Random Forest': (RandomForestRegressor(random_state=42), {
        'n_estimators': st.randint(100, 300), 'max_depth': st.randint(2, 6), 'min_samples_leaf': st.randint(10, 40)}),
    'Gradient Boosting': (GradientBoostingRegressor(random_state=42), {
        'n_estimators': st.randint(50, 200), 'max_depth': st.randint(1, 4),
        'learning_rate': st.loguniform(5e-3, 1e-1), 'subsample': st.uniform(0.5, 0.5)}),
    'XGBoost': (XGBRegressor(random_state=42), {
        'n_estimators': st.randint(50, 200), 'max_depth': st.randint(1, 4),
        'learning_rate': st.loguniform(5e-3, 1e-1), 'subsample': st.uniform(0.5, 0.5),
        'colsample_bytree': st.uniform(0.5, 0.5)}),
}
