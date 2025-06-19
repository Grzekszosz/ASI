import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    matthews_corrcoef, precision_score, recall_score
)

def read_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["Date"])
    return df

def ensure_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["Subject"] = df["Subject"].fillna("").astype(str)
    df["Message"] = df["Message"].fillna("").astype(str)
    return df

def drop_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = ensure_text_columns(data)
    return data.drop(columns=["Message ID"])

def map_spam_ham(data: pd.DataFrame) -> pd.DataFrame:
    data['Spam/Ham'] = data['Spam/Ham'].map({'ham': 0, 'spam': 1})
    return data

def ratio_upper_case_message(df: pd.DataFrame) -> pd.DataFrame:
    df['ratio_upper_case'] = df['Message'].str.count(r'[A-Z]') / df['Message'].str.count(r'[a-z]')
    return df

def ratio_special_case_message(df: pd.DataFrame) -> pd.DataFrame:
    df['ratio_special_case'] = df['Message'].str.count(r'[a-zA-Z]') / (
        df['Message'].str.count(r'[0-9]') + df['Message'].str.count(r'[^\w\s]'))
    return df

def ratio_upper_and_special_case_subject(df: pd.DataFrame) -> pd.DataFrame:
    df['ratio_upper_and_special_case'] = (
        df['Subject'].str.count(r'[a-z]') /
        (df['Subject'].str.count(r'[A-Z]') +
         df['Subject'].str.count(r'[0-9]') +
         df['Subject'].str.count(r'[^\w\s]'))
    )
    return df



def split_data(data: pd.DataFrame, test_size=0.25, random_state=None):
    train, test = train_test_split(data, test_size=test_size, random_state=random_state, stratify=data["Spam/Ham"])
    return train, test

def train_autogluon_model(train_data, target_column: str = "Spam/Ham"):
    hyperparameters = {
        'GBM': [{'num_boost_round': 50, 'learning_rate': 0.1, "lambda_l1": 1.0, "lambda_l2": 1.0}],
        'RF': [{'n_estimators': 50, 'max_depth': 4}],
        'XGB': [{'n_estimators': 50, 'max_depth': 4}],
        'CAT': [{'iterations': 50, 'depth': 4}],
        'XT': {},
        'KNN': {},
    }
    feature_generator = AutoMLPipelineFeatureGenerator(
        enable_text_special_features=True,
        enable_text_ngram_features=True,
        enable_raw_text_features=True
    )
    predictor = TabularPredictor(label=target_column, eval_metric="f1").fit(
        train_data,
        hyperparameters=hyperparameters,
        feature_generator=feature_generator,
        # num_stack_levels=2,
        time_limit=600,
        holdout_frac=0.4
    )

    importance_df = predictor.feature_importance(train_data)
    print("\n📊 Feature importance:\n", importance_df)
    predictor.save("data/06_models/autogluon_predictor")
    return predictor

def evaluate_autogluon_model(predictor, test_data, target_column="Spam/Ham"):
    y_true = test_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_pred = predictor.predict(X_test)
    y_pred_proba = predictor.predict_proba(X_test)[1] if predictor.problem_type == "binary" else None

    metrics = {
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_pred_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
    return metrics

def generate_leaderboard(predictor, test_data, target_column="Spam/Ham"):
    leaderboard_df = predictor.leaderboard(silent=True)
    y_true = test_data[target_column]
    X_test = test_data.drop(columns=[target_column])

    extra_metrics = []
    for model_name in leaderboard_df["model"]:
        y_pred = predictor.predict(X_test, model=model_name)
        y_pred_proba = predictor.predict_proba(X_test, model=model_name)[1] if predictor.problem_type == "binary" else None

        row = {
            "model": model_name,
            "f1": f1_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "mcc": matthews_corrcoef(y_true, y_pred),
        }
        if y_pred_proba is not None:
            row["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        extra_metrics.append(row)

    metrics_df = pd.DataFrame(extra_metrics).set_index("model")
    leaderboard_df = leaderboard_df.set_index("model").join(metrics_df)
    return leaderboard_df.reset_index()
