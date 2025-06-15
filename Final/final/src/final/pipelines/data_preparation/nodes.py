import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data

def drop_columns(data: pd.DataFrame) -> pd.DataFrame:
    data.drop(columns=["Message ID"], inplace=True)
    return data

def mapSpamHam(data: pd.DataFrame) -> pd.DataFrame:
    data['Spam/Ham'] = data['Spam/Ham'].map({'ham': 0, 'spam': 1})
    return data

#stosunek dużych liiter do małych
def ratio_upper_case_message(df: pd.DataFrame) -> pd.DataFrame:
    df['ratio_upper_case'] = df['Message'].str.count(r'[A-Z]') / df['Message'].str.count(r'[a-z]')
    return df

#Stosunek liter do znaków specjalny/cyfr
def ratio_special_case_message(df: pd.DataFrame) -> pd.DataFrame:
    df['ratio_special_case'] = df['Message'].str.count(r'[a-Z]') / (df['Message'].str.count(r'[0-9]') + df['Message'].str.count(r'[^\w\s]'))
    return df

#w temacie Stosunek dużych liter do małych {małe/duże+znakiSpecjalne}
def ratio_upper_and_special_case_subject(df: pd.DataFrame) -> pd.DataFrame:
    df['ratio_upper_and_special_case'] = (df['Subject'].str.count(r'[a-z]') /
    (df['Subject'].str.count(r'[A-Z]') + df['Subject'].str.count(r'[0-9]') + df['Subject'].str.count(r'[^\w\s]') ))
    return df

def generate_datetime_features(df: pd.DataFrame) ->pd.DataFrame:
    ### Extract date time features
    df['Date'] = pd.to_datetime(df['Date'])
    # Add a column for year
    df['year_num'] = df['Date'].dt.year
    # Add a column for month
    df['month_num'] = df['Date'].dt.month
    # Add a column for day of week
    df['dayofweek_num'] = df['Date'].dt.dayofweek
    # Add a column for day of month
    df['dayofmonth'] = df['Date'].dt.day
    # Add a column for day of year
    df['dayofyear_num'] = df['Date'].dt.day_of_year
    return df

def split_data(data, test_size=0.7, random_state=42):
    train, test = train_test_split(data, test_size=test_size, random_state=random_state, stratify=data["Spam/Ham"])
    return train, test
#TODO
# więcej miar nie tylko f1 roc mmc accuracy precision recall

def train_autogluon_model(train_data, target_column: str = "Spam/Ham"):
    # Definicja podstawowych modeli z ich konfiguracjami
    hyperparameters = {
        'GBM': [
            {
                'num_boost_round': 100,
                'learning_rate': 0.1
            }
        ],
        'RF': [
            {
                'n_estimators': 100,
                'max_depth': 6
            }
        ],
        'XGB': [
            {
                'n_estimators': 100,
                'max_depth': 6
            }
        ],
        'CAT': [
            {
                'iterations': 100,
                'depth': 6
            }
        ]
    }

    custom_generator = AutoMLPipelineFeatureGenerator(enable_text_special_features=False,
                                                      enable_text_ngram_features=False)
    predictor = TabularPredictor(label=target_column, eval_metric="f1").fit(
        train_data,
        hyperparameters=hyperparameters,
        feature_generator=custom_generator,
        time_limit=600
    )
    predictor.save("data/06_models/autogluon_predictor")
    return "data/06_models/autogluon_predictor"


# def evaluate_autogluon_model(model, test_data, target_column: str = "Spam/Ham"):
#     if(target_column != "Spam/Ham"):
#         performance = model.evaluate(test_data)
#         return performance
#     else:
#         y_true = test_data[target_column]
#         X_test = test_data.drop(columns=[target_column])
#
#         return model.evaluate_predictions(y_true=y_true, y_pred=model.predict(X_test))

from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    matthews_corrcoef, precision_score, recall_score
)

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

def save_report_as_csv(report: dict):
    df = pd.DataFrame(report.items(), columns=["metric", "value"])
    return df

def generate_leaderboard(predictor, test_data, target_column="Spam/Ham"):
    leaderboard_df = predictor.leaderboard(silent=True)

    # Pobierz predykcje
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

    # Stwórz DataFrame i połącz z leaderboardem
    metrics_df = pd.DataFrame(extra_metrics).set_index("model")
    leaderboard_df = leaderboard_df.set_index("model").join(metrics_df)

    return leaderboard_df.reset_index()