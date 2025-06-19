from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    read_data,
    drop_columns,
    map_spam_ham,
    ratio_upper_case_message,
    ratio_special_case_message,
    ratio_upper_and_special_case_subject,
    split_data,
    train_autogluon_model,
    evaluate_autogluon_model,
    generate_leaderboard
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=read_data,
            inputs="data",
            outputs="raw_df",
            name="read_data_node"
        ),
        node(
            func=drop_columns,
            inputs="raw_df",
            outputs="cleaned_df",
            name="drop_columns_node"
        ),
        node(
            func=map_spam_ham,
            inputs="cleaned_df",
            outputs="mapped_df",
            name="map_spam_ham_node"
        ),
        node(
            func=ratio_upper_case_message,
            inputs="mapped_df",
            outputs="upper_case_df",
            name="ratio_upper_case_message_node"
        ),
        node(
            func=ratio_special_case_message,
            inputs="upper_case_df",
            outputs="special_case_df",
            name="ratio_special_case_message_node"
        ),
        node(
            func=ratio_upper_and_special_case_subject,
            inputs="special_case_df",
            outputs="subject_case_df",
            name="ratio_upper_and_special_case_subject_node"
        ),
        node(
            func=split_data,
            inputs="subject_case_df",
            outputs=["train_df", "test_df"],
            name="split_data_node"
        ),
        node(
            func=train_autogluon_model,
            inputs="train_df",
            outputs="autogluon_model",
            name="train_model_node"
        ),
        node(
            func=evaluate_autogluon_model,
            inputs=["autogluon_model", "test_df"],
            outputs="eval_metrics",
            name="evaluate_model_node"
        ),
        node(
            func=generate_leaderboard,
            inputs=["autogluon_model", "test_df"],
            outputs="model_leaderboard",
            name="generate_leaderboard_node"
        )
    ])
