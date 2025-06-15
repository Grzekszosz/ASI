from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (drop_columns, ratio_upper_case_message,
                    ratio_special_case_message,
                    ratio_upper_and_special_case_subject,
                    generate_datetime_features, split_data
                    ,train_autogluon_model,evaluate_autogluon_model,save_report_as_csv
                    ,mapSpamHam,generate_leaderboard)
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([node(
        func=drop_columns,
        inputs=["mails"],
        outputs="cleared_data",
        name="drop_id_column_node"
    ),
    node (
        func=mapSpamHam,
        inputs="cleared_data",
        outputs="cleared_data_maps",
        name="mapSpamHam_node"
    ),
    node(
        func=ratio_upper_case_message,
        inputs="cleared_data_maps",
        outputs="with_ratio_upper_case_message",
        name="ratio_upper_case_message_node"
    ),
    node(
        func=ratio_upper_and_special_case_subject,
        inputs="with_ratio_upper_case_message",
        outputs="with_ratio_upper_and_special_case_subject",
        name="with_ratio_upper_and_special_case_subject_node"
    ),
    node(
        func=generate_datetime_features,
        inputs="with_ratio_upper_and_special_case_subject",
        outputs="final_data",
        name="final_data_node"
    ),
    node(
        func=split_data,
        inputs="final_data",
        outputs=["train_data", "test_data"],
        name="split_data_node"
    ),
    node(
        func=train_autogluon_model,
        inputs="train_data",
        outputs="autogluon_model",
        name="train_autogluon_model_node"
    ),
    node(
        func=evaluate_autogluon_model,
        inputs=["autogluon_model", "test_data"],
        outputs="evaluation_report",
        name="evaluate_model_node"
        ),
    node(
        func=save_report_as_csv,
        inputs="evaluation_report",
        outputs="report_saved",
        name="save_report_as_csv_node"
    ),
    node(
        func=generate_leaderboard,
        inputs=["autogluon_model", "test_data"],
        outputs="leaderboard",
        name="generate_leaderboard_node"
        )
    ])