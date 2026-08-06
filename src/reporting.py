
import os
import pandas as pd

from src.config import (
    MASTER_SUMMARY_PATH,
    IDENTIFICATION_SUMMARY_PATH,
    VERIFICATION_SUMMARY_PATH,
    CLASSIFICATION_SUMMARY_PATH,
    RESULT_DF_PATH,
)


def clean_table(df, digits=4):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()
    df = df.dropna(axis=1, how="all")

    for col in df.select_dtypes(include=["float"]):
        df[col] = df[col].round(digits)

    return df


def style_table(df):
    if df is None or len(df) == 0:
        return df

    return (
        df.style
        .set_table_styles([
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("background-color", "white"),
                    ("color", "#111111"),
                    ("font-size", "14px"),
                ],
            },
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#f2f2f2"),
                    ("color", "#111111"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("border", "1px solid #d9d9d9"),
                    ("padding", "8px"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("background-color", "white"),
                    ("color", "#111111"),
                    ("text-align", "center"),
                    ("border", "1px solid #e0e0e0"),
                    ("padding", "8px"),
                ],
            },
            {
                "selector": "tbody th",
                "props": [
                    ("background-color", "white"),
                    ("color", "#111111"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("border", "1px solid #e0e0e0"),
                    ("padding", "8px"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even) td",
                "props": [
                    ("background-color", "#fafafa"),
                    ("color", "#111111"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even) th",
                "props": [
                    ("background-color", "#fafafa"),
                    ("color", "#111111"),
                ],
            },
        ])
    )


def build_master_summary_table(dataset_summary_df, split_summary_df, classification_summary_df):
    master_df = pd.concat(
        [dataset_summary_df, split_summary_df, classification_summary_df],
        ignore_index=True
    )
    return master_df


def _build_dataset_table(master_summary_df):
    dataset_df = master_summary_df[
        master_summary_df["section"].isin(["dataset", "dataset_rr"])
    ].copy()

    if len(dataset_df) == 0:
        return pd.DataFrame()

    dataset_df = dataset_df.pivot(index="section", columns="metric", values="value").reset_index()
    return clean_table(dataset_df)


def _build_split_table(master_summary_df):
    split_df = master_summary_df[
        master_summary_df["section"].isin(["split", "split_train", "split_val", "split_test"])
    ].copy()

    if len(split_df) == 0:
        return pd.DataFrame()

    split_df = split_df.pivot(index="section", columns="metric", values="value").reset_index()
    return clean_table(split_df)


def _build_classification_table(classification_summary_df):
    if classification_summary_df is None or len(classification_summary_df) == 0:
        return pd.DataFrame()

    df = classification_summary_df[["metric", "value"]].copy()
    df.columns = ["classification_metric", "value"]
    return clean_table(df)


def _build_identification_table(identification_summary_df):
    if identification_summary_df is None or len(identification_summary_df) == 0:
        return pd.DataFrame()

    df = identification_summary_df[["mode", "num_queries", "similarity_acc", "gallery_acc"]].copy()
    return clean_table(df)


def _build_verification_table(verification_summary_df):
    if verification_summary_df is None or len(verification_summary_df) == 0:
        return pd.DataFrame()

    df = verification_summary_df[["mode", "auc", "eer", "eer_threshold"]].copy()
    return clean_table(df)


def build_pretty_tables(master_summary_df, classification_summary_df, identification_summary_df, verification_summary_df):
    dataset_table = _build_dataset_table(master_summary_df)
    split_table = _build_split_table(master_summary_df)
    classification_table = _build_classification_table(classification_summary_df)
    identification_table = _build_identification_table(identification_summary_df)
    verification_table = _build_verification_table(verification_summary_df)

    return {
        "dataset_table": dataset_table,
        "split_table": split_table,
        "classification_table": classification_table,
        "identification_table": identification_table,
        "verification_table": verification_table,
    }


def print_pretty_tables(results):
    pretty = results["pretty_tables"]

    print("Dataset Summary")
    display(style_table(pretty["dataset_table"]))
    print("\n")

    print("Split Summary")
    display(style_table(pretty["split_table"]))
    print("\n")

    print("Classification Result")
    display(style_table(pretty["classification_table"]))
    print("\n")

    print("Identification Result")
    display(style_table(pretty["identification_table"]))
    print("\n")

    print("Verification Result")
    display(style_table(pretty["verification_table"]))
    print()


def _resolve_save_paths(output_dir=None):
    if output_dir is None:
        return {
            "master_summary": MASTER_SUMMARY_PATH,
            "identification_summary": IDENTIFICATION_SUMMARY_PATH,
            "verification_summary": VERIFICATION_SUMMARY_PATH,
            "classification_summary": CLASSIFICATION_SUMMARY_PATH,
            "result_df": RESULT_DF_PATH,
        }

    os.makedirs(output_dir, exist_ok=True)

    return {
        "master_summary": os.path.join(output_dir, "master_summary.csv"),
        "identification_summary": os.path.join(output_dir, "identification_summary.csv"),
        "verification_summary": os.path.join(output_dir, "verification_summary.csv"),
        "classification_summary": os.path.join(output_dir, "classification_summary.csv"),
        "result_df": os.path.join(output_dir, "result_df.csv"),
    }


def save_tables(
    master_summary_df,
    identification_summary_df,
    verification_summary_df,
    classification_summary_df,
    result_df,
    output_dir=None,
):
    save_paths = _resolve_save_paths(output_dir=output_dir)

    master_summary_df.to_csv(save_paths["master_summary"], index=False)
    identification_summary_df.to_csv(save_paths["identification_summary"], index=False)
    verification_summary_df.to_csv(save_paths["verification_summary"], index=False)
    classification_summary_df.to_csv(save_paths["classification_summary"], index=False)
    result_df.to_csv(save_paths["result_df"], index=False)

    print("Saved:", save_paths["master_summary"])
    print("Saved:", save_paths["identification_summary"])
    print("Saved:", save_paths["verification_summary"])
    print("Saved:", save_paths["classification_summary"])
    print("Saved:", save_paths["result_df"])
