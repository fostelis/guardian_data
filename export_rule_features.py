from pathlib import Path

import pandas as pd

from src.parser import parse_prompt
from src.rules.tools import (
    check_tool_availability,
    check_tool_schema,
    check_argument_grounding,
    check_placeholders,
    check_repeated_failed_calls,
    check_argument_relations,
    check_zip_grounding,
    check_future_flight_cost,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "features"

RULE_COLUMNS = [
    "rule_tool_availability",
    "rule_tool_schema",
    "rule_argument_grounding",
    "rule_placeholder",
    "rule_repeated_failed_call",
    "rule_argument_relations",
    "rule_zip_grounding",
    "rule_future_flight_cost",
]


def extract_features(prompt: str, response: str) -> dict:
    parsed = parse_prompt(prompt)

    availability = check_tool_availability(parsed, response)
    schema = check_tool_schema(parsed, response)
    grounding = check_argument_grounding(parsed, response, prompt)
    placeholders = check_placeholders(parsed, response)
    repeated = check_repeated_failed_calls(parsed, response)
    relations = check_argument_relations(response)

    zip_result = check_zip_grounding(
        original_prompt=prompt,
        response=response,
    )
    future_cost = check_future_flight_cost(
        original_prompt=prompt,
        response=response,
    )

    features = {
        "rule_tool_availability": int(availability.has_unknown_tool),
        "rule_tool_schema": int(schema.has_schema_error),
        "rule_argument_grounding": int(grounding.has_ungrounded_argument),
        "rule_placeholder": int(placeholders.has_placeholder),
        "rule_repeated_failed_call": int(repeated.has_repeated_failed_call),
        "rule_argument_relations": int(relations.has_argument_relation_error),
        "rule_zip_grounding": int(zip_result.has_ungrounded_zip),
        "rule_future_flight_cost": int(
            future_cost.has_future_flight_cost_error
        ),
    }

    features["rule_any"] = int(any(features.values()))
    return features


def calculate_metrics(df: pd.DataFrame) -> dict:
    predicted = df["rule_any"].astype(int)
    actual = df["label"].astype(int)

    tp = int(((predicted == 1) & (actual == 1)).sum())
    fp = int(((predicted == 1) & (actual == 0)).sum())
    fn = int(((predicted == 0) & (actual == 1)).sum())
    tn = int(((predicted == 0) & (actual == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "threshold", "holdout"):
        source = DATA_DIR / f"split_{split}.parquet"

        if not source.exists():
            raise FileNotFoundError(source)

        df = pd.read_parquet(source)

        required = {"id", "prompt", "response", "label"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{source.name} is missing columns: {sorted(missing)}"
            )

        rows = []

        for row in df.itertuples(index=False):
            features = extract_features(row.prompt, row.response)

            rows.append({
                "id": row.id,
                "label": int(row.label),
                **features,
            })

        output = pd.DataFrame(rows)

        columns = ["id", "label"] + RULE_COLUMNS + ["rule_any"]
        output = output[columns]

        destination = OUTPUT_DIR / f"rules_{split}.csv"
        output.to_csv(destination, index=False, encoding="utf-8")

        metrics = calculate_metrics(output)

        print()
        print("=" * 60)
        print(split.upper())
        print("=" * 60)
        print("Rows:", len(output))
        print("Saved:", destination)
        print("TP:", metrics["tp"])
        print("FP:", metrics["fp"])
        print("FN:", metrics["fn"])
        print("TN:", metrics["tn"])
        print("Precision:", f'{metrics["precision"]:.3f}')
        print("Recall:", f'{metrics["recall"]:.3f}')
        print("F1:", f'{metrics["f1"]:.3f}')
        print("Triggered per rule:")

        for column in RULE_COLUMNS:
            print(" ", column, int(output[column].sum()))


if __name__ == "__main__":
    main()
