
from pathlib import Path

from src.parser import load_jsonl, parse_prompt
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


PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data" / "valid.jsonl"


def calculate_metrics(results):
    """Calculate binary classification metrics."""
    tp = fp = fn = tn = 0

    for predicted, actual in results:
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

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


def print_metrics(name, metrics):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    for key in ("tp", "fp", "fn", "tn"):
        print(f"{key.upper()}: {metrics[key]}")

    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")


def analyze():
    examples = load_jsonl(str(DATA_PATH))

    rule_names = [
        "TOOL AVAILABILITY",
        "TOOL SCHEMA",
        "ARGUMENT GROUNDING",
        "PLACEHOLDER",
        "REPEATED FAILED CALL",
        "ARGUMENT RELATIONS",
        "ZIP GROUNDING",
        "FUTURE FLIGHT COST",
    ]

    results = {name: [] for name in rule_names}
    six_rules_results = []
    seven_rules_results = []
    eight_rules_results = []

    false_negatives = []
    false_positives = []

    print("=" * 70)
    print("RULE ANALYSIS")
    print("=" * 70)

    for example in examples:
        parsed = parse_prompt(example.prompt)
        response = example.response
        actual = example.label == 1

        availability = check_tool_availability(parsed, response)
        schema = check_tool_schema(parsed, response)
        grounding = check_argument_grounding(
            parsed, response, example.prompt
        )
        placeholders = check_placeholders(parsed, response)
        repeated = check_repeated_failed_calls(parsed, response)
        relations = check_argument_relations(response)

        zip_result = check_zip_grounding(
            original_prompt=example.prompt,
            response=response,
        )

        future_cost = check_future_flight_cost(
            original_prompt=example.prompt,
            response=response,
        )

        predictions = {
            "TOOL AVAILABILITY": availability.has_unknown_tool,
            "TOOL SCHEMA": schema.has_schema_error,
            "ARGUMENT GROUNDING": grounding.has_ungrounded_argument,
            "PLACEHOLDER": placeholders.has_placeholder,
            "REPEATED FAILED CALL": repeated.has_repeated_failed_call,
            "ARGUMENT RELATIONS": relations.has_argument_relation_error,
            "ZIP GROUNDING": zip_result.has_ungrounded_zip,
            "FUTURE FLIGHT COST": future_cost.has_future_flight_cost_error,
        }

        for name, predicted in predictions.items():
            results[name].append((predicted, actual))

        six_prediction = any(
            predictions[name] for name in rule_names[:6]
        )
        seven_prediction = any(
            predictions[name] for name in rule_names[:7]
        )
        eight_prediction = any(predictions.values())

        six_rules_results.append((six_prediction, actual))
        seven_rules_results.append((seven_prediction, actual))
        eight_rules_results.append((eight_prediction, actual))

        if eight_prediction:
            status = "TP" if actual else "FP"
            triggered = [
                name
                for name, predicted in predictions.items()
                if predicted
            ]

            print(f"\n[{status}] {example.id}")
            print("Triggered rules:", ", ".join(triggered))

            if availability.has_unknown_tool:
                print("  Unknown tools:", availability.unknown_tools)

            for issue in schema.issues:
                print(
                    "  Schema:",
                    issue.tool_name,
                    "missing=", issue.missing_arguments,
                    "unknown=", issue.unknown_arguments,
                    "wrong_type=", issue.wrong_type_arguments,
                )

            for issue in grounding.issues:
                print(
                    "  Ungrounded ID:",
                    issue.tool_name,
                    issue.argument_name,
                    repr(issue.value),
                )

            for issue in placeholders.issues:
                print(
                    "  Placeholder:",
                    issue.tool_name,
                    issue.argument_name,
                    repr(issue.value),
                )

            for issue in repeated.issues:
                print(
                    "  Repeated failed call:",
                    issue.tool_name,
                    issue.arguments,
                )

            for issue in relations.issues:
                print(
                    "  Argument relation:",
                    issue.relation_type,
                    issue.details,
                )

            for issue in future_cost.issues:
                print(
                    '  Past flights in calculation:',
                    issue.past_reservations,
                    issue.past_amounts,
                )

            for issue in zip_result.issues:
                print(
                    "  Ungrounded ZIP:",
                    issue.tool_name,
                    repr(issue.zip_value),
                    issue.details,
                )

        if actual and not eight_prediction:
            false_negatives.append(example.id)

        if not actual and eight_prediction:
            false_positives.append(example.id)

    print("\n" + "#" * 70)
    print("INDIVIDUAL RULE METRICS")
    print("#" * 70)

    for name in rule_names:
        print_metrics(name, calculate_metrics(results[name]))

    before = calculate_metrics(six_rules_results)
    after = calculate_metrics(seven_rules_results)

    print_metrics("ALL SIX RULES (BEFORE)", before)
    print_metrics("ALL SEVEN RULES (AFTER)", after)
    eight = calculate_metrics(eight_rules_results)
    print_metrics("ALL EIGHT RULES", eight)

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(f"Examples: {len(examples)}")
    print(f"F1 before: {before['f1']:.3f}")
    print(f"F1 after:  {after['f1']:.3f}")
    print(f"F1 change: {after['f1'] - before['f1']:+.3f}")

    print(f"TP before: {before['tp']}")
    print(f"TP after:  {after['tp']}")
    print(f"FP before: {before['fp']}")
    print("F1 with 8 rules:", format(eight["f1"], ".3f"))
    print("F1 gain (7 to 8):", format(eight["f1"] - after["f1"], "+.3f"))
    print("TP with 8 rules:", eight["tp"])
    print("FP with 8 rules:", eight["fp"])

    print(f"FP after:  {after['fp']}")

    print("\nFALSE NEGATIVES:")
    for example_id in false_negatives:
        print(" ", example_id)

    print("\nFALSE POSITIVES:")
    for example_id in false_positives:
        print(" ", example_id)


if __name__ == "__main__":
    analyze()
