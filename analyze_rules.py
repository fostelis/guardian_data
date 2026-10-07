from pathlib import Path

from src.parser import load_jsonl, parse_prompt
from src.rules.tools import (
    check_tool_availability,
    check_tool_schema,
    check_argument_grounding,
    check_placeholders,
    check_repeated_failed_calls,
)


project_root = Path(__file__).parent
data_path = project_root / "data" / "valid.jsonl"

examples = load_jsonl(str(data_path))


def calculate_metrics(results):
    """Считает TP, FP, FN, TN, Precision, Recall и F1."""

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for predicted, actual in results:
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

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
    """Красиво печатает метрики правила."""

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(f"TP: {metrics['tp']}")
    print(f"FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']}")
    print(f"TN: {metrics['tn']}")
    print()

    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")


availability_results = []
schema_results = []
grounding_results = []
placeholder_results = []
repeated_failed_results = []

all_rules_results = []


print("=" * 80)
print("RULE ANALYSIS")
print("=" * 80)


for example in examples:
    parsed = parse_prompt(example.prompt)

    # ------------------------------------------------------------------
    # Запускаем все checker'ы
    # ------------------------------------------------------------------

    availability = check_tool_availability(
        parsed=parsed,
        response=example.response,
    )

    schema = check_tool_schema(
        parsed=parsed,
        response=example.response,
    )

    grounding = check_argument_grounding(
        parsed=parsed,
        response=example.response,
        original_prompt=example.prompt,
    )

    placeholders = check_placeholders(
        parsed=parsed,
        response=example.response,
    )

    repeated_failed = check_repeated_failed_calls(
        parsed=parsed,
        response=example.response,
    )

    actual = example.label == 1

    # ------------------------------------------------------------------
    # Получаем предсказание каждого checker'а
    # ------------------------------------------------------------------

    availability_prediction = (
        availability.has_unknown_tool
    )

    schema_prediction = (
        schema.has_schema_error
    )

    grounding_prediction = (
        grounding.has_ungrounded_argument
    )

    placeholder_prediction = (
        placeholders.has_placeholder
    )

    repeated_failed_prediction = (
        repeated_failed.has_repeated_failed_call
    )

    # Диагностическое объединение всех правил.
    #
    # Пока это НЕ финальный классификатор.
    # Если сработало хотя бы одно правило,
    # считаем пример подозрительным.
    all_prediction = (
        availability_prediction
        or schema_prediction
        or grounding_prediction
        or placeholder_prediction
        or repeated_failed_prediction
    )

    # ------------------------------------------------------------------
    # Сохраняем результаты для подсчёта метрик
    # ------------------------------------------------------------------

    availability_results.append(
        (availability_prediction, actual)
    )

    schema_results.append(
        (schema_prediction, actual)
    )

    grounding_results.append(
        (grounding_prediction, actual)
    )

    placeholder_results.append(
        (placeholder_prediction, actual)
    )

    repeated_failed_results.append(
        (repeated_failed_prediction, actual)
    )

    all_rules_results.append(
        (all_prediction, actual)
    )

    # ------------------------------------------------------------------
    # Печатаем примеры, где сработало хотя бы одно правило
    # ------------------------------------------------------------------

    if all_prediction:
        status = "TP" if actual else "FP"

        print()
        print(f"[{status}] {example.id}")
        print(f"Label: {example.label}")

        if availability_prediction:
            print(
                "  unknown_tool:",
                availability.unknown_tools,
            )

        if schema_prediction:
            print("  schema_error:")

            for issue in schema.issues:
                print(
                    f"    tool: {issue.tool_name}"
                )

                if issue.missing_arguments:
                    print(
                        "      missing:",
                        issue.missing_arguments,
                    )

                if issue.unknown_arguments:
                    print(
                        "      unknown:",
                        issue.unknown_arguments,
                    )

                if issue.wrong_type_arguments:
                    print(
                        "      wrong_type:",
                        issue.wrong_type_arguments,
                    )

        if grounding_prediction:
            print("  ungrounded_argument:")

            for issue in grounding.issues:
                print(
                    f"    {issue.tool_name}."
                    f"{issue.argument_name}"
                    f" = {issue.value!r}"
                )

        if placeholder_prediction:
            print("  placeholder:")

            for issue in placeholders.issues:
                print(
                    f"    {issue.tool_name}."
                    f"{issue.argument_name}"
                    f" = {issue.value!r}"
                )

        if repeated_failed_prediction:
            print("  repeated_failed_call:")

            for issue in repeated_failed.issues:
                print(
                    f"    {issue.tool_name}"
                    f" {issue.arguments}"
                )


# ----------------------------------------------------------------------
# Считаем метрики
# ----------------------------------------------------------------------

availability_metrics = calculate_metrics(
    availability_results
)

schema_metrics = calculate_metrics(
    schema_results
)

grounding_metrics = calculate_metrics(
    grounding_results
)

placeholder_metrics = calculate_metrics(
    placeholder_results
)

repeated_failed_metrics = calculate_metrics(
    repeated_failed_results
)

all_metrics = calculate_metrics(
    all_rules_results
)


# ----------------------------------------------------------------------
# Печатаем метрики
# ----------------------------------------------------------------------

print_metrics(
    "TOOL AVAILABILITY",
    availability_metrics,
)

print_metrics(
    "TOOL SCHEMA",
    schema_metrics,
)

print_metrics(
    "ARGUMENT GROUNDING",
    grounding_metrics,
)

print_metrics(
    "PLACEHOLDER",
    placeholder_metrics,
)

print_metrics(
    "REPEATED FAILED CALL",
    repeated_failed_metrics,
)

print_metrics(
    "ALL FIVE RULES",
    all_metrics,
)


# ----------------------------------------------------------------------
# Краткая сводка
# ----------------------------------------------------------------------

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Всего примеров: {len(examples)}")

print(
    "Availability:",
    f"{availability_metrics['tp']} TP / "
    f"{availability_metrics['fp']} FP",
)

print(
    "Schema:",
    f"{schema_metrics['tp']} TP / "
    f"{schema_metrics['fp']} FP",
)

print(
    "Grounding:",
    f"{grounding_metrics['tp']} TP / "
    f"{grounding_metrics['fp']} FP",
)

print(
    "Placeholder:",
    f"{placeholder_metrics['tp']} TP / "
    f"{placeholder_metrics['fp']} FP",
)

print(
    "Repeated failed call:",
    f"{repeated_failed_metrics['tp']} TP / "
    f"{repeated_failed_metrics['fp']} FP",
)

print(
    "All five:",
    f"{all_metrics['tp']} TP / "
    f"{all_metrics['fp']} FP",
)