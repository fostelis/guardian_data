from pathlib import Path

from src.parser import load_jsonl, parse_prompt
from src.rules.tools import check_tool_availability


project_root = Path(__file__).parent
data_path = project_root / "data" / "valid.jsonl"

examples = load_jsonl(str(data_path))


# Счётчики для оценки правила
true_positive = 0
false_positive = 0
false_negative = 0
true_negative = 0


print("=" * 80)
print("TOOL AVAILABILITY CHECK")
print("=" * 80)


for example in examples:
    parsed = parse_prompt(example.prompt)

    result = check_tool_availability(
        parsed=parsed,
        response=example.response,
    )

    # Наше правило предсказывает галлюцинацию,
    # если response вызывает неизвестный инструмент.
    predicted_hallucination = result.has_unknown_tool

    actual_hallucination = example.label == 1

    if predicted_hallucination and actual_hallucination:
        true_positive += 1
        status = "TP"

    elif predicted_hallucination and not actual_hallucination:
        false_positive += 1
        status = "FP"

    elif not predicted_hallucination and actual_hallucination:
        false_negative += 1
        status = "FN"

    else:
        true_negative += 1
        status = "TN"

    # Печатаем только случаи, где правило сработало.
    if predicted_hallucination:
        print()
        print(f"[{status}] {example.id}")
        print(f"Label: {example.label}")
        print(f"Unknown tools: {result.unknown_tools}")


# Считаем метрики

if true_positive + false_positive > 0:
    precision = true_positive / (true_positive + false_positive)
else:
    precision = 0.0

if true_positive + false_negative > 0:
    recall = true_positive / (true_positive + false_negative)
else:
    recall = 0.0

if precision + recall > 0:
    f1 = 2 * precision * recall / (precision + recall)
else:
    f1 = 0.0


print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"Всего примеров: {len(examples)}")

print()
print(f"True Positive  (TP): {true_positive}")
print(f"False Positive (FP): {false_positive}")
print(f"False Negative  (FN): {false_negative}")
print(f"True Negative   (TN): {true_negative}")

print()
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1:        {f1:.3f}")