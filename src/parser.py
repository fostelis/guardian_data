import json
from dataclasses import dataclass
from typing import Any


@dataclass
class Example:
    """
    1 пример из датасета.
    """
    id: str
    prompt: str
    response: str
    label: int | None = None
    explanation: str | None = None


def load_jsonl(path: str) -> list[Example]:
    """
    Загружаем датасет из JSONL-файла.
    """
    examples = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            row: dict[str, Any] = json.loads(line)

            example = Example(
                id=row["id"],
                prompt=row["prompt"],
                response=row["response"],
                label=row.get("label"),
                explanation=row.get("explanation"),
            )

            examples.append(example)

    return examples


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "valid.jsonl"

    examples = load_jsonl(str(data_path))

    print(f"Загружено примеров: {len(examples)}")

    first = examples[0]

    print("\nПервый пример:")
    print("ID:", first.id)
    print("Label:", first.label)
    print("Prompt length:", len(first.prompt))
    print("Response length:", len(first.response))

    print("\nResponse:")
    print(first.response)