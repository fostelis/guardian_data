from pathlib import Path
from src.parser import load_jsonl

project_root = Path(__file__).parent.parent
data_path = project_root / "data" / "valid.jsonl"

examples = load_jsonl(str(data_path))

markers = [
    "⟦SYSTEM⟧",
    "⟦USER",
    "⟦ASSISTANT",
    "⟦TOOL",
    "<instructions>",
    "</instructions>",
    "<policy>",
    "</policy>",
    "<tools>",
    "</tools>",
    "AVAILABLE TOOLS",
    "TOOL_CALL",
    "TOOL_RESULT",
]

# Берём по одному примеру каждого домена

domains = {}

for example in examples:
    domain = example.id.split("__")[0]

    if domain not in domains:
        domains[domain] = example


for domain, example in domains.items():
    print("\n" + "=" * 100)
    print(f"DOMAIN: {domain}")
    print(f"ID: {example.id}")
    print("=" * 100)

    prompt = example.prompt

    print("\nНАЙДЕННЫЕ МАРКЕРЫ:\n")

    for marker in markers:
        count = prompt.count(marker)

        if count > 0:
            print(f"{marker!r}: {count}")

    print("\n" + "-" * 100)
    print("СТРОКИ С ⟦...⟧:")
    print("-" * 100)

    for line in prompt.splitlines():
        stripped = line.strip()

        if stripped.startswith("⟦") and stripped.endswith("⟧"):
            print(stripped)