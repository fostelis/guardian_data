from pathlib import Path
from parser import load_jsonl

project_root = Path(__file__).parent.parent
data_path = project_root / "data" / "valid.jsonl"

examples = load_jsonl(str(data_path))

# Собираем по одному примеру каждого домена

domains = {}

for example in examples:
    domain = example.id.split("__")[0]

    if domain not in domains:
        domains[domain] = example


print("Найденные домены:")
for domain in domains:
    print(f"  - {domain}")


for domain, example in domains.items():
    print("\n" + "=" * 100)
    print(f"DOMAIN: {domain}")
    print(f"ID: {example.id}")
    print(f"PROMPT LENGTH: {len(example.prompt)}")
    print("=" * 100)

    # Пока выводим первые 6000 символов.
    print(example.prompt[:6000])