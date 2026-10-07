import pandas as pd


df = pd.read_parquet("data/valid.parquet")

print("Количество строк и столбцов:")
print(df.shape)

print("\nНазвания столбцов:")
print(df.columns.tolist())

print("\nПервые 5 строк:")
print(df.head())

df.to_json(
    "valid.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

print("\nГотово! Создан файл valid.jsonl")