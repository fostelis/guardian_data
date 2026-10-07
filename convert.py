from pathlib import Path

import pandas as pd


project_root = Path(__file__).parent

input_path = project_root / "data" / "valid.parquet"
output_path = project_root / "data" / "valid.jsonl"


df = pd.read_parquet(input_path)

df.to_json(
    output_path,
    orient="records",
    lines=True,
    force_ascii=False,
)

print(f"Converted: {input_path}")
print(f"Saved to:  {output_path}")
print(f"Rows:      {len(df)}")