# Guardian of Truth

A rule-based feature extraction module for hallucination detection
in customer-service AI responses.

The task is binary classification:
- 0: correct response
- 1: hallucination or policy violation

The dataset covers airline, banking, retail, and telecom domains.

## Implemented components

- Prompt and tool-call parser: `src/parser.py`
- Rule engine: `src/rules/tools.py`
- Rule evaluation: `analyze_rules.py`
- ML feature export: `export_rule_features.py`
- Automated tests: `tests/`

## Eight rules

1. Unavailable tool usage
2. Tool argument schema violations
3. Ungrounded tool arguments
4. Placeholder argument values
5. Repeated failed tool calls
6. Invalid argument relationships
7. Ungrounded ZIP codes
8. Past flight costs included in upcoming-flight calculations

## Setup

Use Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
```

## Run tests

```bash
python -m pytest -q
```

## Analyze rules

Place `valid.jsonl` in the `data/` directory.

```bash
python analyze_rules.py
```

## Export ML features

Place these files in the `data/` directory:

- `split_train.parquet`
- `split_threshold.parquet`
- `split_holdout.parquet`

Run:

```bash
python export_rule_features.py
```

The script generates:

- `features/rules_train.csv`
- `features/rules_threshold.csv`
- `features/rules_holdout.csv`

Each output contains an example ID, label, eight binary rule
features, and `rule_any`.

Join exported features to the corresponding dataset by `id`.

## Evaluation notes

The rules are feature extractors, not the complete ML classifier.
`rule_any` is a diagnostic baseline.

The current dataset was used during rule development.
Consequently, its holdout split is not a fully independent
estimate of generalization performance.

Do not tune the final classifier using holdout labels.

## Data

Dataset files are excluded from Git by `.gitignore`.
Obtain the dataset separately and place it in `data/`.
