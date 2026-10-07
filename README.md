# Guardian of Truth

Rule-based and model-assisted solution for hallucination detection in customer-service agent responses.

The task is binary classification:

- `0` — correct response;
- `1` — hallucinated or policy-violating response.

The dataset contains examples from four domains:

- airline;
- banking knowledge;
- retail;
- telecom.

## Project structure

```text
guardian_data/
├── data/
│   ├── valid.jsonl
│   └── valid.parquet
├── src/
│   ├── parser.py
│   ├── inspect_prompts.py
│   ├── inspect_structure.py
│   └── rules/
│       └── tools.py
├── tests/
│   └── test_tools.py
├── analyze_rules.py
├── convert.py
├── pytest.ini
└── README.md
```

## Current pipeline

The current implementation parses each prompt into structured components:

- instructions;
- policy;
- available tool definitions;
- dialogue messages;
- historical tool calls;
- historical tool responses.

The rule engine currently checks:

1. use of unavailable tools;
2. tool argument schema violations;
3. grounding of ID-like arguments;
4. placeholder values;
5. repeated tool calls that have already failed.

Rule outputs are intended to be used as features in the final classifier, rather than as the complete classification system.

## Tests

Run unit tests from the project root:

```bash
pytest -v
```

## Rule analysis

Run the rules on the validation set:

```bash
python analyze_rules.py
```

The script reports TP, FP, FN, TN, precision, recall and F1 for each rule and for their diagnostic combination.

## Data conversion

If `valid.parquet` is available, it can be converted to JSONL with:

```bash
python convert.py
```

The converted file is saved as:

```text
data/valid.jsonl
```

## Status

The project is under active development.

The current rule engine is only one component of the planned solution. Further policy, sequence, numerical, date and semantic verification will be added before the final classifier is built.
