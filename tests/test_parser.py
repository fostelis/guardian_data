from src.parser import extract_policy


def test_extract_policy_ignores_policy_reference_in_instructions():
    prompt = """
⟦SYSTEM⟧
<instructions>
You are a customer service agent that helps the user
according to the <policy> provided below.
</instructions>

<policy>
# Airline Agent Policy

The current time is 2024-05-15 15:00:00 EST.

The agent must follow the airline policy.
</policy>
"""

    policy = extract_policy(prompt)

    assert policy.startswith("# Airline Agent Policy")
    assert "The current time is 2024-05-15" in policy

    assert "provided below" not in policy
    assert "<instructions>" not in policy


def test_extract_policy_returns_empty_string_when_missing():
    prompt = """
<instructions>
There is no actual policy block here.
</instructions>
"""

    policy = extract_policy(prompt)

    assert policy == ""


def test_extract_policy_preserves_policy_content():
    prompt = """
<instructions>
Follow the <policy> provided below.
</instructions>

<policy>
Rule one.
Rule two.
Rule three.
</policy>
"""

    policy = extract_policy(prompt)

    assert policy == "Rule one.\nRule two.\nRule three."