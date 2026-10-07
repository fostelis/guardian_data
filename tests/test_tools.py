from src.parser import (
    ParsedPrompt,
    ToolArgument,
    ToolDefinition,
    ToolCall,
    ToolResponse,
)

from src.rules.tools import (
    check_tool_availability,
    check_tool_schema,
    check_argument_grounding,
    check_placeholders,
    check_repeated_failed_calls,
)


def make_parsed_prompt(
    tool_names: list[str],
) -> ParsedPrompt:
    """Создаёт простой ParsedPrompt для тестов."""

    tools = {
        name: ToolDefinition(
            name=name,
            description="test tool",
        )
        for name in tool_names
    }

    return ParsedPrompt(tools=tools)


def make_parsed_prompt_with_arguments() -> ParsedPrompt:
    """Создаёт ParsedPrompt с инструментом и аргументами."""

    tool = ToolDefinition(
        name="test_tool",
        description="test tool",
        arguments={
            "user_id": ToolArgument(
                name="user_id",
                type="string",
                required=True,
            ),
            "amount": ToolArgument(
                name="amount",
                type="integer",
                required=True,
            ),
        },
    )

    return ParsedPrompt(
        tools={
            "test_tool": tool,
        }
    )


# ----------------------------------------------------------------------
# TOOL AVAILABILITY TESTS
# ----------------------------------------------------------------------

def test_known_tool_is_allowed():
    parsed = make_parsed_prompt(
        ["get_customer_by_id"]
    )

    response = (
        '→ TOOL_CALL get_customer_by_id: '
        '{"customer_id": "C123"}'
    )

    result = check_tool_availability(
        parsed,
        response,
    )

    assert result.has_unknown_tool is False
    assert result.unknown_tool_count == 0
    assert result.unknown_tools == []


def test_unknown_tool_is_detected():
    parsed = make_parsed_prompt(
        ["get_customer_by_id"]
    )

    response = (
        '→ TOOL_CALL check_network_status: '
        '{"line_id": "L123"}'
    )

    result = check_tool_availability(
        parsed,
        response,
    )

    assert result.has_unknown_tool is True
    assert result.unknown_tool_count == 1
    assert result.unknown_tools == [
        "check_network_status"
    ]


def test_duplicate_unknown_tool_counted_once():
    parsed = make_parsed_prompt([])

    response = """
→ TOOL_CALL run_speed_test: {}
→ TOOL_CALL run_speed_test: {}
"""

    result = check_tool_availability(
        parsed,
        response,
    )

    assert result.has_unknown_tool is True
    assert result.unknown_tool_count == 1
    assert result.unknown_tools == [
        "run_speed_test"
    ]


def test_text_response_without_tool_call_is_allowed():
    parsed = make_parsed_prompt(
        ["get_customer_by_id"]
    )

    response = "Здравствуйте! Чем я могу вам помочь?"

    result = check_tool_availability(
        parsed,
        response,
    )

    assert result.has_unknown_tool is False
    assert result.unknown_tool_count == 0
    assert result.unknown_tools == []


# ----------------------------------------------------------------------
# TOOL SCHEMA TESTS
# ----------------------------------------------------------------------

def test_schema_correct_call():
    parsed = make_parsed_prompt_with_arguments()

    response = (
        '→ TOOL_CALL test_tool: '
        '{"user_id": "U123", "amount": 100}'
    )

    result = check_tool_schema(
        parsed,
        response,
    )

    assert result.has_schema_error is False
    assert result.error_count == 0


def test_schema_missing_required_argument():
    parsed = make_parsed_prompt_with_arguments()

    response = (
        '→ TOOL_CALL test_tool: '
        '{"user_id": "U123"}'
    )

    result = check_tool_schema(
        parsed,
        response,
    )

    assert result.has_schema_error is True
    assert result.issues[0].missing_arguments == [
        "amount"
    ]


def test_schema_unknown_argument():
    parsed = make_parsed_prompt_with_arguments()

    response = (
        '→ TOOL_CALL test_tool: '
        '{"user_id": "U123", "amount": 100, "fake": "abc"}'
    )

    result = check_tool_schema(
        parsed,
        response,
    )

    assert result.has_schema_error is True
    assert result.issues[0].unknown_arguments == [
        "fake"
    ]


def test_schema_wrong_argument_type():
    parsed = make_parsed_prompt_with_arguments()

    response = (
        '→ TOOL_CALL test_tool: '
        '{"user_id": "U123", "amount": "one hundred"}'
    )

    result = check_tool_schema(
        parsed,
        response,
    )

    assert result.has_schema_error is True
    assert result.issues[0].wrong_type_arguments == [
        "amount"
    ]


# ----------------------------------------------------------------------
# ARGUMENT GROUNDING TESTS
# ----------------------------------------------------------------------

def test_grounded_id_is_allowed():
    parsed = make_parsed_prompt(
        ["test_tool"]
    )

    prompt = (
        "Customer account_id is account_12345."
    )

    response = (
        '→ TOOL_CALL test_tool: '
        '{"account_id": "account_12345"}'
    )

    result = check_argument_grounding(
        parsed,
        response,
        prompt,
    )

    assert result.has_ungrounded_argument is False
    assert result.issue_count == 0


def test_ungrounded_id_is_detected():
    parsed = make_parsed_prompt(
        ["test_tool"]
    )

    prompt = (
        "Customer account_id is account_12345."
    )

    response = (
        '→ TOOL_CALL test_tool: '
        '{"account_id": "account_99999"}'
    )

    result = check_argument_grounding(
        parsed,
        response,
        prompt,
    )

    assert result.has_ungrounded_argument is True
    assert result.issue_count == 1

    assert result.issues[0].argument_name == (
        "account_id"
    )

    assert result.issues[0].value == (
        "account_99999"
    )


def test_non_id_argument_is_not_checked():
    parsed = make_parsed_prompt(
        ["test_tool"]
    )

    prompt = "Some previous context."

    response = (
        '→ TOOL_CALL test_tool: '
        '{"reason": "new_reason"}'
    )

    result = check_argument_grounding(
        parsed,
        response,
        prompt,
    )

    assert result.has_ungrounded_argument is False
    assert result.issue_count == 0


def test_unknown_tool_is_skipped_by_grounding():
    parsed = make_parsed_prompt([])

    prompt = "Some previous context."

    response = (
        '→ TOOL_CALL fake_tool: '
        '{"account_id": "invented_123"}'
    )

    result = check_argument_grounding(
        parsed,
        response,
        prompt,
    )

    assert result.has_ungrounded_argument is False
    assert result.issue_count == 0

# ----------------------------------------------------------------------
# PLACEHOLDER TESTS
# ----------------------------------------------------------------------

def test_placeholder_account_id_is_detected():
    parsed = make_parsed_prompt(
        ["test_tool"]
    )

    response = (
        '→ TOOL_CALL test_tool: '
        '{"account_id": "account_id"}'
    )

    result = check_placeholders(
        parsed,
        response,
    )

    assert result.has_placeholder is True
    assert result.issue_count == 1
    assert result.issues[0].argument_name == "account_id"


def test_placeholder_in_angle_brackets_is_detected():
    parsed = make_parsed_prompt(
        ["test_tool"]
    )

    response = (
        '→ TOOL_CALL test_tool: '
        '{"payment_id": "<payment_id>"}'
    )

    result = check_placeholders(
        parsed,
        response,
    )

    assert result.has_placeholder is True
    assert result.issue_count == 1


def test_real_id_is_not_placeholder():
    parsed = make_parsed_prompt(
        ["test_tool"]
    )

    response = (
        '→ TOOL_CALL test_tool: '
        '{"account_id": "account_728194"}'
    )

    result = check_placeholders(
        parsed,
        response,
    )

    assert result.has_placeholder is False
    assert result.issue_count == 0


def test_normal_text_is_not_placeholder():
    parsed = make_parsed_prompt(
        ["test_tool"]
    )

    response = (
        '→ TOOL_CALL test_tool: '
        '{"summary": "Customer needs assistance"}'
    )

    result = check_placeholders(
        parsed,
        response,
    )

    assert result.has_placeholder is False
    assert result.issue_count == 0

# ----------------------------------------------------------------------
# REPEATED FAILED CALL TESTS
# ----------------------------------------------------------------------

def test_repeated_failed_call_is_detected():
    parsed = make_parsed_prompt(
        ["get_user"]
    )

    parsed.tool_calls = [
        ToolCall(
            name="get_user",
            arguments={
                "email": "test@example.com",
            },
        )
    ]

    parsed.tool_responses = [
        ToolResponse(
            name="get_user",
            content="User not found",
            is_error=True,
        )
    ]

    response = (
        '→ TOOL_CALL get_user: '
        '{"email": "test@example.com"}'
    )

    result = check_repeated_failed_calls(
        parsed,
        response,
    )

    assert result.has_repeated_failed_call is True
    assert result.issue_count == 1
    assert result.issues[0].tool_name == "get_user"


def test_changed_arguments_are_allowed():
    parsed = make_parsed_prompt(
        ["get_user"]
    )

    parsed.tool_calls = [
        ToolCall(
            name="get_user",
            arguments={
                "email": "old@example.com",
            },
        )
    ]

    parsed.tool_responses = [
        ToolResponse(
            name="get_user",
            content="User not found",
            is_error=True,
        )
    ]

    response = (
        '→ TOOL_CALL get_user: '
        '{"email": "new@example.com"}'
    )

    result = check_repeated_failed_calls(
        parsed,
        response,
    )

    assert result.has_repeated_failed_call is False
    assert result.issue_count == 0


def test_successful_call_can_be_repeated():
    parsed = make_parsed_prompt(
        ["get_user"]
    )

    parsed.tool_calls = [
        ToolCall(
            name="get_user",
            arguments={
                "email": "test@example.com",
            },
        )
    ]

    parsed.tool_responses = [
        ToolResponse(
            name="get_user",
            content='{"user_id": "U123"}',
            is_error=False,
        )
    ]

    response = (
        '→ TOOL_CALL get_user: '
        '{"email": "test@example.com"}'
    )

    result = check_repeated_failed_calls(
        parsed,
        response,
    )

    assert result.has_repeated_failed_call is False
    assert result.issue_count == 0


def test_different_tool_is_not_repeated_failure():
    parsed = make_parsed_prompt(
        [
            "get_user",
            "search_user",
        ]
    )

    parsed.tool_calls = [
        ToolCall(
            name="get_user",
            arguments={
                "email": "test@example.com",
            },
        )
    ]

    parsed.tool_responses = [
        ToolResponse(
            name="get_user",
            content="User not found",
            is_error=True,
        )
    ]

    response = (
        '→ TOOL_CALL search_user: '
        '{"email": "test@example.com"}'
    )

    result = check_repeated_failed_calls(
        parsed,
        response,
    )

    assert result.has_repeated_failed_call is False
    assert result.issue_count == 0

def test_json_string_with_placeholder_is_detected():
    parsed = make_parsed_prompt(
        ["call_discoverable_agent_tool"]
    )

    response = (
        '→ TOOL_CALL call_discoverable_agent_tool: '
        '{"arguments": '
        '"{\\"account_id\\": '
        '\\"light_blue_account_id_placeholder\\"}"}'
    )

    result = check_placeholders(
        parsed,
        response,
    )

    assert result.has_placeholder is True
    assert result.issue_count == 1

    assert (
        result.issues[0].argument_name
        == "arguments.account_id"
    )

    assert (
        result.issues[0].value
        == "light_blue_account_id_placeholder"
    )


def test_normal_json_string_is_not_placeholder():
    parsed = make_parsed_prompt(
        ["call_discoverable_agent_tool"]
    )

    response = (
        '→ TOOL_CALL call_discoverable_agent_tool: '
        '{"arguments": '
        '"{\\"account_id\\": '
        '\\"light_blue_account_12345\\", '
        '\\"reason\\": '
        '\\"Customer upgraded account\\"}"}'
    )

    result = check_placeholders(
        parsed,
        response,
    )

    assert result.has_placeholder is False
    assert result.issue_count == 0