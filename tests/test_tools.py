from src.parser import (
    ParsedPrompt,
    ToolDefinition,
)

from src.rules.tools import check_tool_availability

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