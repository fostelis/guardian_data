from dataclasses import dataclass

from src.parser import (
    ParsedPrompt,
    extract_tool_calls_from_text,
)


@dataclass
class ToolAvailabilityResult:
    """Результат проверки доступности вызываемых инструментов."""
    has_unknown_tool: bool
    unknown_tool_count: int
    unknown_tools: list[str]


def check_tool_availability(
    parsed: ParsedPrompt,
    response: str,
) -> ToolAvailabilityResult:
    """
    Проверяет, что инструменты, вызванные в candidate response,
    существуют среди AVAILABLE TOOLS.
    """

    available_tool_names = set(parsed.tools.keys())

    response_calls = extract_tool_calls_from_text(response)

    unknown_tools = []

    for call in response_calls:
        if call.name not in available_tool_names:
            unknown_tools.append(call.name)

    unknown_tools = list(dict.fromkeys(unknown_tools))

    return ToolAvailabilityResult(
        has_unknown_tool=len(unknown_tools) > 0,
        unknown_tool_count=len(unknown_tools),
        unknown_tools=unknown_tools,
    )