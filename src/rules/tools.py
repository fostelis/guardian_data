import json

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

@dataclass
class ToolSchemaIssue:
    tool_name: str
    missing_arguments: list[str]
    unknown_arguments: list[str]
    wrong_type_arguments: list[str]


@dataclass
class ToolSchemaResult:
    has_schema_error: bool
    error_count: int
    issues: list[ToolSchemaIssue]


def matches_type(value, expected_type: str) -> bool:
    """
    Проверяет соответствие Python-значения типу,
    указанному в описании инструмента.
    """

    expected_type = expected_type.lower()

    if expected_type == "string":
        return isinstance(value, str)

    if expected_type == "integer":
        # bool в Python является подклассом int,
        # поэтому исключаем его отдельно.
        return isinstance(value, int) and not isinstance(value, bool)

    if expected_type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        )

    if expected_type == "boolean":
        return isinstance(value, bool)

    if expected_type == "array":
        return isinstance(value, list)

    if expected_type == "object":
        return isinstance(value, dict)

    # Если встретился пока неизвестный нам тип,
    # не объявляем его ошибкой.
    return True


def check_tool_schema(
    parsed: ParsedPrompt,
    response: str,
) -> ToolSchemaResult:
    """
    Проверяет аргументы TOOL_CALL в candidate response.

    Проверяем:
    1. наличие обязательных аргументов;
    2. отсутствие неизвестных аргументов;
    3. соответствие типов.
    """

    response_calls = extract_tool_calls_from_text(response)

    issues = []

    for call in response_calls:
        # Неизвестные tools проверяет другой checker.
        # Здесь их пропускаем.
        if call.name not in parsed.tools:
            continue

        tool_definition = parsed.tools[call.name]

        expected_arguments = tool_definition.arguments
        actual_arguments = call.arguments

        missing_arguments = []
        unknown_arguments = []
        wrong_type_arguments = []

        # Проверяем обязательные аргументы.
        for argument_name, argument_definition in expected_arguments.items():
            if (
                argument_definition.required
                and argument_name not in actual_arguments
            ):
                missing_arguments.append(argument_name)

        # Проверяем переданные аргументы.
        for argument_name, value in actual_arguments.items():
            if argument_name not in expected_arguments:
                unknown_arguments.append(argument_name)
                continue

            expected_type = expected_arguments[argument_name].type

            if not matches_type(value, expected_type):
                wrong_type_arguments.append(argument_name)

        if (
            missing_arguments
            or unknown_arguments
            or wrong_type_arguments
        ):
            issues.append(
                ToolSchemaIssue(
                    tool_name=call.name,
                    missing_arguments=missing_arguments,
                    unknown_arguments=unknown_arguments,
                    wrong_type_arguments=wrong_type_arguments,
                )
            )

    return ToolSchemaResult(
        has_schema_error=len(issues) > 0,
        error_count=len(issues),
        issues=issues,
    )

@dataclass
class ArgumentGroundingIssue:
    tool_name: str
    argument_name: str
    value: str


@dataclass
class ArgumentGroundingResult:
    has_ungrounded_argument: bool
    issue_count: int
    issues: list[ArgumentGroundingIssue]


def is_id_argument(argument_name: str) -> bool:
    """
    Определяет, является ли аргумент идентификатором.
    """

    name = argument_name.lower()

    return (
        name == "id"
        or name.endswith("_id")
    )


def value_appears_in_prompt(
    value: str,
    prompt: str,
) -> bool:
    """
    Проверяет, встречалось ли значение раньше в prompt.
    """

    return value in prompt


def check_argument_grounding(
    parsed: ParsedPrompt,
    response: str,
    original_prompt: str,
) -> ArgumentGroundingResult:
    """
    Проверяет происхождение ID-подобных аргументов.

    Если модель передаёт ID в TOOL_CALL,
    значение должно уже встречаться в исходном prompt.
    """

    response_calls = extract_tool_calls_from_text(response)

    issues = []

    for call in response_calls:
        # Неизвестные инструменты уже проверяются
        # availability checker'ом.
        if call.name not in parsed.tools:
            continue

        for argument_name, value in call.arguments.items():
            if not is_id_argument(argument_name):
                continue

            # Пока проверяем только строковые ID.
            if not isinstance(value, str):
                continue

            if not value:
                continue

            if not value_appears_in_prompt(
                value,
                original_prompt,
            ):
                issues.append(
                    ArgumentGroundingIssue(
                        tool_name=call.name,
                        argument_name=argument_name,
                        value=value,
                    )
                )

    return ArgumentGroundingResult(
        has_ungrounded_argument=len(issues) > 0,
        issue_count=len(issues),
        issues=issues,
    )

@dataclass
class PlaceholderIssue:
    tool_name: str
    argument_name: str
    value: str


@dataclass
class PlaceholderResult:
    has_placeholder: bool
    issue_count: int
    issues: list[PlaceholderIssue]


PLACEHOLDER_VALUES = {
    "placeholder",
    "unknown",
    "none",
    "null",
    "n/a",
    "na",
    "tbd",
    "todo",
    "your_id",
    "user_id",
    "account_id",
    "customer_id",
    "payment_id",
    "reservation_id",
    "line_id",
}


def looks_like_placeholder(value: str) -> bool:
    """
    Проверяет, похоже ли строковое значение
    на placeholder вместо настоящего значения.
    """

    normalized = value.strip().lower()

    if normalized in PLACEHOLDER_VALUES:
        return True

    if normalized.startswith("<") and normalized.endswith(">"):
        return True

    if "placeholder" in normalized:
        return True

    return False


def find_placeholders_in_value(
    value,
    path: str,
) -> list[tuple[str, str]]:
    """
    Рекурсивно ищет placeholder-значения.

    Проверяет:
    - обычные строки;
    - словари;
    - списки;
    - JSON, который был передан как строка.
    """

    found = []

    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_path = f"{path}.{key}"

            found.extend(
                find_placeholders_in_value(
                    nested_value,
                    nested_path,
                )
            )

        return found

    if isinstance(value, list):
        for index, nested_value in enumerate(value):
            nested_path = f"{path}[{index}]"

            found.extend(
                find_placeholders_in_value(
                    nested_value,
                    nested_path,
                )
            )

        return found

    if not isinstance(value, str):
        return found

    stripped = value.strip()

    # Некоторые discoverable tools получают arguments
    # как JSON, записанный внутри обычной строки.
    if (
        stripped.startswith("{")
        or stripped.startswith("[")
    ):
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            decoded = None

        if isinstance(decoded, (dict, list)):
            return find_placeholders_in_value(
                decoded,
                path,
            )

    if looks_like_placeholder(value):
        found.append(
            (
                path,
                value,
            )
        )

    return found


def check_placeholders(
    parsed: ParsedPrompt,
    response: str,
) -> PlaceholderResult:
    """
    Ищет placeholder-значения в аргументах TOOL_CALL,
    включая вложенные словари, списки и JSON-строки.
    """

    response_calls = extract_tool_calls_from_text(response)

    issues = []

    for call in response_calls:
        # Неизвестные инструменты проверяются отдельно.
        if call.name not in parsed.tools:
            continue

        for argument_name, value in call.arguments.items():
            found = find_placeholders_in_value(
                value=value,
                path=argument_name,
            )

            for path, placeholder_value in found:
                issues.append(
                    PlaceholderIssue(
                        tool_name=call.name,
                        argument_name=path,
                        value=placeholder_value,
                    )
                )

    return PlaceholderResult(
        has_placeholder=len(issues) > 0,
        issue_count=len(issues),
        issues=issues,
    )

@dataclass
class RepeatedFailedCallIssue:
    tool_name: str
    arguments: dict


@dataclass
class RepeatedFailedCallResult:
    has_repeated_failed_call: bool
    issue_count: int
    issues: list[RepeatedFailedCallIssue]


def normalize_arguments(arguments: dict) -> dict:
    """
    Возвращает аргументы в форме,
    удобной для сравнения двух TOOL_CALL.
    """

    return arguments


def check_repeated_failed_calls(
    parsed: ParsedPrompt,
    response: str,
) -> RepeatedFailedCallResult:
    """
    Проверяет, повторяет ли candidate response
    тот же TOOL_CALL, который раньше уже завершился ошибкой.
    """

    response_calls = extract_tool_calls_from_text(response)

    issues = []

    # Исторические вызовы и ответы на них находятся
    # в parsed.tool_calls и parsed.tool_responses.
    #
    # Parser сохраняет их в порядке появления,
    # поэтому сопоставляем вызов с ответом
    # по имени инструмента и позиции в истории.

    failed_calls = []

    response_index = 0

    for historical_call in parsed.tool_calls:
        matching_response = None

        for index in range(
            response_index,
            len(parsed.tool_responses),
        ):
            historical_response = (
                parsed.tool_responses[index]
            )

            if (
                historical_response.name
                == historical_call.name
            ):
                matching_response = historical_response
                response_index = index + 1
                break

        if (
            matching_response is not None
            and matching_response.is_error
        ):
            failed_calls.append(historical_call)

    # Теперь смотрим, повторил ли candidate
    # какой-либо из уже провалившихся вызовов.
    for candidate_call in response_calls:
        for failed_call in failed_calls:
            same_tool = (
                candidate_call.name
                == failed_call.name
            )

            same_arguments = (
                normalize_arguments(
                    candidate_call.arguments
                )
                == normalize_arguments(
                    failed_call.arguments
                )
            )

            if same_tool and same_arguments:
                issues.append(
                    RepeatedFailedCallIssue(
                        tool_name=candidate_call.name,
                        arguments=candidate_call.arguments,
                    )
                )

                break

    return RepeatedFailedCallResult(
        has_repeated_failed_call=len(issues) > 0,
        issue_count=len(issues),
        issues=issues,
    )