import re
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

@dataclass
class ArgumentRelationIssue:
    relation_type: str
    tool_names: list[str]
    details: str


@dataclass
class ArgumentRelationResult:
    has_argument_relation_error: bool
    issue_count: int
    issues: list[ArgumentRelationIssue]


def check_argument_relations(response: str) -> ArgumentRelationResult:
    """
    Проверяет логические отношения между аргументами tool calls.

    Сейчас проверяются:
    1. item_ids[i] и new_item_ids[i] не должны совпадать
       при замене/модификации товара.
    2. Нельзя одновременно запрашивать return и exchange
       одного и того же delivered order.
    """

    calls = extract_tool_calls_from_text(response)
    issues: list[ArgumentRelationIssue] = []

    # 1. Старый и новый item в одной позиции не должны совпадать.
    for call in calls:
        item_ids = call.arguments.get("item_ids")
        new_item_ids = call.arguments.get("new_item_ids")

        if not isinstance(item_ids, list):
            continue

        if not isinstance(new_item_ids, list):
            continue

        for index, (old_item_id, new_item_id) in enumerate(
            zip(item_ids, new_item_ids)
        ):
            if old_item_id == new_item_id:
                issues.append(
                    ArgumentRelationIssue(
                        relation_type="same_old_and_new_item",
                        tool_names=[call.name],
                        details=(
                            f"item_ids[{index}] == "
                            f"new_item_ids[{index}] == "
                            f"{old_item_id!r}"
                        ),
                    )
                )

    # 2. Return и exchange одного delivered order конфликтуют:
    # после первого действия статус заказа уже перестаёт быть delivered.
    return_orders: set[str] = set()
    exchange_orders: set[str] = set()

    for call in calls:
        order_id = call.arguments.get("order_id")

        if not isinstance(order_id, str):
            continue

        if call.name == "return_delivered_order_items":
            return_orders.add(order_id)

        elif call.name == "exchange_delivered_order_items":
            exchange_orders.add(order_id)

    for order_id in sorted(return_orders & exchange_orders):
        issues.append(
            ArgumentRelationIssue(
                relation_type="return_and_exchange_same_order",
                tool_names=[
                    "return_delivered_order_items",
                    "exchange_delivered_order_items",
                ],
                details=(
                    f"return and exchange requested for "
                    f"the same order {order_id!r}"
                ),
            )
        )

    return ArgumentRelationResult(
        has_argument_relation_error=bool(issues),
        issue_count=len(issues),
        issues=issues,
    )


@dataclass
class ZipGroundingIssue:
    tool_name: str
    zip_value: str
    details: str


@dataclass
class ZipGroundingResult:
    has_ungrounded_zip: bool
    issue_count: int
    issues: list[ZipGroundingIssue]



def check_zip_grounding(
    original_prompt: str,
    response: str,
) -> ZipGroundingResult:
    """Checks whether a ZIP value is supported by conversation history."""
    calls = extract_tool_calls_from_text(response)
    issues = []

    # Remove the instructions, policy and tool definitions.
    # Actual conversations begin with the first role marker.
    history = original_prompt

    if "[AVAILABLE TOOLS]" in history:
        history = history.split("[AVAILABLE TOOLS]", 1)[1]

    import re

    role_pattern = r"(?:\u27e6|\u3010|\[)(?:USER|ASSISTANT)(?:\u27e7|\u3011|\])"
    match = re.search(role_pattern, history, flags=re.IGNORECASE)

    if match:
        history = history[match.start():]
    else:
        history = ""

    # An email local part is not evidence of a postal code.
    history = re.sub(
        r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        " ",
        history,
    )

    for call in calls:
        if call.name != "find_user_id_by_name_zip":
            continue

        zip_value = call.arguments.get("zip")
        if not isinstance(zip_value, str) or not zip_value:
            continue

        pattern = r"(?<!\w)" + re.escape(zip_value) + r"(?!\w)"

        if not re.search(pattern, history):
            issues.append(
                ZipGroundingIssue(
                    tool_name=call.name,
                    zip_value=zip_value,
                    details="ZIP is not supported by conversation history",
                )
            )

    return ZipGroundingResult(
        has_ungrounded_zip=bool(issues),
        issue_count=len(issues),
        issues=issues,
    )


import json as _ff_json
import re as _ff_re
from datetime import date as _ff_date


@dataclass
class FutureFlightCostIssue:
    expression: str
    past_reservations: list[str]
    past_amounts: list[int]


@dataclass
class FutureFlightCostResult:
    has_future_flight_cost_error: bool
    issue_count: int
    issues: list[FutureFlightCostIssue]


def check_future_flight_cost(
    original_prompt: str,
    response: str,
) -> FutureFlightCostResult:
    """Detect past reservations used in upcoming-flight totals."""

    empty = FutureFlightCostResult(False, 0, [])

    date_match = _ff_re.search(
        r"The current time is (\d{4}-\d{2}-\d{2})",
        original_prompt,
    )
    if not date_match:
        return empty

    current_date = _ff_date.fromisoformat(date_match.group(1))


    # Extract user messages using literal role markers.
    # No regex escapes are needed here.
    user_marker = chr(0x27E6) + "USER" + chr(0x27E7)
    role_start = chr(0x27E6)

    user_messages = []

    for section in original_prompt.split(user_marker)[1:]:
        message = section.split(role_start, 1)[0].strip()
        user_messages.append(message)

    if not user_messages:
        return empty

    last_message = user_messages[-1].lower()

    if not any(
        word in last_message
        for word in (
            "upcoming",
            "future flights",
            "\u043f\u0440\u0435\u0434\u0441\u0442\u043e\u044f\u0449",
            "\u0431\u0443\u0434\u0443\u0449",
        )
    ):
        return empty

    if not any(
        word in last_message
        for word in (
            "cost",
            "price",
            "total",
            "\u0441\u0442\u043e\u0438\u043c\u043e\u0441\u0442",
            "\u0441\u0443\u043c\u043c",
            "\u0441\u043a\u043e\u043b\u044c\u043a\u043e",
        )
    ):
        return empty

    calls = [
        call
        for call in extract_tool_calls_from_text(response)
        if call.name == "calculate"
    ]
    if not calls:
        return empty

    reservations = {}

    pattern = (
        r"TOOL_RESPONSE get_reservation_details:\s*"
        r"(\{[^\n]*\})"
    )

    for match in _ff_re.finditer(pattern, original_prompt):
        try:
            data = _ff_json.loads(match.group(1))
        except _ff_json.JSONDecodeError:
            continue

        reservation_id = data.get("reservation_id")
        flights = data.get("flights")
        payments = data.get("payment_history")

        if not reservation_id or not flights or not payments:
            continue

        dates = []

        for flight in flights:
            try:
                dates.append(
                    _ff_date.fromisoformat(flight["date"])
                )
            except (KeyError, ValueError, TypeError):
                dates = []
                break

        if not dates or len(payments) != 1:
            continue

        amount = payments[0].get("amount")

        if type(amount) is not int or amount <= 0:
            continue

        reservations[reservation_id] = {
            "amount": amount,
            "past": all(day < current_date for day in dates),
        }

    amounts = [item["amount"] for item in reservations.values()]

    if len(amounts) != len(set(amounts)):
        return empty

    by_amount = {
        item["amount"]: (reservation_id, item)
        for reservation_id, item in reservations.items()
    }

    issues = []

    for call in calls:
        expression = call.arguments.get("expression")

        if not isinstance(expression, str):
            continue

        if not _ff_re.fullmatch(
            r"\s*\d+(?:\s*\+\s*\d+)*\s*",
            expression,
        ):
            continue

        used_amounts = [
            int(number)
            for number in _ff_re.findall(r"\d+", expression)
        ]

        if len(used_amounts) != len(set(used_amounts)):
            continue

        if any(amount not in by_amount for amount in used_amounts):
            continue

        past_ids = []
        past_amounts = []

        for amount in used_amounts:
            reservation_id, item = by_amount[amount]

            if item["past"]:
                past_ids.append(reservation_id)
                past_amounts.append(amount)

        if past_ids:
            issues.append(
                FutureFlightCostIssue(
                    expression=expression,
                    past_reservations=past_ids,
                    past_amounts=past_amounts,
                )
            )

    return FutureFlightCostResult(
        has_future_flight_cost_error=bool(issues),
        issue_count=len(issues),
        issues=issues,
    )
