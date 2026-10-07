import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass
class Example:
    """Один пример из датасета."""
    id: str
    prompt: str
    response: str
    label: int | None = None
    explanation: str | None = None

@dataclass
class Message:
    """Одно сообщение из истории диалога."""
    role: str
    content: str
    turn: int | None = None

@dataclass
class ToolArgument:
    """Описание одного аргумента инструмента."""
    name: str
    type: str
    required: bool
    description: str = ""


@dataclass
class ToolDefinition:
    """Описание доступного инструмента."""
    name: str
    description: str
    arguments: dict[str, ToolArgument] = field(default_factory=dict)


@dataclass
class ToolCall:
    """Фактический вызов инструмента в истории."""
    name: str
    arguments: dict[str, Any]
    turn: int | None = None


@dataclass
class ToolResponse:
    """Результат вызова инструмента."""
    name: str
    content: str
    is_error: bool = False
    turn: int | None = None

@dataclass
class ParsedPrompt:
    """Структурированное представление prompt."""
    instructions: str = ""
    policy: str = ""
    available_tools: str = ""

    tools: dict[str, ToolDefinition] = field(default_factory=dict)

    messages: list[Message] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_responses: list[ToolResponse] = field(default_factory=list)

def load_jsonl(path: str) -> list[Example]:
    """Загружает датасет из JSONL-файла."""
    examples = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            row: dict[str, Any] = json.loads(line)

            examples.append(
                Example(
                    id=row["id"],
                    prompt=row["prompt"],
                    response=row["response"],
                    label=row.get("label"),
                    explanation=row.get("explanation"),
                )
            )

    return examples

def extract_tag(text: str, tag: str) -> str:
    """
    Извлекает содержимое XML-подобного тега.
    Например:
        <instructions>
        hello
        </instructions>

    -> "hello"
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"

    match = re.search(
        pattern,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match is None:
        return ""

    return match.group(1).strip()

def extract_policy(prompt: str) -> str:
    """
    Извлекает блок policy.
    В instructions встречается текст '<policy>',
    поэтому ищем пару <policy> ... </policy>,
    а не просто первое вхождение строки.
    """
    matches = list(
        re.finditer(
            r"<policy>(.*?)</policy>",
            prompt,
            flags=re.DOTALL | re.IGNORECASE,
        )
    )

    if not matches:
        return ""

    # Настоящий policy — самый большой найденный блок.
    largest_match = max(
        matches,
        key=lambda match: len(match.group(1)),
    )

    return largest_match.group(1).strip()

def extract_available_tools(prompt: str) -> str:
    """
    Пока извлекает секцию AVAILABLE TOOLS как сырой текст.
    Структуру отдельных инструментов разберём следующим этапом.
    """
    marker = "AVAILABLE TOOLS"

    position = prompt.find(marker)

    if position == -1:
        return ""

    text_after_marker = prompt[position + len(marker):]

    # История диалога начинается с первого маркера роли.

    role_match = re.search(
        r"⟦(?:USER|ASSISTANT)(?:\s*·[^⟧]*)?⟧",
        text_after_marker,
    )

    if role_match is not None:
        text_after_marker = text_after_marker[:role_match.start()]

    return text_after_marker.strip()

def parse_tool_definitions(text: str) -> dict[str, ToolDefinition]:
    """
    Разбирает секцию AVAILABLE TOOLS.

    Формат:
    - tool_name — Description.
        argument: string! — Description.
        optional_argument: integer — Description.

    Символ ! означает обязательный аргумент.
    """
    tools: dict[str, ToolDefinition] = {}

    tool_pattern = re.compile(
        r"^- ([A-Za-z_][A-Za-z0-9_]*)\s+—\s+(.*)$"
    )

    argument_pattern = re.compile(
        r"^\s+([A-Za-z_][A-Za-z0-9_]*)"
        r":\s*([A-Za-z_][A-Za-z0-9_]*)(!)?"
        r"(?:\s+—\s+(.*))?$"
    )

    current_tool: ToolDefinition | None = None

    for line in text.splitlines():
        tool_match = tool_pattern.match(line)

        if tool_match:
            name = tool_match.group(1)
            description = tool_match.group(2).strip()

            current_tool = ToolDefinition(
                name=name,
                description=description,
            )

            tools[name] = current_tool
            continue

        argument_match = argument_pattern.match(line)

        if argument_match and current_tool is not None:
            name = argument_match.group(1)
            argument_type = argument_match.group(2)
            required = argument_match.group(3) == "!"
            description = argument_match.group(4) or ""

            current_tool.arguments[name] = ToolArgument(
                name=name,
                type=argument_type,
                required=required,
                description=description.strip(),
            )

    return tools

def extract_tool_calls(messages: list[Message]) -> list[ToolCall]:
    """Извлекает все TOOL_CALL из истории."""

    calls = []

    pattern = re.compile(
        r"→\s*TOOL_CALL\s+([A-Za-z_][A-Za-z0-9_]*)"
        r":\s*(\{[^\n]*\})"
    )

    for message in messages:
        for match in pattern.finditer(message.content):
            name = match.group(1)
            raw_arguments = match.group(2)

            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {}

            calls.append(
                ToolCall(
                    name=name,
                    arguments=arguments,
                    turn=message.turn,
                )
            )

    return calls

def extract_tool_responses(
    messages: list[Message],
) -> list[ToolResponse]:
    """Извлекает TOOL_RESPONSE из истории."""

    responses = []

    pattern = re.compile(
        r"←\s*TOOL_RESPONSE\s+"
        r"([A-Za-z_][A-Za-z0-9_]*)"
        r"(\s+\[ERROR\])?"
        r":\s*(.*)"
    )

    for message in messages:
        for line in message.content.splitlines():
            match = pattern.search(line)

            if not match:
                continue

            name = match.group(1)
            is_error = match.group(2) is not None
            content = match.group(3).strip()

            responses.append(
                ToolResponse(
                    name=name,
                    content=content,
                    is_error=is_error,
                    turn=message.turn,
                )
            )

    return responses

def extract_messages(prompt: str) -> list[Message]:
    """Извлекает USER/ASSISTANT сообщения из промта."""

    pattern = re.compile(
        r"⟦(USER|ASSISTANT)(?:\s*·\s*ход\s*(\d+))?⟧"
    )

    matches = list(pattern.finditer(prompt))

    messages = []

    for index, match in enumerate(matches):
        role = match.group(1).lower()

        turn_text = match.group(2)
        turn = int(turn_text) if turn_text is not None else None

        content_start = match.end()

        if index + 1 < len(matches):
            content_end = matches[index + 1].start()
        else:
            content_end = len(prompt)

        content = prompt[content_start:content_end].strip()

        messages.append(
            Message(
                role=role,
                content=content,
                turn=turn,
            )
        )

    return messages

def parse_prompt(prompt: str) -> ParsedPrompt:
    """Главная функция парсинга prompt."""

    available_tools = extract_available_tools(prompt)
    messages = extract_messages(prompt)

    return ParsedPrompt(
        instructions=extract_tag(prompt, "instructions"),
        policy=extract_policy(prompt),
        available_tools=available_tools,
        tools=parse_tool_definitions(available_tools),
        messages=messages,
        tool_calls=extract_tool_calls(messages),
        tool_responses=extract_tool_responses(messages),
    )

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "valid.jsonl"

    examples = load_jsonl(str(data_path))

    print(f"Загружено примеров: {len(examples)}")

    # Проверяем по одному примеру каждого домена.

    domains = {}

    for example in examples:
        domain = example.id.split("__")[0]

        if domain not in domains:
            domains[domain] = example

    for domain, example in domains.items():
        parsed = parse_prompt(example.prompt)

        print("\n" + "=" * 80)
        print(f"DOMAIN: {domain}")
        print(f"ID: {example.id}")
        print("=" * 80)

        print("Instructions length:", len(parsed.instructions))
        print("Policy length:", len(parsed.policy))
        print("Available tools length:", len(parsed.available_tools))
        print("Messages:", len(parsed.messages))

        user_messages = [
            message
            for message in parsed.messages
            if message.role == "user"
        ]

        assistant_messages = [
            message
            for message in parsed.messages
            if message.role == "assistant"
        ]

        print("User messages:", len(user_messages))
        print("Assistant messages:", len(assistant_messages))
        print("Parsed tools:", len(parsed.tools))
        print("Tool calls:", len(parsed.tool_calls))
        print("Tool responses:", len(parsed.tool_responses))

        print("\nПервые 10 распознанных tools:")

        for tool_name in list(parsed.tools.keys())[:10]:
            tool = parsed.tools[tool_name]

            print(f"  {tool.name}")

            for argument in tool.arguments.values():
                required = "required" if argument.required else "optional"

                print(
                    f"      {argument.name}: "
                    f"{argument.type} ({required})"
                )

        print("\nПоследние 5 TOOL_CALL:")

        for call in parsed.tool_calls[-5:]:
            print(
                f"  turn={call.turn} "
                f"{call.name} "
                f"{call.arguments}"
            )

        print("\nПоследние 5 TOOL_RESPONSE:")

        for response in parsed.tool_responses[-5:]:
            print(
                f"  turn={response.turn} "
                f"{response.name} "
                f"error={response.is_error}"
            )