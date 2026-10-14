
from src.rules.tools import check_future_flight_cost


def make_prompt():
    return (
        "The current time is 2024-05-15 15:00:00 EST.\n"
        "\u27e6USER\u27e7\n"
        "What is the total cost of my upcoming flights?\n"
        "\u27e6ASSISTANT\u27e7\n"
        'TOOL_RESPONSE get_reservation_details: '
        '{"reservation_id":"FUTURE1",'
        '"flights":[{"date":"2024-05-28"}],'
        '"payment_history":[{"amount":402}]}\n'
        'TOOL_RESPONSE get_reservation_details: '
        '{"reservation_id":"PAST1",'
        '"flights":[{"date":"2024-05-14"}],'
        '"payment_history":[{"amount":308}]}\n'
    )


def test_past_flight_detected():
    response = (
        '\u2192 TOOL_CALL calculate: '
        '{"expression":"402 + 308"}'
    )
    result = check_future_flight_cost(make_prompt(), response)

    assert result.has_future_flight_cost_error
    assert result.issues[0].past_reservations == ["PAST1"]


def test_future_only_allowed():
    response = (
        '\u2192 TOOL_CALL calculate: '
        '{"expression":"402"}'
    )
    result = check_future_flight_cost(make_prompt(), response)

    assert not result.has_future_flight_cost_error


def test_other_question_not_flagged():
    prompt = make_prompt().replace(
        "upcoming flights", "all reservations"
    )
    response = (
        '\u2192 TOOL_CALL calculate: '
        '{"expression":"402 + 308"}'
    )
    result = check_future_flight_cost(prompt, response)

    assert not result.has_future_flight_cost_error


def test_unknown_amount_not_flagged():
    response = (
        '\u2192 TOOL_CALL calculate: '
        '{"expression":"402 + 999"}'
    )
    result = check_future_flight_cost(make_prompt(), response)

    assert not result.has_future_flight_cost_error


def test_russian_upcoming_flight_cost():
    prompt = make_prompt().replace(
        "What is the total cost of my upcoming flights?",
        "\u0421\u043a\u043e\u043b\u044c\u043a\u043e "
        "\u0441\u0442\u043e\u044f\u0442 "
        "\u043c\u043e\u0438 "
        "\u043f\u0440\u0435\u0434\u0441\u0442\u043e\u044f\u0449\u0438\u0435 "
        "\u0440\u0435\u0439\u0441\u044b?",
    )
    response = (
        '\u2192 TOOL_CALL calculate: '
        '{"expression":"402 + 308"}'
    )

    result = check_future_flight_cost(prompt, response)
    assert result.has_future_flight_cost_error
