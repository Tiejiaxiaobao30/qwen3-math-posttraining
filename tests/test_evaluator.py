from decimal import Decimal

from scripts.evaluator import extract_answer, score, build_result


def test_extract_integer():
    assert extract_answer(
        "Reasoning...\nFinal answer: 72"
    ) == Decimal("72")


def test_extract_negative_decimal():
    assert extract_answer(
        "Reasoning...\nFinal answer: -3.5"
    ) == Decimal("-3.5")


def test_extract_comma_number():
    assert extract_answer(
        "Reasoning...\nFinal answer: 1,234"
    ) == Decimal("1234")


def test_extract_dollar_number():
    assert extract_answer(
        "Reasoning...\nFinal answer: $72"
    ) == Decimal("72")


def test_missing_protocol():
    assert extract_answer(
        "I think the answer is 72"
    ) is None


def test_empty_final_answer():
    assert extract_answer(
        "Reasoning...\nFinal answer:"
    ) is None


def test_text_after_number():
    assert extract_answer(
        "Reasoning...\nFinal answer: 72 apples"
    ) is None


def test_multiple_final_answers():
    text = (
        "Final answer: 10\n"
        "I changed my mind.\n"
        "Final answer: 20"
    )

    assert extract_answer(text) is None


def test_score_correct():
    assert score(
        "Reasoning...\nFinal answer: 72",
        "72"
    ) == "correct"


def test_score_wrong():
    assert score(
        "Reasoning...\nFinal answer: 73",
        "72"
    ) == "wrong"


def test_score_format_fail():
    assert score(
        "I think the answer is 72",
        "72"
    ) == "format_fail"


def test_empty_text():
    assert extract_answer("") is None


def test_trailing_spaces():
    assert extract_answer(
        "Reasoning...\nFinal answer: 72   "
    ) == Decimal("72")


def test_reference_with_comma():
    assert score(
        "Reasoning...\nFinal answer: 1234",
        "1,234"
    ) == "correct"


def test_build_result_correct():
    result = build_result(
        sample_id=100,
        prediction="Reasoning...\nFinal answer: 72",
        reference="72",
        generated_tokens=120,
        truncated=False,
    )

    assert result["id"] == 100
    assert result["parsed_answer"] == "72"
    assert result["status"] == "correct"
    assert result["generated_tokens"] == 120
    assert result["truncated"] is False


def test_build_result_wrong():
    result = build_result(
        sample_id=101,
        prediction="Reasoning...\nFinal answer: 73",
        reference="72",
        generated_tokens=80,
        truncated=False,
    )

    assert result["parsed_answer"] == "73"
    assert result["status"] == "wrong"


def test_build_result_format_fail():
    result = build_result(
        sample_id=102,
        prediction="I think the answer is 72",
        reference="72",
        generated_tokens=384,
        truncated=True,
    )

    assert result["parsed_answer"] is None
    assert result["status"] == "format_fail"
    assert result["truncated"] is True
