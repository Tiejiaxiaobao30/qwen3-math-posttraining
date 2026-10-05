import re
from decimal import Decimal


FINAL_ANSWER_PATTERN = re.compile(
    r"Final answer:\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*$",
    re.IGNORECASE
)


def extract_answer(text):
    if not text:
        return None

    if text.lower().count("final answer:") != 1:
        return None

    match = FINAL_ANSWER_PATTERN.search(text)

    if match is None:
        return None

    number_text = match.group(1).replace(",", "")

    return Decimal(number_text)


def parse_reference(text):
    number_text = text.replace(",", "").strip()
    return Decimal(number_text)


def score(prediction, reference):
    pred_value = extract_answer(prediction)

    if pred_value is None:
        return "format_fail"

    ref_value = parse_reference(reference)

    if pred_value == ref_value:
        return "correct"

    return "wrong"


def build_result(
    sample_id,
    prediction,
    reference,
    generated_tokens,
    truncated,
):
    parsed = extract_answer(prediction)
    status = score(prediction, reference)

    return {
        "id": sample_id,
        "reference": reference,
        "prediction": prediction,
        "parsed_answer": (
            str(parsed) if parsed is not None else None
        ),
        "status": status,
        "generated_tokens": generated_tokens,
        "truncated": truncated,
    }
