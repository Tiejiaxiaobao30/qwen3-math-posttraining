from scripts.evaluator import extract_answer, parse_reference


def correctness_reward(response, gold):
    """
    Strict correctness reward.

    1.0:
        response follows the strict Final answer format
        AND parsed answer equals gold.

    0.0:
        format cannot be parsed
        OR numerical answer is wrong.
    """
    predicted = extract_answer(response)

    if predicted is None:
        return 0.0

    gold_value = parse_reference(str(gold))

    return 1.0 if predicted == gold_value else 0.0


def format_reward(response):
    """
    Reward only strict final-answer format.

    1.0: evaluator can extract a valid final answer.
    0.0: format is invalid.
    """
    parsed = extract_answer(response)

    return 1.0 if parsed is not None else 0.0


if __name__ == "__main__":
    gold = "16"

    test_cases = [
        "Reasoning...\nFinal answer: 16",
        "Reasoning...\nFinal answer: 20",
        "The answer is 16.",
        "Final answer: 1,600",
        "Final answer: 16\nFinal answer: 16",
    ]

    for i, response in enumerate(test_cases, start=1):
        print("=" * 60)
        print("CASE:", i)
        print("response:", repr(response))
        print("parsed:", extract_answer(response))
        print("gold:", gold)
        print("correctness_reward:", correctness_reward(response, gold))
        print("format_reward:", format_reward(response))
