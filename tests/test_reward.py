from scripts.reward import correctness_reward, format_reward


CASES = [
    # name, response, gold, expected_correctness, expected_format

    (
        "correct_integer",
        "Reasoning...\nFinal answer: 16",
        "16",
        1.0,
        1.0,
    ),

    (
        "wrong_but_valid_format",
        "Reasoning...\nFinal answer: 20",
        "16",
        0.0,
        1.0,
    ),

    (
        "correct_number_wrong_format",
        "The answer is 16.",
        "16",
        0.0,
        0.0,
    ),

    (
        "duplicate_final_answer",
        "Final answer: 16\nFinal answer: 16",
        "16",
        0.0,
        0.0,
    ),

    (
        "comma_number",
        "Final answer: 1,600",
        "1600",
        1.0,
        1.0,
    ),

    (
        "negative_number",
        "Final answer: -3",
        "-3",
        1.0,
        1.0,
    ),

    (
        "decimal_equivalence",
        "Final answer: 16.0",
        "16",
        1.0,
        1.0,
    ),

    (
        "empty_response",
        "",
        "16",
        0.0,
        0.0,
    ),

    (
        "gold_mentioned_but_final_wrong",
        "The reference number might be 16, but after calculation:\nFinal answer: 80",
        "16",
        0.0,
        1.0,
    ),
]


def main():
    passed = 0

    for name, response, gold, expected_c, expected_f in CASES:
        actual_c = correctness_reward(response, gold)
        actual_f = format_reward(response)

        ok = (
            actual_c == expected_c
            and actual_f == expected_f
        )

        print("=" * 70)
        print("case:", name)
        print("correctness:", actual_c, "expected:", expected_c)
        print("format:", actual_f, "expected:", expected_f)
        print("PASS" if ok else "FAIL")

        if ok:
            passed += 1

    print("\nSUMMARY")
    print(f"passed: {passed}/{len(CASES)}")

    assert passed == len(CASES), "Reward audit failed"


if __name__ == "__main__":
    main()
