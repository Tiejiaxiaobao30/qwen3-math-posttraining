import json
import random
from decimal import Decimal
from pathlib import Path

from scripts.evaluator import parse_reference
from scripts.reward import correctness_reward, format_reward


INPUT_PATH = "data/processed/rl_manifest.jsonl"
OUTPUT_PATH = "results/reward_real50_audit.json"


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    assert len(rows) == 1000

    random.seed(42)
    selected = random.sample(rows, 50)

    results = []
    passed = 0

    for row in selected:
        gold = str(row["gold"])
        gold_value = parse_reference(gold)

        correct_response = (
            "Audit reasoning.\n"
            f"Final answer: {gold}"
        )

        wrong_value = gold_value + Decimal("1")

        wrong_response = (
            "Audit reasoning.\n"
            f"Final answer: {wrong_value}"
        )

        correct_reward = correctness_reward(
            correct_response,
            gold,
        )
        correct_format = format_reward(correct_response)

        wrong_reward = correctness_reward(
            wrong_response,
            gold,
        )
        wrong_format = format_reward(wrong_response)

        ok = (
            correct_reward == 1.0
            and correct_format == 1.0
            and wrong_reward == 0.0
            and wrong_format == 1.0
        )

        passed += int(ok)

        results.append(
            {
                "id": row["id"],
                "gold": gold,
                "correct_reward": correct_reward,
                "correct_format": correct_format,
                "wrong_test_answer": str(wrong_value),
                "wrong_reward": wrong_reward,
                "wrong_format": wrong_format,
                "passed": ok,
            }
        )

    audit = {
        "seed": 42,
        "sample_size": 50,
        "passed": passed,
        "all_passed": passed == 50,
        "cases": results,
    }

    Path("results").mkdir(exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            audit,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"real RL reward audit: {passed}/50")
    print("saved:", OUTPUT_PATH)

    assert passed == 50

    print("REAL-50 REWARD AUDIT: PASS")


if __name__ == "__main__":
    main()
