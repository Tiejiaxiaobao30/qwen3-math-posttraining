import json
import random
from pathlib import Path

from scripts.reward import correctness_reward, format_reward


RL_RAW_PATH = "data/processed/rl.jsonl"
RL_MANIFEST_PATH = "data/processed/rl_manifest.jsonl"
PROMPT_PATH = "configs/sft_prompt.txt"
OUTPUT_PATH = "results/reward_audit.json"


REWARD_CASES = [
    {
        "name": "correct_integer",
        "response": "Reasoning...\nFinal answer: 16",
        "gold": "16",
        "expected_correctness": 1.0,
        "expected_format": 1.0,
    },
    {
        "name": "wrong_but_valid_format",
        "response": "Reasoning...\nFinal answer: 20",
        "gold": "16",
        "expected_correctness": 0.0,
        "expected_format": 1.0,
    },
    {
        "name": "correct_number_wrong_format",
        "response": "The answer is 16.",
        "gold": "16",
        "expected_correctness": 0.0,
        "expected_format": 0.0,
    },
    {
        "name": "duplicate_final_answer",
        "response": "Final answer: 16\nFinal answer: 16",
        "gold": "16",
        "expected_correctness": 0.0,
        "expected_format": 0.0,
    },
    {
        "name": "comma_number",
        "response": "Final answer: 1,600",
        "gold": "1600",
        "expected_correctness": 1.0,
        "expected_format": 1.0,
    },
    {
        "name": "negative_number",
        "response": "Final answer: -3",
        "gold": "-3",
        "expected_correctness": 1.0,
        "expected_format": 1.0,
    },
    {
        "name": "decimal_equivalence",
        "response": "Final answer: 16.0",
        "gold": "16",
        "expected_correctness": 1.0,
        "expected_format": 1.0,
    },
    {
        "name": "empty_response",
        "response": "",
        "gold": "16",
        "expected_correctness": 0.0,
        "expected_format": 0.0,
    },
    {
        "name": "copied_gold_but_final_wrong",
        "response": "The problem mentions 16.\nFinal answer: 80",
        "gold": "16",
        "expected_correctness": 0.0,
        "expected_format": 1.0,
    },
    {
        "name": "dollar_sign",
        "response": "Final answer: $16",
        "gold": "16",
        "expected_correctness": 1.0,
        "expected_format": 1.0,
    },
    {
        "name": "unit_after_number",
        "response": "Final answer: 16 apples",
        "gold": "16",
        "expected_correctness": 0.0,
        "expected_format": 0.0,
    },
    {
        "name": "two_different_answers",
        "response": "Final answer: 16\nFinal answer: 20",
        "gold": "16",
        "expected_correctness": 0.0,
        "expected_format": 0.0,
    },
    {
        "name": "trailing_text",
        "response": "Final answer: 16\nDone.",
        "gold": "16",
        "expected_correctness": 0.0,
        "expected_format": 0.0,
    },
    {
        "name": "zero_equivalence",
        "response": "Final answer: 0.0",
        "gold": "0",
        "expected_correctness": 1.0,
        "expected_format": 1.0,
    },
]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    # 1. Reward adversarial audit
    reward_results = []
    reward_passed = 0

    for case in REWARD_CASES:
        actual_c = correctness_reward(
            case["response"],
            case["gold"],
        )
        actual_f = format_reward(case["response"])

        passed = (
            actual_c == case["expected_correctness"]
            and actual_f == case["expected_format"]
        )

        reward_results.append(
            {
                "name": case["name"],
                "actual_correctness": actual_c,
                "actual_format": actual_f,
                "expected_correctness": case["expected_correctness"],
                "expected_format": case["expected_format"],
                "passed": passed,
            }
        )

        reward_passed += int(passed)

    # 2. RL manifest audit
    raw_rows = load_jsonl(RL_RAW_PATH)
    manifest_rows = load_jsonl(RL_MANIFEST_PATH)

    assert len(raw_rows) == 1000
    assert len(manifest_rows) == 1000

    raw_by_id = {row["id"]: row for row in raw_rows}
    manifest_by_id = {row["id"]: row for row in manifest_rows}

    assert set(raw_by_id) == set(manifest_by_id)

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    random.seed(42)
    audit_ids = random.sample(list(raw_by_id.keys()), 50)

    prompt_failures = []
    gold_mismatches = []
    incidental_gold_overlap = []

    for sample_id in audit_ids:
        raw = raw_by_id[sample_id]
        manifest = manifest_by_id[sample_id]

        expected_prompt = prompt_template.format(
            question=raw["question"]
        )

        if manifest["prompt"] != expected_prompt:
            prompt_failures.append(sample_id)

        if str(manifest["gold"]) != str(raw["final_answer"]):
            gold_mismatches.append(sample_id)

        if str(raw["final_answer"]) in manifest["prompt"]:
            incidental_gold_overlap.append(sample_id)

    audit = {
        "reward": {
            "total_cases": len(REWARD_CASES),
            "passed_cases": reward_passed,
            "all_passed": reward_passed == len(REWARD_CASES),
            "cases": reward_results,
        },
        "rl_manifest": {
            "total_rows": len(manifest_rows),
            "audited_rows": 50,
            "seed": 42,
            "prompt_failures": prompt_failures,
            "gold_mismatches": gold_mismatches,
            "incidental_gold_text_overlap_count": len(
                incidental_gold_overlap
            ),
            "incidental_gold_text_overlap_ids": incidental_gold_overlap,
            "audit_passed": (
                not prompt_failures
                and not gold_mismatches
            ),
        },
    }

    Path("results").mkdir(exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    print("Reward cases:")
    print(
        f"{reward_passed}/{len(REWARD_CASES)} passed"
    )

    print("\nRL manifest:")
    print("total rows:", len(manifest_rows))
    print("audited rows:", 50)
    print("prompt failures:", prompt_failures)
    print("gold mismatches:", gold_mismatches)
    print(
        "incidental gold overlap:",
        len(incidental_gold_overlap),
    )

    print("\nsaved:", OUTPUT_PATH)

    assert audit["reward"]["all_passed"]
    assert audit["rl_manifest"]["audit_passed"]

    print("\nDAY10 REWARD / RL AUDIT: PASS")


if __name__ == "__main__":
    main()
