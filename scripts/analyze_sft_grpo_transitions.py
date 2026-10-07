import json
from collections import Counter


SFT_PATH = "results/sft_dev_500.jsonl"
GRPO_PATH = "results/grpo_dev_500.jsonl"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def main():

    sft = load_jsonl(SFT_PATH)
    grpo = load_jsonl(GRPO_PATH)

    assert len(sft) == 500
    assert len(grpo) == 500

    sft_by_id = {
        str(row["id"]): row
        for row in sft
    }

    grpo_by_id = {
        str(row["id"]): row
        for row in grpo
    }

    assert set(sft_by_id) == set(grpo_by_id)

    transitions = Counter()

    improved = []
    regressed = []

    for sample_id in sft_by_id:

        s = sft_by_id[sample_id]
        g = grpo_by_id[sample_id]

        old = s["status"]
        new = g["status"]

        transitions[
            f"{old} -> {new}"
        ] += 1

        if (
            old != "correct"
            and new == "correct"
        ):
            improved.append(sample_id)

        if (
            old == "correct"
            and new != "correct"
        ):
            regressed.append(sample_id)

    print("=== SFT -> GRPO TRANSITIONS ===")

    order = [
        "correct -> correct",
        "correct -> wrong",
        "correct -> format_fail",

        "wrong -> correct",
        "wrong -> wrong",
        "wrong -> format_fail",

        "format_fail -> correct",
        "format_fail -> wrong",
        "format_fail -> format_fail",
    ]

    for key in order:
        print(
            f"{key:28s}",
            transitions[key],
        )

    print()
    print(
        "improved to correct:",
        len(improved),
    )

    print(
        "regressed from correct:",
        len(regressed),
    )

    print(
        "net correct gain:",
        len(improved) - len(regressed),
    )

    print()
    print(
        "improved ids:",
        improved,
    )

    print(
        "regressed ids:",
        regressed,
    )


if __name__ == "__main__":
    main()
