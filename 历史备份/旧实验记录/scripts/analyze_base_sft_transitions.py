import json
from collections import Counter, defaultdict
from pathlib import Path


BASE_PATH = "results/base_dev_500.jsonl"
SFT_PATH = "results/sft_dev_500.jsonl"
OUTPUT_PATH = "results/base_sft_transition_ids.json"


def load_jsonl(path):
    rows = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[row["id"]] = row

    return rows


def main():
    base = load_jsonl(BASE_PATH)
    sft = load_jsonl(SFT_PATH)

    print("base samples:", len(base))
    print("sft samples:", len(sft))

    assert set(base.keys()) == set(sft.keys())
    print("IDs aligned: YES")

    transitions = Counter()
    transition_ids = defaultdict(list)

    for sample_id in base:
        base_status = base[sample_id]["status"]
        sft_status = sft[sample_id]["status"]

        key = (base_status, sft_status)

        transitions[key] += 1
        transition_ids[f"{base_status}->{sft_status}"].append(sample_id)

    print("\nStatus transitions:")

    for (base_status, sft_status), count in sorted(transitions.items()):
        print(
            f"{base_status:12s} -> "
            f"{sft_status:12s}: {count}"
        )

    Path("results").mkdir(exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            transition_ids,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nsaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
