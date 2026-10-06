import json
from pathlib import Path


BASE_PATH = "results/base_dev_500.jsonl"
SFT_PATH = "results/sft_dev_500.jsonl"
OUTPUT_PATH = "results/base_sft_manual_audit_20.jsonl"

TARGET_COUNTS = {
    "format_fail->correct": 6,
    "wrong->correct": 5,
    "correct->wrong": 5,
    "format_fail->wrong": 4,
}


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

    assert set(base.keys()) == set(sft.keys())

    selected = []
    counts = {key: 0 for key in TARGET_COUNTS}

    for sample_id in sorted(base):
        base_row = base[sample_id]
        sft_row = sft[sample_id]

        transition = f'{base_row["status"]}->{sft_row["status"]}'

        if transition not in TARGET_COUNTS:
            continue

        if counts[transition] >= TARGET_COUNTS[transition]:
            continue

        selected.append(
            {
                "id": sample_id,
                "transition": transition,
                "reference": base_row["reference"],
                "base_prediction": base_row["prediction"],
                "base_parsed_answer": base_row["parsed_answer"],
                "sft_prediction": sft_row["prediction"],
                "sft_parsed_answer": sft_row["parsed_answer"],
                "manual_label": "",
                "manual_note": "",
            }
        )

        counts[transition] += 1

    assert len(selected) == 20, counts

    Path("results").mkdir(exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("saved:", OUTPUT_PATH)
    print("total:", len(selected))

    for transition, count in counts.items():
        print(f"{transition:24s}: {count}")


if __name__ == "__main__":
    main()
