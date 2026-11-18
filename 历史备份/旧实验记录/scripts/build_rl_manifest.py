import json
from pathlib import Path


RL_PATH = "data/processed/rl.jsonl"
PROMPT_PATH = "configs/sft_prompt.txt"
OUTPUT_PATH = "data/processed/rl_manifest.jsonl"


def main():
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    rows = []

    with open(RL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)

            prompt = prompt_template.format(
                question=sample["question"]
            )

            assert "{question}" not in prompt

            rows.append(
                {
                    "id": sample["id"],
                    "prompt": prompt,
                    "gold": sample["final_answer"],
                }
            )

    assert len(rows) == 1000

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("saved:", OUTPUT_PATH)
    print("rows:", len(rows))

    print("\n--- first prompt ---")
    print(rows[0]["prompt"])

    print("\n--- gold kept separately ---")
    print(rows[0]["gold"])


if __name__ == "__main__":
    main()
