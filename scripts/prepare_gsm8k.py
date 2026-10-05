import json
import re
from pathlib import Path

from datasets import load_dataset


MANIFEST_PATH = Path("split_manifest.json")
OUTPUT_DIR = Path("data/processed")
CACHE_DIR = "/mnt/workspace/datasets_cache"


def clean_answer(raw_answer):
    answer = re.sub(r"<<[^>]+>>", "", raw_answer)

    answer = re.sub(
        r"####\s*(.+)$",
        r"Final answer: \1",
        answer
    )

    return answer.strip()


def extract_final_answer(raw_answer):
    return raw_answer.split("####")[-1].strip()


def main():
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    ds = load_dataset(
        "modelscope/gsm8k",
        "main",
        split="train",
        cache_dir=CACHE_DIR
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- SFT ----------
    sft_ids = manifest["splits"]["sft"]["ids"]
    sft_path = OUTPUT_DIR / "sft.jsonl"

    with open(sft_path, "w") as f:
        for sample_id in sft_ids:
            sample = ds[sample_id]

            record = {
                "id": sample_id,
                "question": sample["question"].strip(),
                "answer": clean_answer(sample["answer"]),
                "final_answer": extract_final_answer(sample["answer"])
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---------- RL ----------
    rl_ids = manifest["splits"]["rl"]["ids"]
    rl_path = OUTPUT_DIR / "rl.jsonl"

    with open(rl_path, "w") as f:
        for sample_id in rl_ids:
            sample = ds[sample_id]

            record = {
                "id": sample_id,
                "question": sample["question"].strip(),
                "final_answer": extract_final_answer(sample["answer"])
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---------- DEV ----------
    dev_ids = manifest["splits"]["dev"]["ids"]
    dev_path = OUTPUT_DIR / "dev.jsonl"

    with open(dev_path, "w") as f:
        for sample_id in dev_ids:
            sample = ds[sample_id]

            record = {
                "id": sample_id,
                "question": sample["question"].strip(),
                "final_answer": extract_final_answer(sample["answer"])
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("saved:", sft_path, "samples:", len(sft_ids))
    print("saved:", rl_path, "samples:", len(rl_ids))
    print("saved:", dev_path, "samples:", len(dev_ids))


if __name__ == "__main__":
    main()
