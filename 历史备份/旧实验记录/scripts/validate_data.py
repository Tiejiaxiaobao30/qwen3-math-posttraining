import json
from pathlib import Path


MANIFEST_PATH = Path("split_manifest.json")
PROCESSED_DIR = Path("data/processed")


def load_jsonl(path):
    data = []

    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    return data


def main():
    # ---------- Load ----------
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    sft = load_jsonl(PROCESSED_DIR / "sft.jsonl")
    rl = load_jsonl(PROCESSED_DIR / "rl.jsonl")
    dev = load_jsonl(PROCESSED_DIR / "dev.jsonl")

    # ---------- 1. Count ----------
    assert len(sft) == 2000
    assert len(rl) == 1000
    assert len(dev) == 500

    print("count check: PASS")

    # ---------- 2. Manifest ID match ----------
    assert [x["id"] for x in sft] == manifest["splits"]["sft"]["ids"]
    assert [x["id"] for x in rl] == manifest["splits"]["rl"]["ids"]
    assert [x["id"] for x in dev] == manifest["splits"]["dev"]["ids"]

    print("manifest id check: PASS")

    # ---------- 3. Cross-split ID overlap ----------
    sft_ids = {x["id"] for x in sft}
    rl_ids = {x["id"] for x in rl}
    dev_ids = {x["id"] for x in dev}

    assert len(sft_ids & rl_ids) == 0
    assert len(sft_ids & dev_ids) == 0
    assert len(rl_ids & dev_ids) == 0

    print("cross-split id check: PASS")

    # ---------- 4. SFT answer format ----------
    for x in sft:
        assert x["question"].strip()
        assert x["answer"].strip()
        assert x["final_answer"].strip()

        assert "<<" not in x["answer"]
        assert "####" not in x["answer"]

        expected_last_line = f'Final answer: {x["final_answer"]}'
        actual_last_line = x["answer"].splitlines()[-1]

        assert actual_last_line == expected_last_line

    print("SFT format check: PASS")

    # ---------- 5. RL / DEV ----------
    for dataset in [rl, dev]:
        for x in dataset:
            assert x["question"].strip()
            assert x["final_answer"].strip()

    print("RL/DEV non-empty check: PASS")

    print()
    print("ALL DATA VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()
