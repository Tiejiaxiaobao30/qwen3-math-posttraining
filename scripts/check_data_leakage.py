import json
import re
from itertools import combinations

from datasets import load_dataset


CACHE_DIR = "/mnt/workspace/datasets_cache"
MANIFEST_PATH = "split_manifest.json"


def normalize_question(text):
    return " ".join(text.lower().split())


def template_question(text):
    text = text.lower()
    text = re.sub(
        r"\d+(?:,\d{3})*(?:\.\d+)?",
        "<num>",
        text
    )
    return " ".join(text.split())


def main():
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    train_ds = load_dataset(
        "modelscope/gsm8k",
        "main",
        split="train",
        cache_dir=CACHE_DIR
    )

    test_ds = load_dataset(
        "modelscope/gsm8k",
        "main",
        split="test",
        cache_dir=CACHE_DIR
    )

    split_names = ["dev", "sft", "rl", "reserve"]

    # ---------- Exact normalized question ----------
    exact_sets = {}

    for name in split_names:
        ids = manifest["splits"][name]["ids"]

        exact_sets[name] = {
            normalize_question(train_ds[i]["question"])
            for i in ids
        }

    print("===== TRAIN SPLIT EXACT OVERLAP =====")

    for a, b in combinations(split_names, 2):
        count = len(exact_sets[a] & exact_sets[b])
        print(f"{a} ∩ {b} =", count)
        assert count == 0

    # ---------- Template near-duplicate ----------
    template_sets = {}

    for name in ["dev", "sft", "rl"]:
        ids = manifest["splits"][name]["ids"]

        template_sets[name] = {
            template_question(train_ds[i]["question"])
            for i in ids
        }

    print()
    print("===== TRAIN SPLIT TEMPLATE OVERLAP =====")

    for a, b in combinations(["dev", "sft", "rl"], 2):
        count = len(template_sets[a] & template_sets[b])
        print(f"{a} ∩ {b} =", count)
        assert count == 0

    # ---------- Train vs official test ----------
    train_exact = {
        normalize_question(x["question"])
        for x in train_ds
    }

    test_exact = {
        normalize_question(x["question"])
        for x in test_ds
    }

    exact_test_overlap = len(train_exact & test_exact)

    print()
    print("train-test exact overlap =", exact_test_overlap)

    assert exact_test_overlap == 0

    train_templates = {
        template_question(x["question"])
        for x in train_ds
    }

    test_templates = {
        template_question(x["question"])
        for x in test_ds
    }

    template_test_overlap = len(
        train_templates & test_templates
    )

    print(
        "train-test template overlap =",
        template_test_overlap
    )

    assert template_test_overlap == 0

    print()
    print("DATA LEAKAGE CHECK PASSED")


if __name__ == "__main__":
    main()
