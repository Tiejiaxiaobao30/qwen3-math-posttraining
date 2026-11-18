import re

from datasets import load_dataset


CACHE_DIR = "/mnt/workspace/datasets_cache"

FINAL_ANSWER_PATTERN = re.compile(
    r"^-?[\d,]+(?:\.\d+)?$"
)


def main():
    ds = load_dataset(
        "modelscope/gsm8k",
        "main",
        split="train",
        cache_dir=CACHE_DIR
    )

    empty_question = []
    empty_answer = []
    missing_final_marker = []
    bad_final_answer = []

    for i, sample in enumerate(ds):
        question = sample["question"]
        answer = sample["answer"]

        if not question.strip():
            empty_question.append(i)

        if not answer.strip():
            empty_answer.append(i)

        if "####" not in answer:
            missing_final_marker.append(i)
            continue

        final_answer = answer.split("####")[-1].strip()

        if not FINAL_ANSWER_PATTERN.match(final_answer):
            bad_final_answer.append((i, final_answer))

    print("total =", len(ds))
    print("empty_question =", len(empty_question))
    print("empty_answer =", len(empty_answer))
    print("missing_final_marker =", len(missing_final_marker))
    print("bad_final_answer =", len(bad_final_answer))

    assert len(ds) == 7473
    assert len(empty_question) == 0
    assert len(empty_answer) == 0
    assert len(missing_final_marker) == 0
    assert len(bad_final_answer) == 0

    print()
    print("RAW GSM8K VALIDATION PASSED")


if __name__ == "__main__":
    main()
