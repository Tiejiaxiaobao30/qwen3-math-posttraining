import json
import re
from decimal import Decimal
from pathlib import Path


RL_PATH = Path("data/processed/rl.jsonl")
SFT_PATH = Path("data/processed/sft.jsonl")

OUTPUT_PATH = Path(
    "data/processed/extra_sft_rl1000.jsonl"
)


def load_jsonl(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def normalize_number(text):
    text = str(text)

    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.strip()

    return Decimal(text)


def detect_sft_style():
    """
    检查原来的 SFT 数据有没有保留
    GSM8K 的 <<...>> calculator annotations。

    Extra-SFT 尽量沿用原 SFT completion 风格。
    """

    rows = load_jsonl(
        SFT_PATH
    )[:100]

    preserve_annotations = any(
        "<<" in row.get("answer", "")
        for row in rows
    )

    final_marker_ok = all(
        "Final answer:" in row.get(
            "answer",
            "",
        )
        for row in rows
    )

    print(
        "preserve_calc_annotations =",
        preserve_annotations,
    )

    print(
        "existing_sft_has_final_marker =",
        final_marker_ok,
    )

    if not final_marker_ok:
        raise RuntimeError(
            "Existing SFT data style is unexpected: "
            "Final answer marker missing."
        )

    return preserve_annotations


def add_source_rows(
    mapping,
    rows,
):
    for row in rows:

        if not isinstance(
            row,
            dict,
        ):
            continue

        question = row.get(
            "question"
        )

        answer = row.get(
            "answer"
        )

        if (
            question is None
            or answer is None
        ):
            continue

        mapping[
            question.strip()
        ] = str(answer)


def scan_local_data(
    target_questions,
):
    """
    优先寻找项目本地已有的原始 GSM8K train 数据。
    """

    mapping = {}

    print(
        "\n=== SEARCH LOCAL DATA ==="
    )

    for path in Path("data").rglob(
        "*.jsonl"
    ):

        # 不把即将生成的文件重新读进来
        if path == OUTPUT_PATH:
            continue

        try:
            rows = load_jsonl(
                path
            )
        except Exception:
            continue

        before = len(
            mapping
        )

        add_source_rows(
            mapping,
            rows,
        )

        matched = sum(
            1
            for q in target_questions
            if q in mapping
        )

        if len(mapping) > before:
            print(
                path,
                "matched target questions =",
                matched,
            )

    matched_mapping = {
        q: mapping[q]
        for q in target_questions
        if q in mapping
    }

    return matched_mapping


def load_hf_train():
    print(
        "\n=== TRY HUGGING FACE GSM8K TRAIN ==="
    )

    from datasets import load_dataset

    ds = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
    )

    return [
        {
            "question": row["question"],
            "answer": row["answer"],
        }
        for row in ds
    ]


def load_modelscope_train():
    print(
        "\n=== TRY MODELSCOPE GSM8K TRAIN ==="
    )

    from modelscope.msdatasets import (
        MsDataset,
    )

    errors = []

    candidates = [
        "modelscope/gsm8k",
        "openai/gsm8k",
        "gsm8k",
    ]

    for dataset_name in candidates:

        try:
            print(
                "trying:",
                dataset_name,
            )

            ds = MsDataset.load(
                dataset_name,
                subset_name="main",
                split="train",
            )

            rows = []

            for row in ds:
                rows.append(
                    {
                        "question":
                            row["question"],
                        "answer":
                            row["answer"],
                    }
                )

            if rows:
                print(
                    "ModelScope source:",
                    dataset_name,
                )

                return rows

        except Exception as exc:
            errors.append(
                (
                    dataset_name,
                    repr(exc),
                )
            )

    print(
        "\nModelScope attempts failed:"
    )

    for name, error in errors:
        print(
            name,
            error,
        )

    raise RuntimeError(
        "Could not load GSM8K train "
        "from ModelScope."
    )


def get_reference_mapping(
    target_questions,
):
    """
    加载顺序：

    1. 项目本地 data/
    2. Hugging Face train
    3. ModelScope train

    只使用 train split。
    """

    mapping = scan_local_data(
        target_questions
    )

    if len(mapping) == len(
        target_questions
    ):
        print(
            "\nAll RL questions found locally."
        )
        return mapping

    print(
        "\nLocal match:",
        len(mapping),
        "/",
        len(target_questions),
    )

    source_rows = None

    try:
        source_rows = load_hf_train()

    except Exception as exc:
        print(
            "Hugging Face load failed:",
            repr(exc),
        )

    if source_rows is None:
        source_rows = (
            load_modelscope_train()
        )

    add_source_rows(
        mapping,
        source_rows,
    )

    matched_mapping = {
        q: mapping[q]
        for q in target_questions
        if q in mapping
    }

    return matched_mapping


def convert_answer(
    raw_answer,
    expected_final,
    preserve_annotations,
):
    """
    GSM8K 原始格式：

    reasoning...
    #### 42

    转成本项目协议：

    reasoning...
    Final answer: 42
    """

    if "####" not in raw_answer:
        raise ValueError(
            "Missing GSM8K #### marker."
        )

    reasoning, raw_final = (
        raw_answer.rsplit(
            "####",
            1,
        )
    )

    reasoning = (
        reasoning.strip()
    )

    raw_final = (
        raw_final.strip()
    )

    if (
        normalize_number(
            raw_final
        )
        !=
        normalize_number(
            expected_final
        )
    ):
        raise ValueError(
            "Reference final answer mismatch: "
            f"raw={raw_final}, "
            f"expected={expected_final}"
        )

    if not preserve_annotations:

        reasoning = re.sub(
            r"<<.*?>>",
            "",
            reasoning,
        )

        reasoning = re.sub(
            r"[ \t]+\n",
            "\n",
            reasoning,
        )

        reasoning = re.sub(
            r" {2,}",
            " ",
            reasoning,
        )

        reasoning = (
            reasoning.strip()
        )

    completion = (
        reasoning
        + "\n"
        + "Final answer: "
        + str(expected_final)
    )

    return completion


def main():

    rl_rows = load_jsonl(
        RL_PATH
    )

    assert len(rl_rows) == 1000

    print(
        "RL rows =",
        len(rl_rows),
    )

    ids = [
        str(row["id"])
        for row in rl_rows
    ]

    assert len(ids) == len(
        set(ids)
    )

    target_questions = {
        row["question"].strip()
        for row in rl_rows
    }

    assert len(
        target_questions
    ) == 1000

    preserve_annotations = (
        detect_sft_style()
    )

    reference_mapping = (
        get_reference_mapping(
            target_questions
        )
    )

    print(
        "\nreference matches =",
        len(reference_mapping),
        "/ 1000",
    )

    if len(
        reference_mapping
    ) != 1000:

        missing = [
            q
            for q in target_questions
            if q not in reference_mapping
        ]

        print(
            "\nFirst missing questions:"
        )

        for q in missing[:5]:
            print(
                repr(q)
            )

        raise RuntimeError(
            "Not all RL questions were matched "
            "to GSM8K train references."
        )

    output_rows = []

    for row in rl_rows:

        question = (
            row["question"].strip()
        )

        expected_final = str(
            row["final_answer"]
        )

        raw_answer = (
            reference_mapping[
                question
            ]
        )

        completion = convert_answer(
            raw_answer=raw_answer,
            expected_final=expected_final,
            preserve_annotations=(
                preserve_annotations
            ),
        )

        output_rows.append(
            {
                "id": row["id"],
                "question": row[
                    "question"
                ],
                "answer": completion,
                "final_answer":
                    expected_final,
            }
        )

    assert len(
        output_rows
    ) == 1000

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        for row in output_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        "\n=============================="
    )

    print(
        "EXTRA-SFT DATA BUILD PASS"
    )

    print(
        "=============================="
    )

    print(
        "saved:",
        OUTPUT_PATH,
    )

    print(
        "rows:",
        len(output_rows),
    )

    print(
        "\n--- FIRST SAMPLE ---"
    )

    print(
        "id:",
        output_rows[0]["id"],
    )

    print(
        "question:"
    )

    print(
        output_rows[0]["question"]
    )

    print(
        "\ncompletion:"
    )

    print(
        output_rows[0]["answer"]
    )


if __name__ == "__main__":
    main()
