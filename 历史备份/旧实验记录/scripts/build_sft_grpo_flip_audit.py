import json
from pathlib import Path


DEV_PATH = Path("data/processed/dev.jsonl")
SFT_PATH = Path("results/sft_dev_500.jsonl")
GRPO_PATH = Path("results/grpo_dev_500.jsonl")

OUTPUT_JSONL = Path(
    "results/sft_grpo_flip_24.jsonl"
)

OUTPUT_MD = Path(
    "reports/day13/sft_grpo_flip_24.md"
)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def main():

    # ==================================================
    # 1. 加载固定 dev500 + SFT + GRPO 结果
    # ==================================================

    dev_rows = load_jsonl(DEV_PATH)
    sft_rows = load_jsonl(SFT_PATH)
    grpo_rows = load_jsonl(GRPO_PATH)

    assert len(dev_rows) == 500
    assert len(sft_rows) == 500
    assert len(grpo_rows) == 500

    dev_by_id = {
        str(row["id"]): row
        for row in dev_rows
    }

    sft_by_id = {
        str(row["id"]): row
        for row in sft_rows
    }

    grpo_by_id = {
        str(row["id"]): row
        for row in grpo_rows
    }

    assert set(sft_by_id) == set(grpo_by_id)

    # ==================================================
    # 2. 找“正确性发生翻转”的案例
    #
    # improved:
    # wrong / format_fail -> correct
    #
    # regressed:
    # correct -> wrong / format_fail
    # ==================================================

    cases = []

    for sample_id in sft_by_id:

        sft = sft_by_id[sample_id]
        grpo = grpo_by_id[sample_id]

        sft_correct = (
            sft["status"] == "correct"
        )

        grpo_correct = (
            grpo["status"] == "correct"
        )

        # 正确性没发生变化就跳过
        if sft_correct == grpo_correct:
            continue

        if (
            not sft_correct
            and grpo_correct
        ):
            direction = "improved"

        else:
            direction = "regressed"

        dev = dev_by_id[
            sample_id
        ]

        case = {
            "id": sample_id,

            "direction": direction,

            "transition": (
                f"{sft['status']}"
                f" -> "
                f"{grpo['status']}"
            ),

            "question": dev["question"],

            "gold": str(
                dev["final_answer"]
            ),

            "sft_status": (
                sft["status"]
            ),

            "sft_parsed_answer": (
                sft["parsed_answer"]
            ),

            "sft_generated_tokens": (
                sft["generated_tokens"]
            ),

            "sft_truncated": (
                sft["truncated"]
            ),

            "sft_prediction": (
                sft["prediction"]
            ),

            "grpo_status": (
                grpo["status"]
            ),

            "grpo_parsed_answer": (
                grpo["parsed_answer"]
            ),

            "grpo_generated_tokens": (
                grpo["generated_tokens"]
            ),

            "grpo_truncated": (
                grpo["truncated"]
            ),

            "grpo_prediction": (
                grpo["prediction"]
            ),

            # 后面人工分析时填
            "manual_category": "",
            "manual_note": "",
        }

        cases.append(case)

    # improved 放前面，regressed 放后面
    cases.sort(
        key=lambda x: (
            0
            if x["direction"] == "improved"
            else 1,
            int(x["id"])
            if str(x["id"]).isdigit()
            else str(x["id"]),
        )
    )

    improved = [
        x
        for x in cases
        if x["direction"] == "improved"
    ]

    regressed = [
        x
        for x in cases
        if x["direction"] == "regressed"
    ]

    # ==================================================
    # 3. 硬验收
    # ==================================================

    print(
        "correctness-flip cases:",
        len(cases),
    )

    print(
        "improved:",
        len(improved),
    )

    print(
        "regressed:",
        len(regressed),
    )

    print(
        "net:",
        len(improved)
        - len(regressed),
    )

    assert len(cases) == 24
    assert len(improved) == 13
    assert len(regressed) == 11

    # ==================================================
    # 4. 保存 JSONL
    # ==================================================

    OUTPUT_JSONL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_JSONL,
        "w",
        encoding="utf-8",
    ) as f:

        for case in cases:
            f.write(
                json.dumps(
                    case,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ==================================================
    # 5. 生成人类可读 Markdown
    # ==================================================

    OUTPUT_MD.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_MD,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# Day13｜SFT → GRPO "
            "正确性翻转案例审计\n\n"
        )

        f.write(
            "固定 dev500："
            "SFT 390/500 (78.00%) → "
            "GRPO 392/500 (78.40%)。\n\n"
        )

        f.write(
            f"- Improved: {len(improved)}\n"
        )

        f.write(
            f"- Regressed: {len(regressed)}\n"
        )

        f.write(
            f"- Net gain: "
            f"{len(improved) - len(regressed)}\n\n"
        )

        for i, case in enumerate(
            cases,
            start=1,
        ):

            f.write(
                f"## Case {i}｜"
                f"{case['direction'].upper()}｜"
                f"ID {case['id']}\n\n"
            )

            f.write(
                f"**Transition:** "
                f"`{case['transition']}`\n\n"
            )

            f.write(
                f"**Gold:** "
                f"`{case['gold']}`\n\n"
            )

            f.write(
                "**Question**\n\n"
            )

            f.write(
                case["question"]
                + "\n\n"
            )

            f.write(
                "### SFT\n\n"
            )

            f.write(
                f"- status: "
                f"`{case['sft_status']}`\n"
            )

            f.write(
                f"- parsed: "
                f"`{case['sft_parsed_answer']}`\n"
            )

            f.write(
                f"- tokens: "
                f"{case['sft_generated_tokens']}\n\n"
            )

            f.write(
                "```text\n"
            )

            f.write(
                case["sft_prediction"]
                + "\n"
            )

            f.write(
                "```\n\n"
            )

            f.write(
                "### GRPO\n\n"
            )

            f.write(
                f"- status: "
                f"`{case['grpo_status']}`\n"
            )

            f.write(
                f"- parsed: "
                f"`{case['grpo_parsed_answer']}`\n"
            )

            f.write(
                f"- tokens: "
                f"{case['grpo_generated_tokens']}\n\n"
            )

            f.write(
                "```text\n"
            )

            f.write(
                case["grpo_prediction"]
                + "\n"
            )

            f.write(
                "```\n\n"
            )

            f.write(
                "### Manual diagnosis\n\n"
            )

            f.write(
                "- Category: TODO\n"
            )

            f.write(
                "- Note: TODO\n\n"
            )

            f.write(
                "---\n\n"
            )

    print(
        "\nsaved:",
        OUTPUT_JSONL,
    )

    print(
        "saved:",
        OUTPUT_MD,
    )


if __name__ == "__main__":
    main()
