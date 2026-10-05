import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    counts = {
        "correct": 0,
        "wrong": 0,
        "format_fail": 0,
        "truncated": 0,
    }

    total = 0

    with open(input_path, "r") as f:
        for line in f:
            row = json.loads(line)
            total += 1

            status = row["status"]

            if status not in [
                "correct",
                "wrong",
                "format_fail",
            ]:
                raise ValueError(
                    f"Unknown status: {status}"
                )

            counts[status] += 1

            if row["truncated"]:
                counts["truncated"] += 1

    assert (
        counts["correct"]
        + counts["wrong"]
        + counts["format_fail"]
        == total
    )

    metrics = {
        "total": total,
        "correct": counts["correct"],
        "wrong": counts["wrong"],
        "format_fail": counts["format_fail"],
        "truncated": counts["truncated"],
        "strict_accuracy": counts["correct"] / total,
        "format_success_rate": (
            counts["correct"] + counts["wrong"]
        ) / total,
        "truncation_rate": counts["truncated"] / total,
    }

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
