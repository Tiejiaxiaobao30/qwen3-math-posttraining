"""Audit frozen evidence and paired outcomes without loading models (stdlib only)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.evaluator import build_result


ROOT = Path(__file__).resolve().parents[1]
RUNS = ("base", "sft_seed42", "grpo_seed42", "extra_sft", "sft_seed123", "grpo_seed123")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    manifest = read_json(ROOT / "results/frozen_eval_manifest.json")
    for path, key in [("configs/sft_prompt.txt", "prompt_sha256"), ("scripts/evaluator.py", "parser_sha256")]:
        assert sha(ROOT / path) == manifest["evaluation"][key], path
    dev = ROOT / "data/processed/dev.jsonl"
    if dev.exists():
        assert sha(dev) == manifest["evaluation"]["dev500_sha256"]
    provenance = read_json(ROOT / "reports/final/artifact_provenance.json")
    for item in provenance["imported_files"]:
        assert sha(ROOT / item["path"]) == item["sha256"], item["path"]
    test_path = ROOT / "data/official/gsm8k_test_1319.jsonl"
    test = read_jsonl(test_path)
    ids = [r["id"] for r in test]
    assert len(test) == len(set(ids)) == 1319
    refs = {r["id"]: r["final_answer"] for r in test}
    predictions, counts_by_run = {}, {}
    for run in RUNS:
        path = ROOT / f"results/official_test/{run}_test1319.jsonl"
        rows = read_jsonl(path)
        assert [r["id"] for r in rows] == ids, run
        for row in rows:
            assert row["reference"] == refs[row["id"]], (run, row["id"])
            rescored = build_result(row["id"], row["prediction"], row["reference"], row["generated_tokens"], row["truncated"])
            assert row == rescored, (run, row["id"])
        counts = Counter(r["status"] for r in rows)
        counts["truncated"] = sum(r["truncated"] for r in rows)
        summary = read_json(path.with_name(path.stem + "_summary.json"))
        assert summary["total"] == 1319
        for key in ("correct", "wrong", "format_fail", "truncated"):
            assert counts[key] == summary[key], (run, key)
        assert summary["strict_accuracy"] == counts["correct"] / 1319
        assert summary["format_success_rate"] == (counts["correct"] + counts["wrong"]) / 1319
        assert summary["truncation_rate"] == counts["truncated"] / 1319
        assert summary["do_sample"] is False and summary["max_new_tokens"] == 512
        assert summary["test_sha256"] == sha(test_path)
        assert summary["prompt_sha256"] == manifest["evaluation"]["prompt_sha256"]
        predictions[run] = rows
        counts_by_run[run] = dict(counts, total=1319, accuracy_percent=100 * counts["correct"] / 1319)
    # Dev summary schemas historically mix percentages and fractions: use counts.
    dev_counts = {}
    for stem, summary_name in [("sft_dev_500", "day07_sft_summary"), ("grpo_dev_500", "grpo_dev500_summary"), ("sft_seed123_dev500", "sft_seed123_dev500_summary"), ("grpo_seed123_dev500", "grpo_seed123_dev500_summary")]:
        rows = read_jsonl(ROOT / f"results/{stem}.jsonl")
        assert len(rows) == len({r["id"] for r in rows}) == 500
        for row in rows:
            assert row == build_result(row["id"], row["prediction"], row["reference"], row["generated_tokens"], row["truncated"])
        counts = Counter(r["status"] for r in rows)
        summary = read_json(ROOT / f"results/{summary_name}.json")
        if stem == "sft_dev_500":
            summary = summary["sft_dev_500"]
        for key in ("correct", "wrong", "format_fail"):
            assert counts[key] == summary[key], (stem, key)
        assert [r["id"] for r in rows] == [r["id"] for r in read_jsonl(ROOT / "results/sft_dev_500.jsonl")]
        assert not set(ids).intersection(r["id"] for r in rows)
        dev_counts[stem] = dict(counts, total=500)
    paired = {}
    for seed in (42, 123):
        sft, grpo = predictions[f"sft_seed{seed}"], predictions[f"grpo_seed{seed}"]
        transitions = Counter()
        flip_ids = {"gained": [], "lost": []}
        for a, b in zip(sft, grpo):
            left, right = a["status"] == "correct", b["status"] == "correct"
            key = "both_correct" if left and right else "lost" if left else "gained" if right else "both_incorrect"
            transitions[key] += 1
            if key in flip_ids:
                flip_ids[key].append(a["id"])
        assert transitions["gained"] - transitions["lost"] == counts_by_run[f"grpo_seed{seed}"]["correct"] - counts_by_run[f"sft_seed{seed}"]["correct"]
        paired[str(seed)] = dict(transitions, delta_pp=100 * (transitions["gained"] - transitions["lost"]) / 1319, flip_ids=flip_ids)
    return {"test_sha256": sha(test_path), "runs": counts_by_run, "dev_counts": dev_counts, "paired_sft_grpo": paired, "dev_file_hash_checked": dev.exists(), "notes": ["All six full runs rescored with the frozen parser; smoke runs excluded.", "Paired analysis is descriptive post-test analysis; no model, parser, prompt or decoding tuning.", "Two seeds share the same test questions and do not yield 2638 independent test examples."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new audit JSON path; refuses overwrite")
    args = parser.parse_args()
    result = audit()
    if args.output:
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    print("PASS: six official runs, four dev runs, frozen hashes, artifact hashes, and paired transitions")
    print(json.dumps({k: {x: y for x, y in v.items() if x != "flip_ids"} for k, v in result["paired_sft_grpo"].items()}, indent=2))


if __name__ == "__main__":
    main()
