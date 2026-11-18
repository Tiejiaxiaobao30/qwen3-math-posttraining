"""Audit compact V2 evidence; optionally rescore the verified local backup (stdlib)."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import tarfile

from scripts.evaluator import build_result

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ARCHIVE_SHA256 = "404488230140ce611d64e3a90004214239ea25fe81128dc768965b4d6b6290ef"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compact(row):
    return {k: str(row[k]) for k in ("id", "reference", "parsed_answer", "status", "generated_tokens", "truncated")}


def audit(archive=None):
    provenance = read_json(ROOT / "reports/v2/provenance.json")
    expected = read_json(ROOT / "reports/v2/verification.json")
    for item in provenance["imported_files"] + provenance["derived_files"]:
        assert sha((ROOT / item["path"]).read_bytes()) == item["sha256"], item["path"]
    rows_by_split = {}
    for split, total in (("dev500", 500), ("test1319", 1319)):
        with (ROOT / f"results/v2/{split}_outcomes.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len({r["id"] for r in rows}) == total
        reference_path = "results/sft_dev_500.jsonl" if split == "dev500" else "data/official/gsm8k_test_1319.jsonl"
        reference_rows = [json.loads(s) for s in (ROOT / reference_path).read_text(encoding="utf-8").splitlines() if s.strip()]
        assert [r["id"] for r in rows] == [str(r["id"]) for r in reference_rows]
        assert [r["reference"] for r in rows] == [str(r["reference" if split == "dev500" else "final_answer"]) for r in reference_rows]
        counts = {}
        for g in (4, 8):
            statuses = Counter(r[f"g{g}_status"] for r in rows)
            assert set(statuses) <= {"correct", "wrong", "format_fail"}
            counts[str(g)] = dict(statuses, total=total, truncated=sum(r[f"g{g}_truncated"] == "True" for r in rows))
            assert counts[str(g)] == expected[split][str(g)]
            summary = read_json(ROOT / f"results/v2/v2_g{g}_{split}_summary.json")
            for key in ("total", "correct", "wrong", "format_fail", "truncated"):
                assert summary[key] == counts[str(g)][key], (split, g, key)
            accuracy = summary["strict_accuracy"] * (100 if split == "test1319" else 1)
            assert abs(accuracy - 100 * statuses["correct"] / total) < 1e-9
        transitions = Counter("both_correct" if r["g4_status"] == r["g8_status"] == "correct" else "lost" if r["g4_status"] == "correct" else "gained" if r["g8_status"] == "correct" else "both_incorrect" for r in rows)
        assert dict(transitions) == expected[f"{split}_g4_to_g8"]
        assert transitions["gained"] - transitions["lost"] == counts["8"]["correct"] - counts["4"]["correct"]
        rows_by_split[split] = rows
    cfg4, cfg8 = [read_json(ROOT / f"configs/v2_g{g}.json") for g in (4, 8)]
    cfg4["output_dir"] = cfg8["output_dir"]
    cfg4["generation"]["num_generations"] = 8
    assert cfg4 == cfg8
    for g in (4, 8):
        metrics = read_json(ROOT / f"reports/v2/training/g{g}_metrics.json")
        assert metrics["global_step"] == 500 and len(metrics["gradient_audit"]) == 500
        assert [r["optimizer_step"] for r in metrics["gradient_audit"]] == list(range(1, 501))
        assert metrics["nan_detected"] is False
        assert metrics["nonzero_gradient_steps"] == sum(r["nonzero_grad_tensors"] > 0 for r in metrics["gradient_audit"])
        assert metrics["train_metrics"]["train_runtime"] == expected["training"][str(g)]["train_runtime"]
        assert metrics["resolved_config"]["num_generations"] == g
    if archive:
        with archive.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        assert digest == EXPECTED_ARCHIVE_SHA256, "Archive SHA256 mismatch"
        with tarfile.open(archive, "r:gz") as tar:
            for item in provenance["source_members"]:
                assert sha(tar.extractfile(item["path"]).read()) == item["sha256"], item["path"]
            for item in provenance["imported_files"]:
                assert tar.extractfile(item["source"]).read() == (ROOT / item["path"]).read_bytes()
            for split in rows_by_split:
                for g in (4, 8):
                    source = f"results/v2/v2_g{g}_dev500.jsonl" if split == "dev500" else f"results/v2_official_test/v2_g{g}_test1319.jsonl"
                    raw = [json.loads(line) for line in tar.extractfile(source).read().decode("utf-8").splitlines()]
                    assert len(raw) == len(rows_by_split[split])
                    for row, saved in zip(raw, rows_by_split[split]):
                        assert row == build_result(row["id"], row["prediction"], row["reference"], row["generated_tokens"], row["truncated"])
                        assert compact(row) == {"id": saved["id"], "reference": saved["reference"], **{k: saved[f"g{g}_{k}"] for k in ("parsed_answer", "status", "generated_tokens", "truncated")}}
            frozen = read_json(ROOT / "results/frozen_eval_manifest.json")
            assert sha(tar.extractfile("outputs/sft_full/final_adapter/adapter_model.safetensors").read()) == frozen["artifacts"]["sft_seed42"]["sha256"]
            assert sha(tar.extractfile("data/processed/dev.jsonl").read()) == frozen["evaluation"]["dev500_sha256"]
            for name in ("configs/sft_prompt.txt", "scripts/evaluator.py", "scripts/train_grpo_smoke.py", "scripts/eval_official_test.py", "scripts/eval_grpo_pair_dev.py", "data/official/gsm8k_test_1319.jsonl", "results/frozen_eval_manifest.json"):
                assert tar.extractfile(name).read().replace(b'\r\n', b'\n') == (ROOT / name).read_bytes().replace(b'\r\n', b'\n'), name
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Verified local tar.gz; never uploaded")
    args = parser.parse_args()
    audit(args.archive)
    print("PASS: V2 counts, paired outcomes, hashes, configs and 500-step training records")
    print("PASS: all 3638 predictions rescored and backup lineage verified" if args.archive else "Full-text rescoring requires --archive; compact outcomes contain no generated responses.")


if __name__ == "__main__":
    main()
