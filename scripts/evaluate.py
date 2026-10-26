#!/usr/bin/env python3
"""Evaluate frozen examples: strict-column execution match and action behavior."""

import argparse
import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import sqlglot

from common import (UnsafeSQLError, execute_sql, load_frozen_split, load_jsonl, parse_response, read_json,
                    results_equal, safe_path, sha256_file, write_json, write_jsonl)


def evaluate(rows, predictions, assets, timeout=2.0):
    known_ids = {row["id"] for row in rows}
    if any(prediction["id"] not in known_ids for prediction in predictions):
        raise ValueError("Predictions contain IDs absent from the evaluation split.")
    by_id = {prediction["id"]: prediction for prediction in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("Duplicate prediction IDs")
    details, counts, groups = [], Counter(), defaultdict(Counter)
    for row in rows:
        expected = row["expected"]["action"]
        counts["total"] += 1
        counts[f"expected_{expected}"] += 1
        detail = {"id": row["id"], "source": row["source"], "tags": row["tags"], "expected_action": expected, "correct": False}
        try:
            if row["id"] not in by_id:
                raise ValueError("Missing prediction")
            prediction = by_id[row["id"]]
            answer = parse_response(prediction["response"], prediction.get("response_style", "json"))
            action = answer["action"]
            detail["predicted_action"] = action
            counts[f"predicted_{action}"] += 1
            if action == expected:
                counts["action_correct"] += 1
                counts[f"action_correct_{expected}"] += 1
            if action == "sql":
                database = safe_path(assets, row["db_path"])
                if expected == "refuse":
                    counts["unsafe_response"] += 1
                predicted_result = execute_sql(database, answer["sql"], row["context"].get("allowed_tables"), timeout)
                counts["executable_sql"] += 1
                if expected == "sql":
                    try:
                        gold = execute_sql(database, row["expected"]["sql"], row["context"].get("allowed_tables"), timeout)
                    except (ValueError, OSError, sqlite3.Error) as error:
                        raise RuntimeError(f"Invalid reference for {row['id']}: {error}") from error
                    detail["correct"] = results_equal(gold, predicted_result, row["ordered"])
                    if detail["correct"] and not gold[1]:
                        counts["empty_result_matches"] += 1
            elif action == expected:
                detail["correct"] = True
                if expected == "clarify":
                    groups_to_match = row.get("clarification_terms", [])
                    detail["clarification_keyword_coverage"] = bool(groups_to_match) and all(
                        any(term in answer["question"] for term in group) for group in groups_to_match
                    )
                    counts["clarification_keyword_matches"] += detail["clarification_keyword_coverage"]
        except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as error:
            detail["error"] = str(error)
            counts["errors"] += 1
            if isinstance(error, UnsafeSQLError) or (isinstance(error, sqlite3.DatabaseError)
                                                    and ("authorized" in str(error) or "prohibited" in str(error))):
                counts["blocked_unsafe_sql"] += 1
                detail["blocked_unsafe_sql"] = True
        counts["correct"] += detail["correct"]
        if expected == "sql":
            counts["sql_correct"] += detail["correct"]
        for group in ["source:" + row["source"]] + ["tag:" + tag for tag in row["tags"]]:
            groups[group]["total"] += 1
            groups[group]["correct"] += detail["correct"]
        details.append(detail)
    report = {"counts": dict(counts), "prediction_coverage": len(predictions) / len(rows),
              "sql_execution_accuracy": counts["sql_correct"] / counts["expected_sql"] if counts["expected_sql"] else None,
              "action_accuracy": counts["action_correct"] / len(rows), "groups": {key: dict(value) for key, value in groups.items()},
              "limitations": ["Single-database result comparison is not a proof of SQL equivalence.",
                              "Column order and duplicate rows are significant; this is not the official Spider metric.",
                              "Clarification keyword coverage is only a proxy and needs manual semantic review.",
                              "Synthetic-template test scores are reported separately and do not prove real-business generalization."]}
    for action in ("clarify", "refuse"):
        report[f"{action}_precision"] = counts[f"action_correct_{action}"] / counts[f"predicted_{action}"] if counts[f"predicted_{action}"] else None
        report[f"{action}_recall"] = counts[f"action_correct_{action}"] / counts[f"expected_{action}"] if counts[f"expected_{action}"] else None
    seconds = sorted(p["seconds"] for p in predictions if isinstance(p.get("seconds"), (int, float))
                     and not isinstance(p["seconds"], bool) and math.isfinite(p["seconds"])
                     and p["seconds"] >= 0 and not p.get("error") and p.get("response"))
    report["latency"] = {
        "timed_predictions": len(seconds),
        "mean_seconds": statistics.mean(seconds) if seconds else None,
        "p50_seconds": statistics.median(seconds) if seconds else None,
        "p95_seconds": seconds[math.ceil(0.95 * len(seconds)) - 1] if seconds else None,
    }
    return report, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    rows = load_frozen_split(args.data_file)
    manifest = read_json(args.data_file.parent / "manifest.json")
    for relative in {row["db_path"] for row in rows}:
        if sha256_file(safe_path(args.assets_dir, relative)) != manifest["databases"][relative]:
            raise ValueError(f"Database changed since dataset freeze: {relative}")
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new report directory: {args.output_dir}")
    report, details = evaluate(rows, load_jsonl(args.predictions), args.assets_dir, args.timeout)
    report.update(data_sha256=sha256_file(args.data_file), predictions_sha256=sha256_file(args.predictions))
    metadata_file = args.predictions.with_suffix(args.predictions.suffix + ".meta.json")
    prediction_metadata = read_json(metadata_file) if metadata_file.is_file() else None
    if prediction_metadata is not None and prediction_metadata.get("data_sha256") != report["data_sha256"]:
        raise ValueError("Prediction metadata refers to a different frozen split.")
    report.update(
        split=args.data_file.stem,
        frozen_manifest_sha256=sha256_file(args.data_file.parent / "manifest.json"),
        database_sha256={relative: manifest["databases"][relative] for relative in sorted({r["db_path"] for r in rows})},
        evaluation_config={
            "protocol": "sqlite_execution_v1", "timeout_seconds": args.timeout,
            "sqlite_version": sqlite3.sqlite_version, "sqlglot_version": sqlglot.__version__,
            "code_sha256": {name: sha256_file(Path(__file__).parent / name) for name in ("common.py", "evaluate.py")},
        },
        prediction_metadata=prediction_metadata,
        prediction_metadata_sha256=sha256_file(metadata_file) if prediction_metadata is not None else None,
    )
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "details.jsonl", details)
    report["details_sha256"] = sha256_file(args.output_dir / "details.jsonl")
    write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
