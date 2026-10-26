#!/usr/bin/env python3
"""Compare matched base/adapter test reports and apply explicitly configured acceptance criteria."""

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

from common import load_jsonl, read_json, sha256_file, write_json, write_jsonl


CRITERIA = {
    "min_sql_gain_pp": 100, "max_action_drop_pp": 100,
    "min_clarify_precision": 1, "min_clarify_recall": 1,
    "min_refuse_precision": 1, "min_refuse_recall": 1,
    "max_unsafe_responses": None, "max_blocked_unsafe_sql": None,
}
METRICS = ("sql_execution_accuracy", "action_accuracy", "clarify_precision", "clarify_recall",
           "refuse_precision", "refuse_recall")


def validate_criteria(criteria):
    if not isinstance(criteria, dict) or set(criteria) != set(CRITERIA):
        raise ValueError(f"Criteria must contain exactly these keys: {', '.join(CRITERIA)}")
    for name, maximum in CRITERIA.items():
        value = criteria[name]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid nonnegative criterion: {name}")
        if maximum is None and type(value) is not int:
            raise ValueError(f"Count criterion must be an integer: {name}")
        if maximum is not None and value > maximum:
            raise ValueError(f"Criterion out of range: {name}")


def require_hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Missing or invalid SHA256: {name}")


def load_report(directory):
    report = read_json(directory / "report.json")
    for name in ("data_sha256", "frozen_manifest_sha256", "predictions_sha256", "details_sha256",
                 "prediction_metadata_sha256"):
        require_hash(report.get(name), name)
    if report.get("split") != "test":
        raise ValueError("Final acceptance requires the frozen test split, not dev/train.")
    details_path = directory / "details.jsonl"
    if sha256_file(details_path) != report["details_sha256"]:
        raise ValueError(f"Evaluation details changed: {directory}")
    details = load_jsonl(details_path)
    counts = report["counts"]
    if not details or type(counts.get("total")) is not int or counts["total"] != len(details):
        raise ValueError("Report total disagrees with details.")
    if type(report.get("prediction_coverage")) not in (int, float) or report["prediction_coverage"] != 1:
        raise ValueError("Both reports must cover every test example.")
    for name in METRICS:
        value = report.get(name)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"Invalid report metric: {name}")
    for action in ("sql", "clarify", "refuse"):
        selected = [d for d in details if d["expected_action"] == action]
        if not selected or type(counts.get(f"expected_{action}")) is not int or counts[f"expected_{action}"] != len(selected):
            raise ValueError(f"Test data needs verified {action} examples.")
    if sum(counts[f"expected_{action}"] for action in ("sql", "clarify", "refuse")) != len(details):
        raise ValueError("Details contain an unknown expected action.")
    if any(type(d.get("correct")) is not bool for d in details):
        raise ValueError("Details must contain boolean correctness flags.")
    sql = [d for d in details if d["expected_action"] == "sql"]
    sql_accuracy = sum(d["correct"] for d in sql) / len(sql)
    action_accuracy = sum(d.get("predicted_action") == d["expected_action"] for d in details) / len(details)
    if report["sql_execution_accuracy"] != sql_accuracy or report["action_accuracy"] != action_accuracy:
        raise ValueError("Report accuracy disagrees with per-example details.")
    for action in ("clarify", "refuse"):
        correct = sum(d.get("predicted_action") == action == d["expected_action"] for d in details)
        predicted = sum(d.get("predicted_action") == action for d in details)
        precision = correct / predicted if predicted else None
        recall = correct / counts[f"expected_{action}"]
        if report[f"{action}_precision"] != precision or report[f"{action}_recall"] != recall:
            raise ValueError(f"Report {action} metrics disagree with details.")
    for name in ("unsafe_response", "blocked_unsafe_sql"):
        value = counts.get(name, 0)
        if type(value) is not int or not 0 <= value <= len(details):
            raise ValueError(f"Invalid safety count: {name}")
    unsafe = sum(d["expected_action"] == "refuse" and d.get("predicted_action") == "sql" for d in details)
    blocked = sum(d.get("blocked_unsafe_sql") is True for d in details)
    if counts.get("unsafe_response", 0) != unsafe or counts.get("blocked_unsafe_sql", 0) != blocked:
        raise ValueError("Safety counts disagree with per-example details.")
    metadata = report.get("prediction_metadata")
    if not isinstance(metadata, dict) or metadata.get("data_sha256") != report["data_sha256"]:
        raise ValueError("Missing or mismatched prediction metadata; regenerate predictions/evaluation.")
    if metadata.get("prompt_style") != "json" or not metadata.get("model_signature") or not metadata.get("system_prompt"):
        raise ValueError("Acceptance requires the same full JSON behavior protocol and base-model fingerprint.")
    require_hash(metadata.get("inference_code_sha256"), "inference_code_sha256")
    profile = metadata.get("generation_profile", {})
    required = {"do_sample", "load_in_4bit", "quant_type", "double_quant", "dtype", "attn_implementation",
                "use_cache", "versions", "cuda_version", "gpu"}
    if not isinstance(profile, dict) or not required.issubset(profile):
        raise ValueError("Generation runtime/profile is missing; regenerate predictions.")
    if not report.get("database_sha256") or not report.get("evaluation_config"):
        raise ValueError("Database fingerprints and evaluation protocol are required.")
    return report, details


def compare(base, candidate, base_details, candidate_details, criteria):
    validate_criteria(criteria)
    for name in ("split", "data_sha256", "frozen_manifest_sha256", "database_sha256", "evaluation_config"):
        if base[name] != candidate[name]:
            raise ValueError(f"Reports are not comparable: {name} differs.")
    left, right = base["prediction_metadata"], candidate["prediction_metadata"]
    for name in ("model_signature", "system_prompt", "prompt_style", "max_input_tokens", "max_new_tokens",
                 "generation_profile", "inference_code_sha256"):
        if name not in left or left[name] != right.get(name):
            raise ValueError(f"Inference conditions differ: {name}")
    if left.get("adapter_dir") is not None or left.get("adapter_sha256") is not None:
        raise ValueError("Baseline must be the unadapted base model.")
    if not right.get("adapter_dir"):
        raise ValueError("Candidate must identify the evaluated adapter.")
    require_hash(right.get("adapter_sha256"), "adapter_sha256")
    require_hash(right.get("adapter_config_sha256"), "adapter_config_sha256")
    before = {d["id"]: d for d in base_details}
    after = {d["id"]: d for d in candidate_details}
    if before.keys() != after.keys():
        raise ValueError("Per-example IDs differ.")
    changes = []
    for identity, old in before.items():
        new = after[identity]
        if any(old[key] != new[key] for key in ("source", "tags", "expected_action")):
            raise ValueError(f"Per-example task identity differs: {identity}")
        if old["correct"] != new["correct"]:
            changes.append({"id": identity, "source": old["source"], "tags": old["tags"],
                            "expected_action": old["expected_action"],
                            "change": "improved" if new["correct"] else "regressed",
                            "baseline_error": old.get("error"), "finetuned_error": new.get("error")})
    metrics = {name: {"baseline": base.get(name), "finetuned": candidate.get(name),
                      "delta_pp": 100 * (candidate[name] - base[name])
                      if candidate.get(name) is not None and base.get(name) is not None else None} for name in METRICS}
    values = {
        "min_sql_gain_pp": metrics["sql_execution_accuracy"]["delta_pp"],
        "max_action_drop_pp": -metrics["action_accuracy"]["delta_pp"],
        **{f"min_{name}": candidate.get(name) for name in METRICS if name.startswith(("clarify", "refuse"))},
        "max_unsafe_responses": candidate["counts"].get("unsafe_response", 0),
        "max_blocked_unsafe_sql": candidate["counts"].get("blocked_unsafe_sql", 0),
    }
    checks = []
    for name, threshold in criteria.items():
        actual = values[name]
        status = "pending" if threshold is None else "fail" if actual is None else (
            "pass" if (actual + 1e-9 >= threshold if name.startswith("min_") else actual <= threshold + 1e-9) else "fail"
        )
        checks.append({"criterion": name, "threshold": threshold, "actual": actual, "status": status})
    statuses = {check["status"] for check in checks}
    status = "fail" if "fail" in statuses else "pending" if "pending" in statuses else "pass"
    groups = {}
    for key in sorted(base["groups"].keys() | candidate["groups"].keys()):
        old, new = base["groups"].get(key), candidate["groups"].get(key)
        if not old or not new or old["total"] != new["total"] or not old["total"]:
            raise ValueError(f"Group coverage differs: {key}")
        groups[key] = {"total": old["total"], "baseline": old["correct"] / old["total"],
                       "finetuned": new["correct"] / new["total"],
                       "delta_pp": 100 * (new["correct"] - old["correct"]) / old["total"]}
    result = {
        "status": status, "scope": "automatic_checks_only_manual_semantic_review_required",
        "test_data_sha256": base["data_sha256"], "criteria": criteria, "checks": checks,
        "metrics": metrics, "groups": groups, "test_examples": base["counts"]["total"],
        "improved": sum(c["change"] == "improved" for c in changes),
        "regressed": sum(c["change"] == "regressed" for c in changes),
        "safety": {"baseline": {name: base["counts"].get(name, 0) for name in ("unsafe_response", "blocked_unsafe_sql")},
                   "finetuned": {name: candidate["counts"].get(name, 0) for name in ("unsafe_response", "blocked_unsafe_sql")}},
        "latency": {"baseline": base.get("latency"), "finetuned": candidate.get("latency")},
        "limitations": list(dict.fromkeys(base.get("limitations", []) + candidate.get("limitations", []))) + [
            "Automatic pass does not replace manual review of metric definitions, clarification semantics or permissions.",
            "Latency is descriptive on the recorded GPU/runtime, not a controlled performance benchmark.",
            "Thresholds must be fixed before formal training; this script cannot prove when they were chosen.",
        ],
    }
    return result, changes


def markdown(result):
    labels = {"pass": "通过", "fail": "未通过", "pending": "待配置"}
    lines = ["# 微调对比与自动验收报告", "", f"自动检查结论：**{labels[result['status']]}**。", "",
             "此结论仍需人工复核指标口径、澄清语义与安全用例。", "",
             f"测试样本：{result['test_examples']}；改善：{result['improved']}；回退：{result['regressed']}。", "",
             "## 总体指标", "", "| 指标 | 基座 | 微调 | 变化（百分点） |", "|---|---:|---:|---:|"]
    for name, value in result["metrics"].items():
        old = "未定义" if value["baseline"] is None else f"{100 * value['baseline']:.2f}%"
        new = "未定义" if value["finetuned"] is None else f"{100 * value['finetuned']:.2f}%"
        delta = "—" if value["delta_pp"] is None else f"{value['delta_pp']:+.2f}"
        lines.append(f"| {name} | {old} | {new} | {delta} |")
    lines += ["", "## 验收条件", "", "| 条件 | 阈值 | 实际值 | 结果 |", "|---|---:|---:|---|"]
    for check in result["checks"]:
        lines.append(f"| {check['criterion']} | {check['threshold']} | {check['actual']} | {labels[check['status']]} |")
    lines += ["", "## 分来源与能力标签", "", "| 分组 | 样本数 | 基座 | 微调 | 变化（百分点） |", "|---|---:|---:|---:|---:|"]
    for name, value in result["groups"].items():
        lines.append(f"| {name.replace('|', '/')} | {value['total']} | {100 * value['baseline']:.2f}% | {100 * value['finetuned']:.2f}% | {value['delta_pp']:+.2f} |")
    lines += ["", "分组 correct 沿用原评测口径；澄清/拒绝的 action 正确不等于语义已审定。", "",
              "## 成本与安全", "", "```json", json.dumps({"latency": result["latency"], "safety": result["safety"],
              "training": result.get("training")}, ensure_ascii=False, indent=2), "```", "",
              "延迟为成功生成且记录了耗时的样本统计；样本数不同或错误较多时不宜直接比较。", "",
              "## 限制", ""]
    lines += ["- " + note for note in result["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--finetuned-dir", type=Path, required=True)
    parser.add_argument("--criteria", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-metrics", type=Path)
    args = parser.parse_args()
    try:
        if args.output_dir.exists():
            raise FileExistsError(f"Use a new comparison output directory: {args.output_dir}")
        criteria = read_json(args.criteria)
        baseline, old = load_report(args.baseline_dir)
        finetuned, new = load_report(args.finetuned_dir)
        result, changes = compare(baseline, finetuned, old, new, criteria)
        result["provenance"] = {
            "baseline_report_sha256": sha256_file(args.baseline_dir / "report.json"),
            "finetuned_report_sha256": sha256_file(args.finetuned_dir / "report.json"),
            "criteria_sha256": sha256_file(args.criteria),
        }
        if args.training_metrics:
            training = read_json(args.training_metrics)
            result["training"] = {key: training[key] for key in ("train_runtime", "train_samples_per_second",
                "train_steps_per_second", "peak_cuda_memory_bytes", "gpu") if key in training}
            result["provenance"]["training_metrics_sha256"] = sha256_file(args.training_metrics)
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".compare-", dir=args.output_dir.parent) as temporary:
            staging = Path(temporary)
            write_json(staging / "comparison.json", result)
            write_jsonl(staging / "changes.jsonl", changes)
            (staging / "comparison.md").write_text(markdown(result), encoding="utf-8")
            staging.rename(args.output_dir)
        print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}, ensure_ascii=False))
        return {"pass": 0, "fail": 1, "pending": 2}[result["status"]]
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(f"Comparison rejected: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
