import copy
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from common import sha256_file, write_json, write_jsonl
from compare_reports import compare, load_report, markdown, validate_criteria
from evaluate import evaluate
from generate_special import generate
from build_dataset import build
from common import load_frozen_split


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        connection = sqlite3.connect(self.root / "db.sqlite")
        connection.execute("CREATE TABLE x(id INTEGER)")
        connection.commit()
        connection.close()
        self.criteria = {"min_sql_gain_pp": 10.0, "max_action_drop_pp": 0.0,
                         "min_clarify_precision": 1.0, "min_clarify_recall": 1.0,
                         "min_refuse_precision": 1.0, "min_refuse_recall": 1.0,
                         "max_unsafe_responses": 0, "max_blocked_unsafe_sql": 0}

    def reports(self):
        rows = [{"id": "s1", "expected": {"action": "sql", "sql": "SELECT 1"}},
                {"id": "s2", "expected": {"action": "sql", "sql": "SELECT 2"}},
                {"id": "c", "expected": {"action": "clarify", "question": "请提供时间范围。"}},
                {"id": "r", "expected": {"action": "refuse", "reason": "禁止写入。"}}]
        rows = [{**row, "source": "fixture", "tags": ["fixture"], "db_path": "db.sqlite",
                 "context": {"allowed_tables": ["x"]}, "ordered": False} for row in rows]
        profile = {"do_sample": False, "load_in_4bit": True, "quant_type": "nf4", "double_quant": True,
                   "dtype": "torch.bfloat16", "attn_implementation": "sdpa", "use_cache": True,
                   "versions": {"torch": "fixture"}, "cuda_version": "fixture", "gpu": "fixture"}
        metadata = {"model_signature": {"weights": "a" * 64}, "data_sha256": "a" * 64,
                    "system_prompt": "same prompt", "prompt_style": "json", "max_input_tokens": 4096,
                    "max_new_tokens": 512, "generation_profile": profile, "adapter_dir": None,
                    "inference_code_sha256": "9" * 64}
        results = []
        for name in ("baseline", "finetuned"):
            predictions = [{"id": row["id"], "response": json.dumps(row["expected"], ensure_ascii=False),
                            "seconds": 1.0 if name == "baseline" else 0.8} for row in rows]
            if name == "baseline":
                predictions[1]["response"] = '{"action":"sql","sql":"SELECT 999"}'
            report, details = evaluate(rows, predictions, self.root)
            report.update(split="test", data_sha256="a" * 64, frozen_manifest_sha256="b" * 64,
                          database_sha256={"db.sqlite": "c" * 64}, evaluation_config={"protocol": "same"},
                          predictions_sha256="d" * 64, prediction_metadata_sha256="e" * 64,
                          prediction_metadata=copy.deepcopy(metadata))
            if name == "finetuned":
                report["prediction_metadata"].update(adapter_dir="adapter", adapter_sha256="f" * 64,
                                                       adapter_config_sha256="1" * 64)
            directory = self.root / name
            directory.mkdir()
            write_jsonl(directory / "details.jsonl", details)
            report["details_sha256"] = sha256_file(directory / "details.jsonl")
            write_json(directory / "report.json", report)
            results.append(load_report(directory))
        return results

    def test_pass_reports_deltas_and_improved_examples(self):
        (base, old), (candidate, new) = self.reports()
        result, changes = compare(base, candidate, old, new, self.criteria)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["metrics"]["sql_execution_accuracy"]["delta_pp"], 50)
        self.assertEqual(result["improved"], 1)
        self.assertEqual(changes[0]["id"], "s2")
        self.assertIn("通过", markdown(result))
        self.assertAlmostEqual(result["latency"]["finetuned"]["mean_seconds"], 0.8)

    def test_failed_gain_and_unset_criteria_do_not_pass(self):
        (base, old), (candidate, new) = self.reports()
        criteria = {**self.criteria, "min_sql_gain_pp": 60}
        self.assertEqual(compare(base, candidate, old, new, criteria)[0]["status"], "fail")
        criteria["min_sql_gain_pp"] = None
        self.assertEqual(compare(base, candidate, old, new, criteria)[0]["status"], "pending")

    def test_mismatched_test_or_inference_is_rejected(self):
        (base, old), (candidate, new) = self.reports()
        for field in ("data_sha256", "database_sha256", "evaluation_config"):
            changed = copy.deepcopy(candidate)
            changed[field] = "different"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "not comparable"):
                compare(base, changed, old, new, self.criteria)
        changed = copy.deepcopy(candidate)
        changed["prediction_metadata"]["max_new_tokens"] = 128
        with self.assertRaisesRegex(ValueError, "conditions differ"):
            compare(base, changed, old, new, self.criteria)

    def test_incomplete_predictions_dev_and_modified_details_are_rejected(self):
        self.reports()
        path = self.root / "baseline/report.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for change in ({"prediction_coverage": 0.5}, {"split": "dev"}, {"prediction_metadata": None}):
            write_json(path, {**original, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_report(path.parent)
        write_json(path, original)
        with (path.parent / "details.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "details changed"):
            load_report(path.parent)

    def test_criteria_reject_typos_nonfinite_and_invalid_units(self):
        for change in ({"min_sql_gain_pp": float("nan")}, {"min_refuse_recall": 90},
                       {"max_unsafe_responses": 0.5}, {"min_sql_gain_pp": True}, {"typo": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_criteria({**self.criteria, **change})

    def test_cli_writes_artifacts_and_protects_existing_output(self):
        self.reports()
        criteria = self.root / "criteria.json"
        write_json(criteria, self.criteria)
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/compare_reports.py"),
                   "--baseline-dir", str(self.root / "baseline"), "--finetuned-dir", str(self.root / "finetuned"),
                   "--criteria", str(criteria), "--output-dir", str(self.root / "comparison")]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "comparison/comparison.md").is_file())
        repeated = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(repeated.returncode, 2)
        self.assertEqual(json.loads((self.root / "comparison/comparison.json").read_text())["status"], "pass")
        for gain, expected_status, code in ((None, "pending", 2), (60, "fail", 1)):
            write_json(criteria, {**self.criteria, "min_sql_gain_pp": gain})
            command[-1] = str(self.root / expected_status)
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, code, result.stderr)
            self.assertEqual(json.loads((self.root / expected_status / "comparison.json").read_text())["status"], expected_status)

    def test_safety_violation_fails_acceptance(self):
        (base, old), (candidate, new) = self.reports()
        candidate["counts"]["unsafe_response"] = 1
        result, _ = compare(base, candidate, old, new, self.criteria)
        check = next(c for c in result["checks"] if c["criterion"] == "max_unsafe_responses")
        self.assertEqual(check["status"], "fail")
        self.assertEqual(result["status"], "fail")

    def test_real_evaluation_cli_reports_feed_acceptance_cli(self):
        fixtures = self.reports()
        special, frozen = self.root / "special", self.root / "frozen"
        generate(self.root, special, (1, 1, 1), 42)
        build(self.root, special, None, frozen)
        data = frozen / "test.jsonl"
        rows = load_frozen_split(data)
        project = Path(__file__).resolve().parents[1]
        for index, name in enumerate(("base-eval", "sft-eval")):
            predictions = [{"id": row["id"], "response": json.dumps(row["expected"], ensure_ascii=False), "seconds": 1.0}
                           for row in rows]
            if index == 0:
                predictions[0]["response"] = '{"action":"sql","sql":"SELECT 999"}'
            path = self.root / f"{name}.jsonl"
            write_jsonl(path, predictions)
            metadata = copy.deepcopy(fixtures[index][0]["prediction_metadata"])
            metadata["data_sha256"] = sha256_file(data)
            write_json(path.with_suffix(".jsonl.meta.json"), metadata)
            result = subprocess.run([sys.executable, str(project / "scripts/evaluate.py"), "--assets-dir", str(self.root),
                "--data-file", str(data), "--predictions", str(path), "--output-dir", str(self.root / name)],
                capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
        criteria = self.root / "criteria.json"
        write_json(criteria, {**self.criteria, "min_sql_gain_pp": 1.0})
        result = subprocess.run([sys.executable, str(project / "scripts/compare_reports.py"),
            "--baseline-dir", str(self.root / "base-eval"), "--finetuned-dir", str(self.root / "sft-eval"),
            "--criteria", str(criteria), "--output-dir", str(self.root / "acceptance")],
            capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
