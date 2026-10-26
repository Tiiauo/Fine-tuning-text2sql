import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from common import (UnsafeSQLError, execute_sql, load_frozen_split, load_jsonl, parse_response,
                    model_signature, results_equal, tokenize_example, write_json, write_jsonl)
from prepare_data import prepare, rebuild_sesql
from generate_special import generate
from build_dataset import build
from evaluate import evaluate
import train_qlora


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.assets = Path(self.temporary.name)

    def database(self, path, table="items"):
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        connection.execute(f'CREATE TABLE "{table}" (id INTEGER, amount INTEGER)')
        connection.executemany(f'INSERT INTO "{table}" VALUES (?,?)', [(1, 1), (2, 2)])
        connection.commit()
        connection.close()
        return path

    def raw_fixture(self):
        raw = self.assets / "data/raw/cspider"
        schemas = []
        for split in ("train", "dev", "test"):
            table = "items_" + split
            folder = "test_database" if split == "test" else "database"
            self.database(raw / folder / split / f"{split}.sqlite", table)
            schema = {"db_id": split, "table_names_original": [table],
                      "column_names_original": [[-1, "*"], [0, "id"], [0, "amount"]], "foreign_keys": []}
            schemas.append(schema)
            row = {"db_id": split, "question": "合计金额？", "query": f"SELECT SUM(amount) FROM {table}"}
            relative = "test_data/test.json" if split == "test" else f"{split}.json"
            path = raw / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, [row, dict(row), {**row, "query": f"DELETE FROM {table}"}])
        write_json(raw / "tables.json", schemas)
        write_json(raw / "test_data/tables_test.json", schemas)

    def test_prepare_generate_freeze_evaluate_end_to_end(self):
        self.raw_fixture()
        prepared, special, frozen = (self.assets / name for name in ("prepared", "special", "frozen"))
        prepare(self.assets, prepared)
        self.assertEqual(len(load_jsonl(prepared / "rejected.jsonl")), 6)
        generate(self.assets, special, (1, 1, 1), 42)
        build(self.assets, prepared, special, frozen)
        rows = load_frozen_split(frozen / "dev.jsonl", "dev")
        predictions = [{"id": row["id"], "response": json.dumps(row["expected"], ensure_ascii=False)} for row in rows]
        report, details = evaluate(rows, predictions, self.assets)
        self.assertEqual(report["prediction_coverage"], 1)
        self.assertEqual(report["sql_execution_accuracy"], 1)
        self.assertEqual(report["action_accuracy"], 1)
        predictions[0]["response"] = '{"action":"sql","sql":"SELECT 999"}'
        report, _ = evaluate(rows, predictions, self.assets)
        self.assertLess(report["sql_execution_accuracy"], 1)
        with (frozen / "dev.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "hash"):
            load_frozen_split(frozen / "dev.jsonl")

    def test_sql_sandbox_blocks_writes_permissions_and_multiple_statements(self):
        database = self.database(self.assets / "db.sqlite")
        for sql in ("DELETE FROM items", "SELECT 1; SELECT 2", "PRAGMA journal_mode=WAL"):
            with self.subTest(sql=sql), self.assertRaises(UnsafeSQLError):
                execute_sql(database, sql)
        with self.assertRaises(sqlite3.DatabaseError):
            execute_sql(database, "SELECT * FROM items", allowed_tables=[])
        with self.assertRaises(sqlite3.DatabaseError):
            execute_sql(database, "SELECT load_extension('not-a-library')")
        self.assertEqual(execute_sql(database, "SELECT COUNT(*) FROM items")[1], [(2,)])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            execute_sql(database, "SELECT * FROM items", max_rows=1)

    def test_results_keep_duplicates_column_order_and_sort_semantics(self):
        self.assertTrue(results_equal((["x"], [(1,), (2,)]), (["y"], [(2.0,), (1.0,)])))
        self.assertFalse(results_equal((["x"], [(1,), (2,)]), (["x"], [(2,), (1,)]), ordered=True))
        self.assertFalse(results_equal((["x"], [(1,), (1,)]), (["x"], [(1,), (2,)])))
        self.assertFalse(results_equal((["a", "b"], [(1, 2)]), (["b", "a"], [(2, 1)])))

    def test_sesql_preserves_non_numeric_strings(self):
        path = self.assets / "rebuilt.sqlite"
        content = {"tables": {"例表": {"header": ["编号", "金额（元）"], "type": ["number", "number"],
                                     "cell": [["item_1", "12.5"]]}}}
        schema = {"table_names_original": ["例表"], "column_names_original": [[-1, "*"], [0, "编号"], [0, "金额（元）"]]}
        rebuild_sesql(content, schema, path)
        self.assertEqual(execute_sql(path, 'SELECT 编号,金额（元） FROM 例表')[1], [("item_1", 12.5)])

    def test_build_rejects_cross_split_schema_leakage(self):
        self.raw_fixture()
        prepared = self.assets / "prepared"
        prepare(self.assets, prepared)
        train = load_jsonl(prepared / "train.jsonl")
        dev = load_jsonl(prepared / "dev.jsonl")
        dev[0]["schema_id"] = train[0]["schema_id"]
        write_jsonl(prepared / "dev.jsonl", dev)
        with self.assertRaisesRegex(ValueError, "leakage"):
            build(self.assets, prepared, None, self.assets / "frozen")
        self.assertFalse((self.assets / "frozen").exists())

    def test_completion_only_mask_and_non_thinking_template(self):
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
                if enable_thinking:
                    raise AssertionError("Thinking must remain disabled")
                text = "".join(message["role"] + ":" + message["content"] + "|" for message in messages)
                return list(text + ("assistant:" if add_generation_prompt else ""))

        row = {"id": "test", "question": "query?", "context": {"schema": "CREATE TABLE x(id INT)"},
               "expected": {"action": "sql", "sql": "SELECT 1"}}
        encoded = tokenize_example(Tokenizer(), row)
        self.assertIn(0, encoded["completion_mask"])
        self.assertIn(1, encoded["completion_mask"])
        self.assertTrue(all(encoded["completion_mask"][i] <= encoded["completion_mask"][i + 1]
                            for i in range(len(encoded["completion_mask"]) - 1)))
        with self.assertRaises(ValueError):
            parse_response('{"action":"sql","sql":"SELECT 1","unexpected":true}')

    def test_model_signature_detects_changed_weights(self):
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (self.assets / name).write_text("{}")
        write_json(self.assets / "model.safetensors.index.json", {"weight_map": {"x": "weights.safetensors"}})
        weights = self.assets / "weights.safetensors"
        weights.write_bytes(b"original")
        before = model_signature(self.assets)
        weights.write_bytes(b"modified")
        self.assertNotEqual(before, model_signature(self.assets))

    def test_missing_predictions_are_not_excluded_from_denominator(self):
        row = {"id": "x", "source": "test", "tags": [], "expected": {"action": "refuse", "reason": "no"}}
        report, details = evaluate([row], [], self.assets)
        self.assertEqual(report["prediction_coverage"], 0)
        self.assertEqual(report["action_accuracy"], 0)
        self.assertFalse(details[0]["correct"])

    def test_runaway_recursive_query_is_interrupted(self):
        database = self.database(self.assets / "db.sqlite")
        query = "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x WHERE n<100000000) SELECT SUM(n) FROM x"
        with self.assertRaisesRegex(sqlite3.OperationalError, "interrupted"):
            execute_sql(database, query, timeout=0.01)

    def test_malformed_model_response_does_not_abort_evaluation(self):
        row = {"id": "x", "source": "test", "tags": [], "expected": {"action": "refuse", "reason": "no"}}
        for response in ("```", "```json\n{}", None, '{"action":[]}'):
            with self.subTest(response=response):
                report, details = evaluate([row], [{"id": "x", "response": response}], self.assets)
                self.assertEqual(report["counts"]["errors"], 1)
                self.assertFalse(details[0]["correct"])
        self.assertEqual(parse_response('```json\n{"action":"refuse","reason":"no"}\n```')["action"], "refuse")

    def test_command_line_pipeline_and_gpu_script_help(self):
        self.raw_fixture()
        project = Path(__file__).resolve().parents[1]

        def run(script, *arguments):
            result = subprocess.run([sys.executable, str(project / "scripts" / script), *map(str, arguments)],
                                    cwd=project, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        run("prepare_data.py", "--assets-dir", self.assets)
        run("generate_special.py", "--assets-dir", self.assets, "--train-databases", 1,
            "--dev-databases", 1, "--test-databases", 1)
        run("build_dataset.py", "--assets-dir", self.assets, "--special-dir", self.assets / "data/special")
        data = self.assets / "data/sft-v1/dev.jsonl"
        predictions = self.assets / "predictions.jsonl"
        write_jsonl(predictions, [{"id": row["id"], "response": json.dumps(row["expected"], ensure_ascii=False)}
                                  for row in load_frozen_split(data)])
        run("evaluate.py", "--assets-dir", self.assets, "--data-file", data,
            "--predictions", predictions, "--output-dir", self.assets / "report")
        report = json.loads((self.assets / "report/report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["sql_execution_accuracy"], 1)
        run("predict.py", "--help")
        run("train_qlora.py", "--help")

    def test_resume_rejects_missing_random_state_before_loading_gpu(self):
        self.raw_fixture()
        prepared, frozen = self.assets / "prepared", self.assets / "frozen"
        prepare(self.assets, prepared)
        build(self.assets, prepared, None, frozen)
        model = self.assets / "model"
        model.mkdir()
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (model / name).write_text("{}")
        write_json(model / "model.safetensors.index.json", {"weight_map": {"x": "weights.safetensors"}})
        (model / "weights.safetensors").write_bytes(b"fixture")
        output = self.assets / "run"
        checkpoint = output / "checkpoint-2"
        checkpoint.mkdir(parents=True)
        for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_config.json", "adapter_model.safetensors"):
            (checkpoint / name).write_bytes(b"fixture")
        arguments = ["train_qlora.py", "--model-dir", str(model), "--data-dir", str(frozen),
                     "--output-dir", str(output), "--resume-from-checkpoint", str(checkpoint)]
        with patch.object(sys, "argv", arguments), patch.object(train_qlora, "version", return_value="fixture"):
            with self.assertRaisesRegex(ValueError, "rng_state.pth"):
                train_qlora.main()


if __name__ == "__main__":
    unittest.main()
