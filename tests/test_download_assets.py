import importlib.util
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location(
    "download_assets", Path(__file__).resolve().parents[1] / "scripts" / "download_assets.py"
)
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def model_fixture(self):
        for name, data in (
            ("config.json", {"model_type": "qwen3"}),
            ("tokenizer.json", {}), ("tokenizer_config.json", {}), ("generation_config.json", {}),
            ("model.safetensors.index.json", {
                "weight_map": {"weight": "weights.safetensors"}, "metadata": {"total_size": 4},
            }),
        ):
            (self.root / name).write_text(json.dumps(data), encoding="utf-8")
        header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        shard = self.root / "weights.safetensors"
        shard.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
        return shard

    def test_complete_and_truncated_checkpoint(self):
        shard = self.model_fixture()
        self.assertEqual(assets.validate_model(self.root)["tensor_bytes"], 4)
        shard.write_bytes(shard.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "Truncated"):
            assets.validate_model(self.root)

    def test_path_escape_in_checkpoint_index(self):
        self.model_fixture()
        (self.root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"weight": "../outside.safetensors"}, "metadata": {"total_size": 4},
        }))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            assets.validate_model(self.root)

    def test_unsafe_archives_never_publish_or_escape(self):
        for name in ("../outside", "/outside", "C:/outside"):
            with self.subTest(name=name):
                archive = self.root / "bad.zip"
                with zipfile.ZipFile(archive, "w") as bundle:
                    bundle.writestr(name, "bad")
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    assets.install_archive(archive, self.root / "installed", "dataset", lambda p: {}, "test")
                self.assertFalse((self.root / "installed").exists())
                self.assertFalse((self.root / "outside").exists())
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            assets.safe_path(self.root, "folder\\outside")

    def test_failed_validation_does_not_publish(self):
        archive = self.root / "data.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("dataset/data.json", "invalid")
        with self.assertRaises(json.JSONDecodeError):
            assets.install_archive(archive, self.root / "installed", "dataset",
                                   lambda p: assets.read_json(p / "data.json"), "test")
        self.assertFalse((self.root / "installed").exists())

    def test_repeated_install_preserves_existing_files(self):
        archive = self.root / "data.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("dataset/data.json", '{"rows": 1}')
        destination = self.root / "installed"
        validator = lambda p: assets.read_json(p / "data.json")
        assets.install_archive(archive, destination, "dataset", validator, "test")
        (destination / "notes.txt").write_text("keep")
        assets.install_archive(archive, destination, "dataset", validator, "test")
        self.assertEqual((destination / "notes.txt").read_text(), "keep")

    def test_cspider_requires_databases_for_every_split(self):
        for relative in ("train.json", "dev.json", "test_data/test.json"):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps([{"db_id": "tiny", "question": "count?", "query": "SELECT 1"}]))
        for relative in ("tables.json", "test_data/tables_test.json"):
            (self.root / relative).write_text('[{"db_id": "tiny"}]')
        for relative in ("database/tiny/tiny.sqlite", "test_database/tiny/tiny.sqlite"):
            target = self.root / relative
            target.parent.mkdir(parents=True)
            target.write_bytes(b"SQLite format 3\x00")
        self.assertEqual(assets.validate_cspider(self.root)["referenced_databases"], 2)
        (self.root / "test_database/tiny/tiny.sqlite").unlink()
        with self.assertRaises(FileNotFoundError):
            assets.validate_cspider(self.root)

    def test_archive_digest_rejects_corruption(self):
        archive = self.root / "data.zip"
        archive.write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            assets.download_archive("https://unused", archive, 3, "0" * 64)

    def test_model_providers_use_the_selected_revision(self):
        for source, module in (("modelscope", "modelscope.hub.snapshot_download"),
                               ("huggingface", "huggingface_hub")):
            with self.subTest(source=source):
                sdk = Mock()
                with patch.dict("sys.modules", {module: sdk}), patch.object(assets, "validate_model", return_value={}):
                    assets.download_model(self.root / source, source, "fixed-revision")
                self.assertEqual(sdk.snapshot_download.call_args.kwargs["revision"], "fixed-revision")


if __name__ == "__main__":
    unittest.main()
