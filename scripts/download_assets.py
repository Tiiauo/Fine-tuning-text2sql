#!/usr/bin/env python3
"""Download the selected Qwen3-8B checkpoint and original Text2SQL datasets."""

import argparse
import hashlib
import json
import stat
import struct
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


MODEL_ID = "Qwen/Qwen3-8B"
CSPIDER_ID = "1ecF1YUhbvFgN6lXzzIQeEwdiuhWj_o32"
CSPIDER_URL = f"https://drive.google.com/uc?id={CSPIDER_ID}"
CSPIDER_SHA256 = "edea769706e91bd71741e61de30c41ab8da97b365e15eb492c02ed281f3ebaf8"
SESQL_URL = (
    "https://raw.githubusercontent.com/SUDA-LA/SeSQL/"
    "4c62ccf316c3678ebd63bf0777440d825d5661da/data/SeSQL.zip"
)
SESQL_SHA256 = "0f5d4e554e4b9d7d6a87b9e64fa7c11883445a9e4cfc4a81299ac5a02d197ce3"


def read_json(path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def safe_path(root, relative):
    path = PurePosixPath(relative)
    if not path.parts or "\\" in relative or ":" in relative or path.is_absolute():
        raise ValueError(f"Unsafe relative path: {relative!r}")
    if ".." in path.parts:
        raise ValueError(f"Unsafe relative path: {relative!r}")
    result = root.joinpath(*path.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes output directory: {relative!r}")
    return result


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_model(directory):
    config = read_json(directory / "config.json")
    if config.get("model_type") != "qwen3":
        raise ValueError("Checkpoint config is not Qwen3.")
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        read_json(directory / name)
    index = read_json(directory / "model.safetensors.index.json")
    shards = set(index["weight_map"].values())
    if not shards:
        raise ValueError("Checkpoint index has no weight shards.")
    data_size = 0
    for name in sorted(shards):
        shard = safe_path(directory, name)
        with shard.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Truncated weight shard: {name}")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_size <= min(16 * 1024**2, shard.stat().st_size - 8):
                raise ValueError(f"Invalid safetensors header: {name}")
            header = json.loads(stream.read(header_size))
        offsets = [value["data_offsets"] for key, value in header.items() if key != "__metadata__"]
        if not offsets or any(
            len(pair) != 2 or any(type(v) is not int for v in pair)
            or not 0 <= pair[0] <= pair[1] for pair in offsets
        ):
            raise ValueError(f"Invalid tensor offsets: {name}")
        payload_size = max(pair[1] for pair in offsets)
        if shard.stat().st_size != 8 + header_size + payload_size:
            raise ValueError(f"Truncated or oversized weight shard: {name}")
        data_size += payload_size
    if data_size != index["metadata"]["total_size"]:
        raise ValueError("Weight sizes disagree with the checkpoint index.")
    return {"weight_shards": len(shards), "tensor_bytes": data_size}


def download_model(output, source, revision):
    destination = output / "models" / "Qwen3-8B"
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_ID} from {source} -> {destination}", flush=True)
    patterns = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE*", "README*"]
    if source == "modelscope":
        from modelscope.hub.snapshot_download import snapshot_download

        snapshot_download(
            MODEL_ID, revision=revision, local_dir=str(destination),
            allow_file_pattern=patterns, max_workers=4,
        )
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=MODEL_ID, revision=revision, local_dir=str(destination),
            allow_patterns=patterns, max_workers=4,
        )
    summary = validate_model(destination)
    summary.update(model_id=MODEL_ID, source=source, revision=revision)
    (destination / "download_info.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Model validated: {summary}", flush=True)


def download_archive(url, destination, expected_size, expected_sha256=None):
    if not destination.exists():
        import gdown

        destination.parent.mkdir(parents=True, exist_ok=True)
        result = gdown.download(
            url=url, output=str(destination), resume=True,
            timeout=60, retries=3, use_cookies=False,
        )
        if result is None:
            raise RuntimeError(f"Download failed: {destination.name}")
    if destination.stat().st_size != expected_size:
        raise ValueError(f"Archive size mismatch: {destination}. Keep it for inspection or move it aside before retrying.")
    if expected_sha256 and sha256_file(destination) != expected_sha256:
        raise ValueError(f"Archive SHA256 mismatch: {destination}")
    return destination


def validate_cspider(directory):
    summary = {}
    database_ids = set()
    for split, relative, schema_name, database_folder in (
        ("train", "train.json", "tables.json", "database"),
        ("dev", "dev.json", "tables.json", "database"),
        ("test", "test_data/test.json", "test_data/tables_test.json", "test_database"),
    ):
        examples = read_json(directory / relative)
        schemas = {schema["db_id"] for schema in read_json(directory / schema_name)}
        if not isinstance(examples, list) or not examples:
            raise ValueError(f"CSpider {split} is empty or has an unexpected format.")
        for example in examples:
            db_id = example["db_id"]
            if db_id not in schemas or not example.get("question") or not example.get("query"):
                raise ValueError(f"CSpider {split} has missing question, SQL or schema: {db_id}")
            database = safe_path(directory / database_folder, f"{db_id}/{db_id}.sqlite")
            database_ids.add(database)
        summary[split] = len(examples)
    for database in database_ids:
        with database.open("rb") as stream:
            if stream.read(16) != b"SQLite format 3\x00":
                raise ValueError(f"Missing or invalid SQLite database: {database}")
    summary["referenced_databases"] = len(database_ids)
    return summary


def validate_sesql(directory):
    for name in ("tables.json", "db_content.json"):
        if not read_json(directory / name):
            raise ValueError(f"SeSQL {name} is empty.")
    counts = {}
    for split in ("train", "dev", "test"):
        examples = read_json(directory / "single-round-question-completed" / f"{split}.json")
        if not isinstance(examples, list) or not examples:
            raise ValueError(f"SeSQL single-round {split} is empty or invalid.")
        counts[split] = len(examples)
    return counts


def install_archive(archive, destination, archive_root, validator, source):
    if destination.exists():
        summary = validator(destination)
        print(f"Existing dataset validated: {destination}: {summary}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as bundle:
            seen = set()
            for entry in bundle.infolist():
                target = safe_path(staging, entry.filename)
                if target in seen or stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError(f"Duplicate or symlink ZIP entry: {entry.filename}")
                seen.add(target)
            # Extracting into a fresh directory verifies each member's CRC before publication.
            bundle.extractall(staging)
        extracted = safe_path(staging, archive_root)
        if not extracted.is_dir():
            raise ValueError(f"Expected archive directory is missing: {archive_root}")
        summary = validator(extracted)
        summary.update(source=source, archive_sha256=sha256_file(archive))
        (extracted / "download_info.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        extracted.rename(destination)
    print(f"Dataset ready: {destination}: {summary}", flush=True)


def download_data(output, cspider_archive=None, include_sesql=False):
    raw = output / "data" / "raw"
    cspider = raw / "cspider"
    if cspider.exists():
        print(f"CSpider validated: {validate_cspider(cspider)}", flush=True)
    else:
        if cspider_archive is not None and not cspider_archive.is_file():
            raise FileNotFoundError(f"Uploaded CSpider archive not found: {cspider_archive}")
        archive = download_archive(
            CSPIDER_URL, cspider_archive or output / "downloads" / "full_CSpider.zip",
            109744506, CSPIDER_SHA256,
        )
        install_archive(
            archive, cspider, "full_CSpider/CSpider", validate_cspider,
            "https://taolusi.github.io/CSpider-explorer/",
        )
    if include_sesql:
        sesql = raw / "sesql"
        if sesql.exists():
            print(f"SeSQL validated: {validate_sesql(sesql)}", flush=True)
        else:
            archive = download_archive(
                SESQL_URL, output / "downloads" / "SeSQL.zip", 8170525, SESQL_SHA256,
            )
            install_archive(archive, sesql, "SeSQL", validate_sesql, SESQL_URL)
        print("SeSQL supplies JSON table contents, not SQLite databases; conversion is a later step.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--only", choices=("all", "model", "data"), default="all")
    parser.add_argument("--model-source", choices=("modelscope", "huggingface"), default="modelscope")
    parser.add_argument("--revision", help="Model branch, tag or commit; default: master (ModelScope) / main (HF).")
    parser.add_argument("--include-sesql", action="store_true", help="Also download the optional SeSQL archive.")
    parser.add_argument("--cspider-archive", type=Path, help="Use a manually uploaded official full_CSpider.zip.")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    try:
        if args.only in ("all", "data"):
            archive = args.cspider_archive.expanduser().resolve() if args.cspider_archive else None
            download_data(output, archive, args.include_sesql)
        if args.only in ("all", "model"):
            revision = args.revision or ("master" if args.model_source == "modelscope" else "main")
            download_model(output, args.model_source, revision)
    except ImportError as error:
        print(f"Missing download dependency: {error}. Install requirements-download.txt.", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Download/validation failed: {error}", file=sys.stderr)
        print("Rerun the same command to resume. Existing validated datasets are not overwritten.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
