#!/usr/bin/env python3
"""Freeze prepared and special examples without crossing schema boundaries."""

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from common import load_jsonl, safe_path, sha256_file, write_json, write_jsonl


def build(assets, prepared, special, output):
    assets, output = assets.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Frozen data must have a new directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    split_schemas, seen_ids, database_hashes = {}, set(), {}
    manifest = {"inputs": {}, "splits": {}, "databases": database_hashes, "data_sha256": {}}
    with tempfile.TemporaryDirectory(prefix=".freeze-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for split in ("train", "dev", "test"):
            rows = []
            for directory in (prepared, special):
                if directory is None:
                    continue
                path = directory / f"{split}.jsonl"
                manifest["inputs"][str(path.resolve())] = sha256_file(path)
                for row in load_jsonl(path):
                    if row["split"] != split or row["id"] in seen_ids:
                        raise ValueError(f"Split mismatch or repeated ID: {row['id']}")
                    seen_ids.add(row["id"])
                    owner = split_schemas.setdefault(row["schema_id"], split)
                    if owner != split:
                        raise ValueError(f"Schema leakage between {owner} and {split}: {row['id']}")
                    relative = row["db_path"]
                    if relative not in database_hashes:
                        database_hashes[relative] = sha256_file(safe_path(assets, relative))
                    if row["expected"]["action"] not in ("sql", "clarify", "refuse"):
                        raise ValueError(f"Invalid target action: {row['id']}")
                    rows.append(row)
            if not rows:
                raise ValueError(f"Empty {split} split")
            path = staging / f"{split}.jsonl"
            write_jsonl(path, rows)
            manifest["data_sha256"][path.name] = sha256_file(path)
            manifest["splits"][split] = {"rows": len(rows), "actions": dict(Counter(r["expected"]["action"] for r in rows)),
                                          "sources": dict(Counter(r["source"] for r in rows))}
        write_json(staging / "manifest.json", manifest)
        staging.rename(output)
    print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--special-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    build(args.assets_dir, args.prepared_dir or args.assets_dir / "data/prepared",
          args.special_dir, args.output_dir or args.assets_dir / "data/sft-v1")


if __name__ == "__main__":
    main()
