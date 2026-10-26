#!/usr/bin/env python3
"""Clean CSpider, optionally rebuild SeSQL, and retain database-disjoint official splits."""

import argparse
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from sqlglot import exp

from common import (execute_sql, make_record, parse_query, quote_identifier, read_json,
                    safe_path, schema_context, sha256_file, write_json, write_jsonl)


def rebuild_sesql(content, schema, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(destination)
    try:
        expected_tables = schema["table_names_original"]
        if set(content["tables"]) != set(expected_tables):
            raise ValueError("SeSQL content/schema tables disagree.")
        for table_index, table in enumerate(expected_tables):
            source = content["tables"][table]
            columns = [name for index, name in schema["column_names_original"] if index == table_index]
            if columns != source["header"]:
                raise ValueError(f"SeSQL content/schema columns disagree: {table}")
            definitions = ", ".join(
                quote_identifier(column) + (" NUMERIC" if kind == "number" else " TEXT")
                for column, kind in zip(columns, source["type"])
            )
            connection.execute(f"CREATE TABLE {quote_identifier(table)} ({definitions})")
            if any(len(row) != len(columns) for row in source["cell"]):
                raise ValueError(f"SeSQL row length mismatch: {table}")
            # Source values stay intact; SQLite affinity handles numeric strings without coercing IDs in Python.
            connection.executemany(
                f"INSERT INTO {quote_identifier(table)} VALUES ({','.join('?' for _ in columns)})", source["cell"]
            )
        connection.commit()
    finally:
        connection.close()


def prepare(assets, output, include_sesql=False, timeout=2.0):
    assets, output = assets.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory; raw/prepared data will not be overwritten: {output}")
    output.relative_to(assets)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temporary:
        staging = Path(temporary)
        rows = []
        rejected = []
        manifests = {}
        for source in ("cspider", "sesql") if include_sesql else ("cspider",):
            raw = assets / "data/raw" / source
            schemas = {item["db_id"]: item for item in read_json(raw / "tables.json")}
            if source == "cspider":
                schemas.update({item["db_id"]: item for item in read_json(raw / "test_data/tables_test.json")})
            contents = {item["db_id"]: item for item in read_json(raw / "db_content.json")} if source == "sesql" else {}
            cache = {}
            for split in ("train", "dev", "test"):
                relative = f"{split}.json" if source == "cspider" else f"single-round-question-completed/{split}.json"
                if source == "cspider" and split == "test":
                    relative = "test_data/test.json"
                source_file = raw / relative
                manifests[f"{source}/{relative}"] = sha256_file(source_file)
                for index, example in enumerate(read_json(source_file)):
                    identity = f"{source}:{split}:{index}"
                    try:
                        db_id = example["db_id"]
                        database_folder = "test_database" if split == "test" else "database"
                        if source == "cspider":
                            database = safe_path(raw / database_folder, f"{db_id}/{db_id}.sqlite")
                            final_database = database
                        else:
                            local = f"databases/sesql/{db_id}/{db_id}.sqlite"
                            database = safe_path(staging, local)
                            final_database = safe_path(output, local)
                            if not database.exists():
                                rebuild_sesql(contents[db_id], schemas[db_id], database)
                        key = str(database)
                        if key not in cache:
                            cache[key] = schema_context(database, schemas[db_id])
                        context, fingerprint = cache[key]
                        question, sql = example["question"].strip(), example["query"].strip()
                        if not question:
                            raise ValueError("Empty question")
                        tree = parse_query(sql)
                        canonical = tree.sql(dialect="sqlite").lower()
                        if any(word in canonical for word in ("random(", "randomblob(", "'now'", "current_date", "current_timestamp")):
                            raise ValueError("Nondeterministic reference requires manual review.")
                        execute_sql(database, sql, context["allowed_tables"], timeout=timeout)
                        tags = ["field_mapping", "sqlite"]
                        if list(tree.find_all(exp.Join)):
                            tags.append("join")
                        rows.append(make_record(source, split, index, question, sql, context, fingerprint,
                                                final_database.relative_to(assets).as_posix(), tags))
                    except (ValueError, KeyError, IndexError, TypeError, OSError, sqlite3.Error) as error:
                        rejected.append({"id": identity, "reason": str(error)})
                print(f"Checked {source}/{split}: {len(rows)} accepted before deduplication", flush=True)

        # Held-out schemas win; never move their examples into training while mixing sources.
        priority = {"train": 0, "dev": 1, "test": 2}
        owners = {}
        for row in rows:
            fingerprint = row["schema_id"]
            owners[fingerprint] = max(owners.get(fingerprint, 0), priority[row["split"]])
        retained = {name: [] for name in priority}
        seen = set()
        for row in rows:
            if priority[row["split"]] != owners[row["schema_id"]]:
                rejected.append({"id": row["id"], "reason": "schema_overlaps_held_out_split"})
                continue
            normalized = parse_query(row["expected"]["sql"]).sql(dialect="sqlite")
            key = (row["schema_id"], " ".join(row["question"].split()), normalized)
            if key in seen:
                rejected.append({"id": row["id"], "reason": "duplicate_question_sql_within_schema"})
                continue
            seen.add(key)
            retained[row["split"]].append(row)
        if any(not records for records in retained.values()):
            raise ValueError("A cleaned split is empty; inspect the input data.")
        for split, records in retained.items():
            write_jsonl(staging / f"{split}.jsonl", records)
        write_jsonl(staging / "rejected.jsonl", rejected)
        write_json(staging / "manifest.json", {
            "input_sha256": manifests, "counts": {name: len(value) for name, value in retained.items()},
            "rejection_reasons": dict(Counter(row["reason"] for row in rejected)),
            "notes": "Execution checks cannot certify question-SQL semantic correctness. Manual review remains required.",
        })
        staging.rename(output)
    print(f"Prepared data: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--include-sesql", action="store_true")
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    prepare(args.assets_dir, args.output_dir or args.assets_dir / "data/prepared", args.include_sesql, args.timeout)


if __name__ == "__main__":
    main()
