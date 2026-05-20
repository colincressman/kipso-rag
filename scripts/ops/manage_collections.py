#!/usr/bin/env python3
"""
Collection management CLI.

Collections support a two-level hierarchy: a parent collection acts as a
folder, sub-collections hold the actual documents. Querying a parent
automatically searches all its sub-collections.

Usage examples
--------------
# List all collections (tree view):
    python scripts/collections.py list

# Create a top-level parent collection (a folder):
    python scripts/collections.py create CS7646 "Machine Learning for Trading"

# Create sub-collections inside it:
    python scripts/collections.py create CS7646/notes "Lecture Notes" --parent CS7646
    python scripts/collections.py create CS7646/books "Textbooks" --parent CS7646
    python scripts/collections.py create CS7646/projects "Projects" --parent CS7646

# Ingest a file into a sub-collection:
    python scripts/ingest_one.py notes/lecture1.md --collection CS7646/notes

# Query scoped to the whole class (all sub-collections):
    python scripts/query_cli.py "how does Q-learning work?" --collection CS7646 --answer

# Query scoped to just one sub-collection:
    python scripts/query_cli.py "equity curve" --collection CS7646/projects --answer

# Show documents in a collection:
    python scripts/collections.py show CS7646
    python scripts/collections.py show CS7646/notes

# Assign existing documents to a collection (by doc_id):
    python scripts/collections.py assign CS7646/books --doc-id <doc_id>

# Delete a collection (chunks are un-assigned, not deleted):
    python scripts/collections.py delete CS7646/notes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.client import (
    assign_to_collection,
    create_collection,
    delete_collection,
    get_collection,
    list_collections,
    list_unassigned_documents,
    unassign_from_collection,
)
from utils.runtime_defaults import DEFAULT_DB_DSN


def cmd_list(args: argparse.Namespace) -> int:
    cols = list_collections(args.db)
    if not cols:
        print("No collections defined yet.")
        print("Create one with:  python scripts/collections.py create <id> <name>")
        return 0

    # Separate top-level from sub-collections for tree rendering
    parents = [c for c in cols if not c["parent_id"]]
    children_by_parent: dict = {}
    for c in cols:
        if c["parent_id"]:
            children_by_parent.setdefault(c["parent_id"], []).append(c)

    print(f"{'Collection':<32} {'Docs':>5} {'Chunks':>7}  Description")
    print("-" * 72)
    for p in parents:
        desc = (p["description"] or "")[:30]
        print(f"{p['collection_id']:<32} {p['doc_count']:>5} {p['chunk_count']:>7}  {desc}")
        for child in children_by_parent.get(p["collection_id"], []):
            cdesc = (child["description"] or "")[:28]
            cid = f"  └─ {child['collection_id']}"
            print(f"{cid:<32} {child['doc_count']:>5} {child['chunk_count']:>7}  {cdesc}")
    # Any collections whose parent doesn't exist (orphaned)
    all_parent_ids = {p["collection_id"] for p in parents}
    orphans = [c for c in cols if c["parent_id"] and c["parent_id"] not in all_parent_ids]
    for o in orphans:
        print(f"  [orphan] {o['collection_id']:<24} {o['doc_count']:>5} {o['chunk_count']:>7}")
    return 0


def cmd_create(args: argparse.Namespace) -> int:
    try:
        create_collection(
            args.db, args.collection_id, args.name,
            description=args.description,
            parent_id=args.parent,
        )
        if args.parent:
            print(f"Created sub-collection '{args.collection_id}' under '{args.parent}' — {args.name}")
        else:
            print(f"Created collection '{args.collection_id}' — {args.name}")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    info = get_collection(args.db, args.collection_id)
    if info is None:
        print(f"Collection '{args.collection_id}' not found.", file=sys.stderr)
        return 1
    print(f"Collection: {info['collection_id']}  ({info['name']})")
    if info["parent_id"]:
        print(f"Parent:      {info['parent_id']}")
    if info["description"]:
        print(f"Description: {info['description']}")
    print(f"Created:     {info['created_at']}")
    subs = info.get("sub_collections", [])
    if subs:
        print(f"\nSub-collections ({len(subs)}):")
        for s in subs:
            print(f"  {s['collection_id']}  ({s['name']})")
    docs = info["documents"]
    if not docs:
        print("\nNo documents assigned yet.")
    else:
        print(f"\n{'doc_id':<14} {'source_type':<12} {'chunks':>6}  filename")
        print("-" * 70)
        for d in docs:
            print(f"{d['doc_id'][:12]:<14} {d['source_type']:<12} {d['chunk_count']:>6}  {d['filename']}")
    return 0


def cmd_assign(args: argparse.Namespace) -> int:
    if not args.doc_id and not args.source_type:
        print("ERROR: Provide at least one of --doc-id or --source-type", file=sys.stderr)
        return 1
    updated = assign_to_collection(
        args.db,
        args.collection_id,
        doc_ids=list(args.doc_id) if args.doc_id else None,
        source_type=args.source_type,
    )
    print(f"Assigned — collection '{args.collection_id}' now has {updated} chunk(s).")
    return 0


def cmd_unassign(args: argparse.Namespace) -> int:
    if not args.doc_id:
        print("ERROR: Provide at least one --doc-id", file=sys.stderr)
        return 1
    cleared = unassign_from_collection(args.db, list(args.doc_id))
    print(f"Cleared collection_id for {cleared} chunk(s).")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    info = get_collection(args.db, args.collection_id)
    if info is None:
        print(f"Collection '{args.collection_id}' not found.", file=sys.stderr)
        return 1
    doc_count = len(info["documents"])
    if doc_count > 0 and not args.yes:
        print(f"Collection '{args.collection_id}' has {doc_count} document(s).")
        print("Chunks will be un-assigned (not deleted). Continue? [y/N] ", end="", flush=True)
        answer = input().strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return 0
    cleared = delete_collection(args.db, args.collection_id, clear_chunks=True)
    print(f"Deleted collection '{args.collection_id}'. {cleared} chunk(s) un-assigned.")
    return 0


def cmd_unassigned(args: argparse.Namespace) -> int:
    docs = list_unassigned_documents(args.db)
    if not docs:
        print("All documents are assigned to a collection.")
        return 0
    print(f"{len(docs)} document(s) with no collection:\n")
    print(f"{'doc_id':<14} {'source_type':<12} {'chunks':>6}  filename")
    print("-" * 70)
    for d in docs:
        print(f"{d['doc_id'][:12]:<14} {d['source_type']:<12} {d['chunk_count']:>6}  {d['filename']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage RAG document collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", default=DEFAULT_DB_DSN, help="SQLite DB path")
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List all collections with document and chunk counts")

    # create
    p_create = sub.add_parser("create", help="Create a new collection")
    p_create.add_argument("collection_id", help="Short slug/ID for the collection (e.g. CS7646, CS7646/notes)")
    p_create.add_argument("name", help="Human-readable display name")
    p_create.add_argument("--description", default=None, help="Optional description")
    p_create.add_argument(
        "--parent", default=None, metavar="PARENT_ID",
        help="Make this a sub-collection inside an existing parent (e.g. --parent CS7646)",
    )

    # show
    p_show = sub.add_parser("show", help="Show documents assigned to a collection")
    p_show.add_argument("collection_id", help="Collection ID to inspect")

    # assign
    p_assign = sub.add_parser("assign", help="Assign existing documents to a collection")
    p_assign.add_argument("collection_id", help="Target collection ID")
    p_assign.add_argument(
        "--doc-id", action="append", default=[], metavar="DOC_ID",
        help="Document ID to assign (repeatable). Use scripts/db_status.py to find doc_ids.",
    )
    p_assign.add_argument(
        "--source-type", default=None,
        help="Assign all documents of this source_type (e.g. pdf_book, notes)",
    )

    # unassign
    p_unassign = sub.add_parser("unassign", help="Remove documents from their collection")
    p_unassign.add_argument(
        "--doc-id", action="append", default=[], metavar="DOC_ID",
        help="Document ID to unassign (repeatable)",
    )

    # delete
    p_delete = sub.add_parser("delete", help="Delete a collection (chunks are un-assigned, not deleted)")
    p_delete.add_argument("collection_id", help="Collection ID to delete")
    p_delete.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    # unassigned
    sub.add_parser("unassigned", help="List documents not assigned to any collection")

    args = parser.parse_args()

    dispatch = {
        "list": cmd_list,
        "create": cmd_create,
        "show": cmd_show,
        "assign": cmd_assign,
        "unassign": cmd_unassign,
        "delete": cmd_delete,
        "unassigned": cmd_unassigned,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
