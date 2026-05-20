"""CLI entry point for the large-document extraction pipeline.

Usage examples:

  # Run a saved project by slug
  python -m scripts.ops.run_extraction run --project my-spec-review

  # Run with an inline document (auto-creates a transient project)
  python -m scripts.ops.run_extraction run \
      --doc path/to/doc.pdf \
      --suggest                        # auto-suggest branches via LLM

  # Auto-suggest branches for a document (does not run extraction)
  python -m scripts.ops.run_extraction suggest \
      --doc path/to/doc.pdf \
      --doc-type "procurement specification" \
      --title "System Requirements v2"

  # List saved projects
  python -m scripts.ops.run_extraction projects

  # Show / dump a saved project as JSON
  python -m scripts.ops.run_extraction show --project my-spec-review
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _emit(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Subcommand: projects
# ---------------------------------------------------------------------------

def cmd_projects(args: argparse.Namespace) -> int:
    from extraction.project_runner import list_projects
    from utils.config import load_yaml_config
    cfg = load_yaml_config("configs/extraction.yaml") or {}
    projects_dir = cfg.get("flag_library", {}).get("projects_dir", "data/flag_library/projects")
    projects = list_projects(projects_dir)
    if not projects:
        print("No saved projects found.")
        return 0
    for p in projects:
        print(f"  {p['slug']:<30} {p['name']}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: show
# ---------------------------------------------------------------------------

def cmd_show(args: argparse.Namespace) -> int:
    from extraction.branch_config import ProjectConfig
    from utils.config import load_yaml_config
    cfg = load_yaml_config("configs/extraction.yaml") or {}
    projects_dir = cfg.get("flag_library", {}).get("projects_dir", "data/flag_library/projects")
    try:
        project = ProjectConfig.load(args.project, projects_dir)
    except FileNotFoundError:
        print(f"Project '{args.project}' not found in {projects_dir}", file=sys.stderr)
        return 1
    print(json.dumps(project.to_dict() if hasattr(project, "to_dict") else project.__dict__, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: suggest
# ---------------------------------------------------------------------------

def cmd_suggest(args: argparse.Namespace) -> int:
    from extraction.suggest import suggest_branches
    from extraction.project_runner import _make_llm_fn
    from utils.runtime_defaults import DEFAULT_DB_DSN

    llm_fn = _make_llm_fn()
    branches = suggest_branches(
        llm_fn,
        document_path=args.doc,
        db_dsn=args.db or DEFAULT_DB_DSN,
        collection_id=args.collection,
        doc_type=args.doc_type or "",
        title=args.title or "",
    )
    if not branches:
        print("No branches suggested. Try with --collection or check that the document is ingested.")
        return 0

    print(f"\nSuggested {len(branches)} branch(es):\n")
    for b in branches:
        kws = ", ".join(b.keywords) if b.keywords else "(semantic)"
        print(f"  [{b.mode}] {b.name}")
        if b.keywords:
            print(f"    Keywords: {kws}")
        if b.topic_description:
            print(f"    Topic: {b.topic_description}")
    if args.json:
        print("\nJSON:\n" + json.dumps([b.to_dict() for b in branches], indent=2))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    from extraction.branch_config import DocumentSource, ProjectConfig, BranchConfig
    from extraction.project_runner import run_project, _make_llm_fn
    from extraction.suggest import suggest_branches
    from utils.config import load_yaml_config
    from utils.runtime_defaults import DEFAULT_DB_DSN

    cfg_yaml = load_yaml_config("configs/extraction.yaml") or {}
    projects_dir = cfg_yaml.get("flag_library", {}).get("projects_dir", "data/flag_library/projects")

    if args.project:
        # Load saved project
        try:
            project = ProjectConfig.load(args.project, projects_dir)
        except FileNotFoundError:
            print(f"Project '{args.project}' not found.", file=sys.stderr)
            return 1
    elif args.doc:
        # Build transient project from CLI args
        doc_path = args.doc
        slug = Path(doc_path).stem.replace(" ", "_")[:40].lower()
        sources = [DocumentSource(path=doc_path, role="primary")]

        if args.suggest:
            llm_fn = _make_llm_fn()
            _emit("Auto-suggesting branches…")
            branches = suggest_branches(
                llm_fn,
                document_path=doc_path,
                db_dsn=args.db or DEFAULT_DB_DSN,
                collection_id=args.collection,
                doc_type=args.doc_type or "",
                title=args.title or slug,
            )
            if not branches:
                _emit("No branches suggested; aborting.")
                return 1
            _emit(f"Suggested {len(branches)} branch(es).")
        elif args.branches:
            try:
                raw_branches = json.loads(Path(args.branches).read_text(encoding="utf-8"))
                branches = [BranchConfig.from_dict(b) for b in raw_branches]
            except Exception as exc:
                print(f"Failed to load branches from {args.branches}: {exc}", file=sys.stderr)
                return 1
        else:
            print("Provide --suggest or --branches FILE when using --doc.", file=sys.stderr)
            return 1

        project = ProjectConfig(
            slug=slug,
            name=args.title or Path(doc_path).name,
            document_sources=sources,
            branches=branches,
            collection_id=args.collection or f"extraction_{slug}",
            keep_collection_after_run=args.keep_collection,
            report_output_path=args.output or cfg_yaml.get("report_output_path", "data/extraction_reports"),
        )
    else:
        print("Provide --project SLUG or --doc PATH.", file=sys.stderr)
        return 1

    result = run_project(
        project,
        db_dsn=args.db or DEFAULT_DB_DSN,
        emit=_emit,
    )

    if result.error:
        print(f"\n✗ Extraction failed: {result.error}", file=sys.stderr)
        return 1

    total = sum(len(br.items) for br in result.branch_results)
    print(f"\nExtraction complete — {total} item(s) in {result.elapsed_seconds}s")
    if result.report_path:
        print(f"Report: {result.report_path}")

    if args.json:
        print(json.dumps({
            "slug": result.project_slug,
            "report_path": result.report_path,
            "elapsed_seconds": result.elapsed_seconds,
            "branches": [
                {
                    "name": br.branch_name,
                    "status": br.status,
                    "items": len(br.items),
                }
                for br in result.branch_results
            ],
        }, indent=2))

    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_extraction",
        description="Large-document extraction CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # projects
    sub.add_parser("projects", help="List saved projects")

    # show
    p_show = sub.add_parser("show", help="Show a saved project as JSON")
    p_show.add_argument("--project", required=True, metavar="SLUG")

    # suggest
    p_suggest = sub.add_parser("suggest", help="Auto-suggest branches for a document")
    p_suggest.add_argument("--doc", metavar="PATH", help="Document path")
    p_suggest.add_argument("--collection", metavar="ID", help="Collection ID")
    p_suggest.add_argument("--db", metavar="PATH", help="SQLite DB path")
    p_suggest.add_argument("--doc-type", metavar="TYPE", help="Document type hint")
    p_suggest.add_argument("--title", metavar="TITLE", help="Document title hint")
    p_suggest.add_argument("--json", action="store_true", help="Also print JSON")

    # run
    p_run = sub.add_parser("run", help="Run an extraction project")
    grp = p_run.add_mutually_exclusive_group()
    grp.add_argument("--project", metavar="SLUG", help="Saved project slug to run")
    grp.add_argument("--doc", metavar="PATH", help="Single document to run ad-hoc")
    p_run.add_argument("--suggest", action="store_true", help="Auto-suggest branches (ad-hoc)")
    p_run.add_argument("--branches", metavar="FILE", help="JSON file with branch configs (ad-hoc)")
    p_run.add_argument("--collection", metavar="ID", help="Collection ID (ad-hoc)")
    p_run.add_argument("--db", metavar="PATH", help="SQLite DB path")
    p_run.add_argument("--output", metavar="DIR", help="Report output directory")
    p_run.add_argument("--doc-type", metavar="TYPE")
    p_run.add_argument("--title", metavar="TITLE")
    p_run.add_argument("--keep-collection", action="store_true")
    p_run.add_argument("--json", action="store_true", help="Print JSON summary at end")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    handlers = {
        "projects": cmd_projects,
        "show": cmd_show,
        "suggest": cmd_suggest,
        "run": cmd_run,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    sys.exit(handler(args))


if __name__ == "__main__":
    main()
