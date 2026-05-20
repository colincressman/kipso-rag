"""Personal AI — main entry point.

Usage
-----
    python main.py serve                     # start the web server (default)
    python main.py serve --host 0.0.0.0      # LAN-accessible
    python main.py serve --port 9000

    python main.py ingest <path>             # ingest a single file
    python main.py ingest <path> --collection my-collection

    python main.py query "<question>"        # one-shot CLI query
    python main.py query "<question>" --json # machine-readable output

    python main.py status                    # GPU / model status report
    python main.py status --json
    python main.py status --discover         # measure VRAM for all models

Future
------
    python main.py ui                        # launch native GUI (not yet implemented)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of how this is invoked
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def _install_sigterm_handler() -> None:
    """Register a SIGTERM handler that re-raises as KeyboardInterrupt.

    On POSIX systems uvicorn already handles SIGTERM correctly.  On Windows
    (where SIGTERM is not normally sent) this ensures that NSSM or other
    process managers that send SIGTERM still trigger a clean uvicorn shutdown
    via the standard KeyboardInterrupt path.
    """
    import signal

    def _handler(signum, frame):  # noqa: ANN001, ARG001
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _handler)
    except (OSError, ValueError):
        # May fail in sub-interpreter contexts or if already handling SIGTERM.
        pass


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI web server."""
    import os
    import threading
    import webbrowser
    import uvicorn
    from utils.frozen import is_frozen, get_install_dir

    _install_sigterm_handler()

    if is_frozen():
        # In a bundled exe, make all relative paths (db, data, configs) resolve
        # beside the exe rather than wherever the process was launched from.
        install_dir = get_install_dir()
        os.chdir(install_dir)
        # Ensure writable directories exist on first run.
        for subdir in ("data/db", "data/diagnostics", "data/feedback", "data/metadata"):
            (install_dir / subdir).mkdir(parents=True, exist_ok=True)
        # Hot-reload is incompatible with the frozen bundle.
        args.reload = False

    url = f"http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}"
    if is_frozen():
        # Open the browser a moment after the server is ready.
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        "server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a file into the RAG corpus."""
    from services.rag import ingest_file
    result = ingest_file(
        args.path,
        collection_id=args.collection or None,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = result.get("status", "?")
        chunks = result.get("chunk_count", "?")
        doc_id = result.get("doc_id", "?")
        print(f"[{status}] {args.path}")
        print(f"  doc_id      : {doc_id}")
        print(f"  chunk_count : {chunks}")
        if result.get("collection_id"):
            print(f"  collection  : {result['collection_id']}")


def cmd_query(args: argparse.Namespace) -> None:
    """Run a one-shot RAG query and print the answer."""
    from services.rag import retrieve
    from services.llm import answer

    retrieval = retrieve(args.question, top_k=args.top_k)
    response = answer(args.question, retrieval)

    if args.json:
        print(json.dumps(response, indent=2, default=str))
    else:
        print("\n" + response.get("answer", "(no answer)"))
        citations = response.get("citations") or []
        if citations:
            print("\nSources:")
            for c in citations:
                title = c.get("document_title") or c.get("source_name") or c.get("chunk_id", "")
                page = c.get("page_start")
                print(f"  - {title}" + (f" (p. {page})" if page else ""))


def cmd_status(args: argparse.Namespace) -> None:
    """Show GPU / model status and VRAM feasibility."""
    from utils.gpu_detect import main as _gpu_main
    # gpu_detect.main() reads sys.argv — reconstruct the relevant flags
    _argv = ["gpu_detect"]
    if args.json:
        _argv.append("--json")
    if args.discover:
        _argv.append("--discover")
    sys.argv = _argv
    _gpu_main()


def cmd_ui(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Launch the Tkinter launcher GUI (default when no subcommand given)."""
    from launcher.app import main as _launcher_main
    _launcher_main()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Personal AI — RAG + LLM system",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # serve
    p_serve = sub.add_parser("serve", help="Start the web server")
    p_serve.add_argument("--host", default="127.0.0.1",
                         help="Bind address (use 0.0.0.0 for LAN access)")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true",
                         help="Enable hot-reload (development only)")
    p_serve.set_defaults(func=cmd_serve)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a document into the corpus")
    p_ingest.add_argument("path", help="Path to the file to ingest")
    p_ingest.add_argument("--collection", default="",
                          help="Assign to this collection ID")
    p_ingest.add_argument("--json", action="store_true",
                          help="Output machine-readable JSON")
    p_ingest.set_defaults(func=cmd_ingest)

    # query
    p_query = sub.add_parser("query", help="One-shot RAG query")
    p_query.add_argument("question", help="The question to ask")
    p_query.add_argument("--top-k", type=int, default=5, dest="top_k")
    p_query.add_argument("--json", action="store_true",
                         help="Output machine-readable JSON")
    p_query.set_defaults(func=cmd_query)

    # status
    p_status = sub.add_parser("status", help="GPU / model status and VRAM feasibility")
    p_status.add_argument("--json", action="store_true",
                          help="Output machine-readable JSON")
    p_status.add_argument("--discover", action="store_true",
                          help="Load each model, measure VRAM, save to models.yaml")
    p_status.set_defaults(func=cmd_status)

    # ui  (default when double-clicked or run with no args)
    p_ui = sub.add_parser("ui", help="Launch the GUI launcher (default)")
    p_ui.set_defaults(func=cmd_ui)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        # Default: show the GUI launcher
        args = parser.parse_args(["ui"])

    args.func(args)


if __name__ == "__main__":
    main()
