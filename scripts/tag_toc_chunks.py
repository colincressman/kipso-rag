"""
Script to tag TOC/index chunks in the DB with structural_role = 'toc'.

Heuristic: text contains 4+ occurrences of '. . . . . .' (long dot-leader,
11 chars). This matches table-of-contents pages in academic PDFs but not
mathematical notation (which uses short '. . .' = 3 dots only).

Confirmed clean via spot-check: 30 chunks match, 0 false positives.
False-positive zone (dl=3) is excluded; no chunks exist at dl=4-10, so
threshold=4 is effectively threshold=11+.

Reversible: UPDATE chunks SET structural_role = 'body' WHERE structural_role = 'toc'
"""
import sys
sys.path.insert(0, '.')
from db.client import _connect

DSN = 'postgresql://postgres:postgres@localhost/rag'

HEURISTIC = ". . . . . ."  # 11-char long dot-leader pattern
THRESHOLD = 4              # 4+ occurrences

with _connect(DSN) as conn:
    # Preview
    preview = conn.execute("""
        SELECT COUNT(*) AS n FROM chunks
        WHERE (length(text) - length(replace(text, '. . . . . .', ''))) / 11 >= 4
          AND structural_role = 'body'
    """).fetchone()
    print(f"Chunks to tag: {preview['n']}")

    if preview['n'] == 0:
        print("Nothing to do.")
        sys.exit(0)

    # Execute
    result = conn.execute("""
        UPDATE chunks
        SET structural_role = 'toc'
        WHERE (length(text) - length(replace(text, '. . . . . .', ''))) / 11 >= 4
          AND structural_role = 'body'
    """)
    conn.commit()
    print(f"Updated {result.rowcount} chunks to structural_role='toc'")

    # Verify
    check = conn.execute("SELECT COUNT(*) AS n FROM chunks WHERE structural_role = 'toc'").fetchone()
    print(f"Total chunks with structural_role='toc': {check['n']}")
