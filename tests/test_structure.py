import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.structure.enrich import enrich_document, enrich_markdown
from pipeline.structure.md_parser import parse_markdown


def _sample_markdown() -> str:
    return """---
doc_id: abc123
filename: spec.md
title: Power Unit Spec
---

<!-- page:1 -->
# Power Unit Spec
Overview paragraph.

## 1 Scope
The unit shall operate at 24V.

### 1.1 Inputs
- Input A
- Input B

<!-- page:2 -->
## 2 Requirements
| Req | Value |
| --- | --- |
| Voltage | 24V |
"""


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------

def test_parse_markdown_frontmatter_and_sections():
    parsed = parse_markdown(_sample_markdown(), source_path="x.md")
    assert parsed.metadata["doc_id"] == "abc123"
    assert parsed.metadata["filename"] == "spec.md"
    assert len(parsed.sections) == 4
    assert parsed.sections[0].title == "Power Unit Spec"
    assert parsed.sections[1].title == "1 Scope"


def test_parse_markdown_no_frontmatter():
    md = "# Simple Title\n\nSome body text.\n"
    parsed = parse_markdown(md)
    assert parsed.metadata == {}
    assert len(parsed.sections) == 1
    assert parsed.sections[0].title == "Simple Title"


def test_parse_markdown_frontmatter_values_stripped():
    md = "---\ntitle:   My Doc   \nauthor: Alice\n---\n# Heading\n"
    parsed = parse_markdown(md)
    assert parsed.metadata["title"] == "My Doc"
    assert parsed.metadata["author"] == "Alice"


# ---------------------------------------------------------------------------
# Section hierarchy and parent linking
# ---------------------------------------------------------------------------

def test_parse_markdown_hierarchy_and_paths():
    parsed = parse_markdown(_sample_markdown())
    sec_scope = parsed.sections[1]
    sec_inputs = parsed.sections[2]
    assert sec_inputs.parent_id == sec_scope.section_id
    assert sec_inputs.path == ["Power Unit Spec", "1 Scope", "1.1 Inputs"]


def test_top_level_section_has_no_parent():
    parsed = parse_markdown(_sample_markdown())
    assert parsed.sections[0].parent_id is None


def test_h2_parent_is_h1():
    md = "# Top\n\n## Sub\n\nContent."
    parsed = parse_markdown(md)
    h1 = parsed.sections[0]
    h2 = parsed.sections[1]
    assert h2.parent_id == h1.section_id


def test_sibling_h2s_both_parent_to_h1():
    md = "# Root\n\n## Alpha\n\nA.\n\n## Beta\n\nB."
    parsed = parse_markdown(md)
    assert parsed.sections[1].parent_id == parsed.sections[0].section_id
    assert parsed.sections[2].parent_id == parsed.sections[0].section_id


def test_level_skip_h1_to_h3():
    """Level skips (h1 → h3 with no h2) should still set parent correctly."""
    md = "# Top\n\n### Deep\n\nContent."
    parsed = parse_markdown(md)
    # Deep's path should include Top
    assert "Top" in parsed.sections[1].path


# ---------------------------------------------------------------------------
# Page markers
# ---------------------------------------------------------------------------

def test_parse_markdown_page_markers():
    parsed = parse_markdown(_sample_markdown())
    requirements = parsed.sections[3]
    assert requirements.page_start == 2
    assert requirements.page_end == 2


def test_section_spanning_multiple_pages():
    # page_start is set from current_page at heading-creation time, so the
    # page marker must appear BEFORE the heading for page_start to be non-None.
    md = "<!-- page:1 -->\n# Long Chapter\n\nFirst page content.\n\n<!-- page:2 -->\nSecond page.\n\n<!-- page:3 -->\nThird page."
    parsed = parse_markdown(md)
    chapter = parsed.sections[0]
    assert chapter.page_start == 1
    assert chapter.page_end == 3


def test_section_before_any_page_marker_has_none_page():
    md = "# Title\n\nContent with no page marker."
    parsed = parse_markdown(md)
    assert parsed.sections[0].page_start is None


# ---------------------------------------------------------------------------
# Content assignment and table detection
# ---------------------------------------------------------------------------

def test_section_content_assigned():
    md = "# Title\n\nThis is the body."
    parsed = parse_markdown(md)
    assert "This is the body." in parsed.sections[0].content


def test_has_table_flag_set_for_markdown_table():
    parsed = parse_markdown(_sample_markdown())
    requirements = parsed.sections[3]
    assert requirements.has_table is True


def test_has_table_false_for_plain_section():
    parsed = parse_markdown(_sample_markdown())
    scope = parsed.sections[1]
    assert scope.has_table is False


# ---------------------------------------------------------------------------
# Source path and preamble
# ---------------------------------------------------------------------------

def test_source_path_preserved():
    parsed = parse_markdown("# Hi\n", source_path="/docs/foo.md")
    assert parsed.source_path == "/docs/foo.md"


def test_content_before_first_heading_is_preamble():
    md = "Some intro text.\n\n# First Section\n\nBody."
    parsed = parse_markdown(md)
    assert "Some intro text" in parsed.preamble
    assert len(parsed.sections) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_input_returns_no_sections():
    parsed = parse_markdown("")
    assert parsed.sections == []


def test_only_frontmatter_no_headings():
    md = "---\ntitle: Empty\n---\n\nJust preamble text."
    parsed = parse_markdown(md)
    assert parsed.sections == []
    assert "Just preamble" in parsed.preamble


def test_section_ids_are_unique():
    parsed = parse_markdown(_sample_markdown())
    ids = [s.section_id for s in parsed.sections]
    assert len(ids) == len(set(ids))


def test_crlf_input_handled():
    md = "# Title\r\n\r\nBody text.\r\n"
    parsed = parse_markdown(md)
    assert parsed.sections[0].title == "Title"


def test_enrich_document_stats_and_flags():
    parsed = parse_markdown(_sample_markdown())
    enriched = enrich_document(parsed)
    assert enriched["stats"]["section_count"] == 4
    assert enriched["stats"]["sections_with_tables"] == 1

    sections = enriched["sections"]
    req = sections[3]
    assert req["has_table"] is True
    assert req["word_count"] > 0
    assert "path_text" in req


def test_enrich_markdown_entrypoint():
    enriched = enrich_markdown(_sample_markdown(), source_path="spec.md")
    assert enriched["source_path"] == "spec.md"
    assert len(enriched["sections"]) == 4
