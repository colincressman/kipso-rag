"""
Tests for pipeline extraction utilities — page-type-aware extraction pipeline.

Test strategy
-------------
Unit tests only: pure functions + model_manager mutual exclusion (fully mocked,
                 no real PDFs, no real models, no DB).

Note: TestIngestPdfV2Integration (pipeline.ingest_full) was removed in
session 66 when ingest_full.py was deleted. All new ingest callers use
pipeline.ingest_v3.ingest_file() instead.
"""

from __future__ import annotations

import gc
import importlib
import json
import sys
import threading
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── ensure repo root on sys.path ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _reset_model_manager() -> None:
    """Reset all module-level state in model_manager between tests."""
    import pipeline.extract.model_manager as mm
    with mm._lock:
        mm._active_model   = None
        mm._marker_models  = None
        mm._pix2tex_model  = None


# ═══════════════════════════════════════════════════════════════════════════════
# 1. assemble() — pure function, no mocks needed
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssemble:
    from pipeline.normalize.to_markdown2 import assemble

    def test_pages_joined_in_order(self):
        from pipeline.normalize.to_markdown2 import assemble
        result = assemble({2: "page two", 0: "page zero", 1: "page one"})
        assert result == "page zero\n\npage one\n\npage two"

    def test_empty_pages_skipped(self):
        from pipeline.normalize.to_markdown2 import assemble
        result = assemble({0: "first", 1: "   ", 2: "third"})
        assert result == "first\n\nthird"

    def test_all_empty_returns_empty_string(self):
        from pipeline.normalize.to_markdown2 import assemble
        assert assemble({0: "", 1: "  \n "}) == ""

    def test_single_page(self):
        from pipeline.normalize.to_markdown2 import assemble
        assert assemble({0: "  hello  "}) == "hello"

    def test_empty_dict(self):
        from pipeline.normalize.to_markdown2 import assemble
        assert assemble({}) == ""


# ═══════════════════════════════════════════════════════════════════════════════
# 2. _table_to_markdown() — pure function
# ═══════════════════════════════════════════════════════════════════════════════

class TestTableToMarkdown:
    from pipeline.extract.table_extractor import _table_to_markdown

    def test_basic_two_row_table(self):
        from pipeline.extract.table_extractor import _table_to_markdown
        rows = [["Name", "Value"], ["Alpha", "1"], ["Beta", "2"]]
        md = _table_to_markdown(rows)
        lines = md.splitlines()
        assert lines[0] == "| Name | Value |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| Alpha | 1 |"
        assert lines[3] == "| Beta | 2 |"

    def test_empty_input(self):
        from pipeline.extract.table_extractor import _table_to_markdown
        assert _table_to_markdown([]) == ""

    def test_none_cells_normalised(self):
        from pipeline.extract.table_extractor import _table_to_markdown
        rows = [["A", None], ["B", None]]
        md = _table_to_markdown(rows)
        assert "None" not in md

    def test_ragged_rows_padded(self):
        from pipeline.extract.table_extractor import _table_to_markdown
        rows = [["H1", "H2", "H3"], ["only one"]]
        md = _table_to_markdown(rows)
        data_line = md.splitlines()[2]
        # Should have 3 columns even on the short row
        assert data_line.count("|") == 4  # leading + 3 separators + trailing

    def test_whitespace_normalised_in_cells(self):
        from pipeline.extract.table_extractor import _table_to_markdown
        rows = [["  col  \t1  ", "col2"]]
        md = _table_to_markdown(rows)
        assert "col  \t1" not in md
        assert "col 1" in md


# ═══════════════════════════════════════════════════════════════════════════════
# 3. _is_math_char() — pure function
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsMathChar:
    def test_greek_letter_is_math(self):
        from pipeline.extract.page_classifier import _is_math_char
        assert _is_math_char(ord("α"))   # U+03B1
        assert _is_math_char(ord("Ω"))   # U+03A9

    def test_math_operator_is_math(self):
        from pipeline.extract.page_classifier import _is_math_char
        assert _is_math_char(0x2200)     # ∀ FOR ALL
        assert _is_math_char(0x222B)     # ∫ INTEGRAL

    def test_ascii_letter_not_math(self):
        from pipeline.extract.page_classifier import _is_math_char
        for ch in "ABCDEFGabcdefg0123456789":
            assert not _is_math_char(ord(ch)), f"{ch!r} should not be math"

    def test_pua_is_math(self):
        from pipeline.extract.page_classifier import _is_math_char
        assert _is_math_char(0xE001)     # Private Use Area

    def test_superscript_is_math(self):
        from pipeline.extract.page_classifier import _is_math_char
        assert _is_math_char(0x2070)     # ⁰ superscript zero


# ═══════════════════════════════════════════════════════════════════════════════
# 4. model_manager — mutual exclusion (all imports mocked)
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelManager:
    """
    Marker and pix2tex must never be resident at the same time.
    All actual model imports are patched out — no GPU, no disk access.
    """

    def setup_method(self):
        _reset_model_manager()

    def teardown_method(self):
        _reset_model_manager()

    def test_get_marker_loads_marker(self):
        import pipeline.extract.model_manager as mm
        fake_models = object()
        with patch("pipeline.extract.model_manager._free_cuda_cache"):
            with patch("builtins.__import__", side_effect=_make_import_interceptor(
                "marker.models", "load_all_models", fake_models
            )):
                result = mm.get_marker()
        assert result is fake_models
        assert mm._active_model == "marker"

    def test_get_marker_twice_returns_same(self):
        import pipeline.extract.model_manager as mm
        fake_models = object()
        with patch("pipeline.extract.model_manager._free_cuda_cache"):
            with patch("builtins.__import__", side_effect=_make_import_interceptor(
                "marker.models", "load_all_models", fake_models
            )):
                first  = mm.get_marker()
                second = mm.get_marker()
        assert first is second
        assert mm._active_model == "marker"

    def test_get_pix2tex_after_marker_unloads_marker(self):
        import pipeline.extract.model_manager as mm
        fake_marker  = object()
        fake_pix2tex = MagicMock()

        with patch("pipeline.extract.model_manager._free_cuda_cache"):
            with patch("builtins.__import__", side_effect=_make_import_interceptor(
                "marker.models", "load_all_models", fake_marker
            )):
                mm.get_marker()

        assert mm._active_model == "marker"
        assert mm._marker_models is not None

        with patch("pipeline.extract.model_manager._free_cuda_cache"):
            with patch("builtins.__import__", side_effect=_make_import_interceptor(
                "pix2tex.cli", "LatexOCR", fake_pix2tex
            )):
                mm.get_pix2tex()

        assert mm._active_model   == "pix2tex"
        assert mm._marker_models  is None   # unloaded
        assert mm._pix2tex_model  is not None

    def test_get_marker_after_pix2tex_unloads_pix2tex(self):
        import pipeline.extract.model_manager as mm
        fake_marker  = object()
        fake_pix2tex = MagicMock()

        with patch("pipeline.extract.model_manager._free_cuda_cache"):
            with patch("builtins.__import__", side_effect=_make_import_interceptor(
                "pix2tex.cli", "LatexOCR", fake_pix2tex
            )):
                mm.get_pix2tex()

        assert mm._active_model == "pix2tex"

        with patch("pipeline.extract.model_manager._free_cuda_cache"):
            with patch("builtins.__import__", side_effect=_make_import_interceptor(
                "marker.models", "load_all_models", fake_marker
            )):
                mm.get_marker()

        assert mm._active_model   == "marker"
        assert mm._pix2tex_model  is None   # unloaded

    def test_unload_all_clears_state(self):
        import pipeline.extract.model_manager as mm
        fake_marker = object()
        with patch("pipeline.extract.model_manager._free_cuda_cache"):
            with patch("builtins.__import__", side_effect=_make_import_interceptor(
                "marker.models", "load_all_models", fake_marker
            )):
                mm.get_marker()

        mm.unload_all()
        assert mm._active_model  is None
        assert mm._marker_models is None
        assert mm._pix2tex_model is None

    def test_pix2tex_missing_raises_import_error(self):
        import pipeline.extract.model_manager as mm
        _reset_model_manager()

        def _raise_on_pix2tex(name, *args, **kwargs):
            if "pix2tex" in name:
                raise ImportError("pix2tex not installed")
            return original_import(name, *args, **kwargs)

        original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

        with patch("builtins.__import__", side_effect=_raise_on_pix2tex):
            with pytest.raises(ImportError, match="pix2tex"):
                mm.get_pix2tex()

    def test_mutual_exclusion_threaded(self):
        """
        Two threads calling get_marker / get_pix2tex concurrently must
        not leave both models loaded at the same time.
        """
        import pipeline.extract.model_manager as mm

        errors: list[Exception] = []
        results: list[str] = []

        fake_marker  = object()
        fake_pix2tex = MagicMock()

        def load_marker():
            try:
                with patch("pipeline.extract.model_manager._free_cuda_cache"):
                    with patch("builtins.__import__", side_effect=_make_import_interceptor(
                        "marker.models", "load_all_models", fake_marker
                    )):
                        mm.get_marker()
                results.append("marker")
            except Exception as exc:
                errors.append(exc)

        def load_pix2tex():
            try:
                with patch("pipeline.extract.model_manager._free_cuda_cache"):
                    with patch("builtins.__import__", side_effect=_make_import_interceptor(
                        "pix2tex.cli", "LatexOCR", fake_pix2tex
                    )):
                        mm.get_pix2tex()
                results.append("pix2tex")
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=load_marker)
        t2 = threading.Thread(target=load_pix2tex)
        t1.start(); t2.start()
        t1.join();  t2.join()

        assert not errors, f"Thread errors: {errors}"
        # Only one model should be active after both threads finish
        assert mm._active_model in ("marker", "pix2tex")
        both_loaded = (mm._marker_models is not None) and (mm._pix2tex_model is not None)
        assert not both_loaded, "Both models were resident simultaneously"


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _make_import_interceptor(module_name: str, attr_name: str, return_value: Any):
    """
    Return a ``__import__`` side_effect that intercepts one module and makes
    its named attribute return (or be) *return_value*, passing everything else
    to the real importer.

    ``return_value`` is used in two ways:
    - If the attr is a class (e.g. ``LatexOCR``), we make it a callable that
      returns ``return_value()`` i.e. an instance mock.
    - If the attr is a function (e.g. ``load_all_models``), it is called with
      no arguments and must return the fake models.
    """
    original = __import__

    def _intercept(name, *args, **kwargs):
        if name == module_name:
            fake_mod = types.ModuleType(module_name)

            if callable(return_value):
                # It's a class or callable — wrap so LatexOCR() works
                setattr(fake_mod, attr_name, lambda *a, **kw: return_value)
            else:
                # It's a plain value — wrap in a zero-arg callable
                setattr(fake_mod, attr_name, lambda: return_value)

            # Also register parent packages so dotted imports resolve
            parts = module_name.split(".")
            for i in range(1, len(parts)):
                parent_name = ".".join(parts[:i])
                if parent_name not in sys.modules:
                    sys.modules[parent_name] = types.ModuleType(parent_name)
            sys.modules[module_name] = fake_mod
            return fake_mod
        return original(name, *args, **kwargs)

    return _intercept
