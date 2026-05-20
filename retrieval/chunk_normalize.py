"""Normalize PDF-extraction noise in chunk text before it reaches the LLM.

Fixes noisy PDF extraction artifacts:
  "$1 , 000"   -> "$1,000"
  "0 . 6468"   -> "0.6468"
  "[7 , 930]"  -> "(7,930)"  (numeric-only brackets -> parens)

Also normalizes pseudocode / math notation:
  Unicode arrows / assignment operators -> ASCII
  Greek letters -> spelled-out names (so LLM can read them)
  Lone quote-chars in math context (PDF epsilon artifact) -> epsilon
  Spaced function-call notation: "Q ( a )" -> "Q(a)"
"""

from __future__ import annotations

import re
from typing import Dict

_SPACED_NUM_COMMA_RE = re.compile(r"(\d)\s*,\s*(\d)")
_SPACED_DECIMAL_RE   = re.compile(r"(\d)\s+\.\s+(\d)")
# Only replace brackets whose entire contents are digits/spaces/commas/dots.
_NUMERIC_BRACKET_RE  = re.compile(r"\[(\s*\d[\d\s,.]*)\]")

# Unicode arrows / operators that appear in algorithm pseudocode
_UNICODE_ARROW_RE = re.compile(r"[⇢←→⟵⟶↑↓⟸⟹⇐⇒➤➜]")
# Unicode minus / en-dash / em-dash in math expressions
_UNICODE_MINUS_RE = re.compile(r"[−–—]")
# Math bracket symbols used in pseudocode — strip them.
# Includes confirmed PDF artifact chars U+21E4/U+21E5 (⇤⇥) and common others.
_MATH_BRACKET_RE  = re.compile(r"[⇤⇥⟤⟥⌈⌉⌊⌋⟦⟧〈〉⟨⟩]")
# Greek letters that appear in ML/RL pseudocode (PDF may preserve Unicode)
_GREEK_MAP: Dict[str, str] = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "ε": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta",
    "λ": "lambda", "μ": "mu", "π": "pi", "σ": "sigma",
    "τ": "tau", "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
    "Σ": "Sigma", "Π": "Pi", "Δ": "Delta", "Θ": "Theta", "Λ": "Lambda",
}
# Spaced function-call notation from PDF extraction: "Q ( a )" -> "Q(a)"
_SPACED_FUNC_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s+\(\s*([^)]{0,30}?)\s*\)")
# Straight quote " (\u0022) used as epsilon substitution in PDF font encoding.
# Verified in DB: the Sutton & Barto bandit chunk stores epsilon as \u0022.
# Pattern: quote preceded by math operator/space AND followed by whitespace.
# The (?=\s) lookahead prevents matching prose opening quotes like "epsilon-greedy".
_LONE_QUOTE_AS_EPSILON_RE = re.compile(r'(?<=[\s,\-+*\/=<>])"(?=[\s(])')


def normalize_chunk_text(text: str) -> str:
    """Normalize PDF-extraction noise in chunk text before it reaches the LLM."""
    if not text:
        return text
    # Numeric formatting
    text = _SPACED_NUM_COMMA_RE.sub(r"\1,\2", text)
    text = _SPACED_DECIMAL_RE.sub(r"\1.\2", text)
    text = _NUMERIC_BRACKET_RE.sub(r"(\1)", text)
    # Math / pseudocode normalization
    text = _UNICODE_ARROW_RE.sub("->", text)
    text = _UNICODE_MINUS_RE.sub("-", text)
    text = _MATH_BRACKET_RE.sub("", text)
    for sym, name in _GREEK_MAP.items():
        text = text.replace(sym, name)
    # Collapse spaced function-call notation first (before epsilon sub, so
    # "epsilon (prose...)" is not collapsed into a fake function call)
    text = _SPACED_FUNC_RE.sub(r"\1(\2)", text)
    # Lone quote used as epsilon PDF artifact — run after spaced-func.
    # Lookahead (?=[\s(]) is zero-width, so original space is preserved.
    text = _LONE_QUOTE_AS_EPSILON_RE.sub("epsilon", text)
    return text
