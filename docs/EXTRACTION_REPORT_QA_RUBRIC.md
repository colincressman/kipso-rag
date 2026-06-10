# Extraction Report QA Rubric

Use this rubric to grade any Extraction Studio output as a deliverable, not just as a pipeline artifact.

The goal is consistency:
- compare runs against the same baseline
- separate "smart but messy" from "client-ready"
- identify whether failures come from evidence quality, synthesis quality, category organization, or context limits

## Scoring Model

Score each category on a `0-5` scale:

- `5` = excellent / production-ready
- `4` = strong with minor issues
- `3` = usable but clearly flawed
- `2` = weak / unreliable
- `1` = poor
- `0` = failed or missing

Apply the category weights below to produce a weighted score out of `100`.

## Categories

### 1. Evidence Quality (`25%`)

Question:
Does the raw evidence shown in the report feel curated, relevant, and readable?

Look for:
- low noise
- low OCR corruption
- minimal `[ TABLE ]` / pipe garbage / page debris
- limited appendix-only spam
- evidence that actually supports the section it appears in

High score:
- evidence is readable and relevant
- little obvious extraction noise
- duplicate or low-value snippets are rare

Low score:
- evidence looks like a dump
- repeated forms, TOCs, appendix junk, or giant table fragments dominate
- encoding artifacts or pipe-heavy text reduce readability

### 2. Synthesis Quality (`20%`)

Question:
Do the summaries actually understand the material and express it clearly?

Look for:
- coherent summary writing
- correct project framing
- useful abstraction from noisy evidence
- grounded statements instead of generic fluff

High score:
- summaries are concise, useful, and accurate
- major themes are identified correctly
- the model ignores low-value noise well

Low score:
- summaries are vague, repetitive, or hallucinated
- important distinctions are blurred
- the report reads like generic AI prose

### 3. Category Fidelity (`15%`)

Question:
Did the content end up in the right buckets?

Look for:
- procurement content in procurement sections
- controls content in controls sections
- network/fiber content in infrastructure sections
- risks gathered in risk sections rather than scattered everywhere

High score:
- categories are tight and intuitive
- minimal cross-contamination

Low score:
- sections bleed heavily into one another
- the same content appears in many unrelated categories
- important categories are under-filled while others are overloaded

### 4. Completeness (`15%`)

Question:
Did the report finish its thoughts and cover the important topics?

Look for:
- no mid-sentence cutoffs
- no abruptly ended sections
- required sections all present
- major project requirements represented

High score:
- sections feel complete
- no obvious context-window collapse

Low score:
- truncated sections
- missing major themes
- "summary started but never landed" feeling

### 5. Traceability and Grounding (`10%`)

Question:
Can a reviewer trace claims back to evidence with reasonable confidence?

Look for:
- section claims visibly grounded in source material
- citations or source naming where appropriate
- evidence-to-summary relationship is understandable

High score:
- reviewer can verify claims without too much guesswork

Low score:
- claims feel polished but unsupported
- evidence shown does not match the summary tone/content

### 6. Professional Polish (`10%`)

Question:
Does the report feel like something a human would want to read and share?

Look for:
- clean formatting
- no encoding corruption
- no stray extraction markers
- sensible section ordering

High score:
- presentation is clean and client-ready

Low score:
- visual clutter, broken characters, artifacts, or formatting instability distract from the content

### 7. Actionability (`5%`)

Question:
Does the report help a user decide what to do next?

Look for:
- clear risks
- clear next actions
- useful prioritization

High score:
- user could act on the report immediately

Low score:
- the report is descriptive but not decision-useful

## Hard Failure Conditions

Any of the following should cap the overall grade at `C+` or lower unless explicitly waived:

- one or more major sections cut off mid-sentence
- repeated context-window truncation
- evidence layer is dominated by extraction garbage
- major category misrouting that changes the meaning of the report
- unsupported claims that would mislead a reviewer

Any of the following should cap the overall grade at `B` or lower:

- noticeable encoding corruption throughout
- repeated duplication across sections
- appendix/forms/TOC noise still regularly surfaces in user-facing sections

## Letter Grade Bands

- `A` = `93-100`
- `A-` = `90-92`
- `B+` = `87-89`
- `B` = `83-86`
- `B-` = `80-82`
- `C+` = `77-79`
- `C` = `73-76`
- `C-` = `70-72`
- `D` = `60-69`
- `F` = `<60`

## Practical Interpretation

- `A`: client-ready, only minor polish left
- `B`: strong internal report, useful now, but not fully polished
- `C`: promising but clearly flawed; needs pipeline fixes before trust
- `D`: too unstable or noisy to rely on
- `F`: failed run or misleading output

## Review Template

Use this template when grading a report:

```md
# Extraction Report Review

## Overall Grade
- Grade: `B`
- Weighted Score: `85/100`

## Category Scores
- Evidence Quality (`25%`): `3/5`
- Synthesis Quality (`20%`): `4/5`
- Category Fidelity (`15%`): `4/5`
- Completeness (`15%`): `2/5`
- Traceability and Grounding (`10%`): `4/5`
- Professional Polish (`10%`): `3/5`
- Actionability (`5%`): `4/5`

## What Worked
- ...
- ...

## Why It Is Not An A
- ...
- ...

## Hard Failures / Grade Caps
- ...

## Recommended Fixes
- Immediate:
- Near-term:
- Longer-term:
```

## Current Baseline Heuristics

For the current pipeline, use these expectations:

- If raw evidence is messy but summaries are strong, expect a `B` range rather than an `A`.
- If summaries are strong but sections truncate, cap at `C+`.
- If category organization is mostly right and the report is useful despite noise, `B-` to `B` is appropriate.
- If the report feels analyst-useful but not shareable as-is, it is probably in the `B` band.

## Notes For Future Iteration

As the pipeline improves, this rubric should get stricter:

- once truncation is fixed, incomplete sections should be rare and penalized harder
- once table normalization improves, raw evidence noise should be scored more aggressively
- once category routing stabilizes, cross-category bleed should no longer be treated as normal
