# Extraction Pipeline Audit

Date: 2026-06-06

Scope audited:

- branch extraction and categorization flow
- structured report generation flow
- chunk / bullet ID handling
- rehydration behavior
- report chunking / reduce / final concatenation
- checkpoint rerun path
- modularity / genericity for new domains and scripted usage
- stale or leftover code / docs that no longer match runtime

Files reviewed:

- `extraction/batch.py`
- `extraction/branch_config.py`
- `extraction/branch_runner.py`
- `extraction/prompts.py`
- `extraction/project_runner.py`
- `extraction/report_builder.py`
- `extraction/guided_reports.py`
- `server/routes/extraction.py`
- `scripts/ops/run_extraction.py`
- `tests/test_extraction.py`
- `README.md`
- `LargeDocIngest.md`
- `docs/structured_report_pipeline_walkthrough.md`
- `docs/grounded_structured_report_mode.md`
- `configs/extraction.yaml`

Verification note:

- I performed a code-path audit and consistency review.
- I was not able to run `pytest` inside the project venv because the checked-in venv Python is broken on this machine (`.venv\Scripts\python.exe` resolves to a blocked Windows Store path), and the bundled Codex Python does not have `pytest` installed.

## Executive Summary

The current pipeline does implement the core design you described:

1. documents are ingested into chunks
2. branches act as evidence categories
3. evidence is selected using short numeric IDs
4. selected IDs are rehydrated back to full chunks
5. extracted items are organized by branch/category
6. guided reports compile into a grounded structured-report pass
7. that pass routes evidence into report sections
8. each section is written from rehydrated evidence only
9. the final report is assembled by concatenating ordered section outputs

The strongest part of the design is that the branch extractor is generic and grounded. It does not generate branch bullets from scratch; it selects source chunks and turns them into verbatim `ExtractionItem`s. That is a good foundation for domain portability.

The main caveat is that some comments, tests, and older docs still describe an older "LLM writes bullets" or "JSON selector is primary" architecture that is not the runtime architecture anymore. The runtime is now more grounded than some of the surrounding docs imply.

My overall verdict:

- The core extraction-to-report flow is real and mostly coherent.
- The engine is generic enough for new domains if users can express their domain as branches plus report headings.
- The report layer is not yet exposed as a clean standalone library API for "I already have bullets/items, just build the report."
- There is definite stale code and stale documentation that should be cleaned up.

## End-to-End Validation

### 1. Categorization into branches is real

The first organizational layer is the branch system, not the structured report.

Evidence:

- `run_project()` orchestrates enabled branches in series: `extraction/project_runner.py:935`
- `run_branch()` is the branch executor: `extraction/branch_runner.py:342`
- each branch is configured by `BranchConfig` and stored in `ProjectConfig`: `extraction/branch_config.py`

What actually happens:

1. retrieval gets candidate chunks for one branch
2. keyword filtering / boosting and source-priority weighting are applied
3. the scan pass selects candidate chunk IDs
4. those IDs are rehydrated to full chunks
5. the synthesis pass selects which full chunks become final extracted items
6. those extracted items are stored under that branch/category

This means "organize bullets into categories" is implemented as "organize verbatim extracted items into branches."

### 2. Bullet / chunk IDs are central to the flow

There are two distinct ID layers in the current design.

Short numeric IDs:

- scan pass IDs are assigned by `assign_ids_and_batch()`: `extraction/batch.py:47`
- rehydration back to full chunks is done by `rehydrate()`: `extraction/batch.py:139`
- synthesis sub-batches get their own numeric IDs via `assign_ids_for_synthesis()`: `extraction/batch.py:260`

Persistent source chunk IDs:

- extracted items carry `source_chunk_id`: `extraction/branch_config.py` (`ExtractionItem`)
- post-pass inputs format those as `[chunk:<id>] [<branch>] <text>` in `_format_item_for_pass()`

Conclusion:

- yes, IDs are deeply involved
- yes, there is a true rehydration step
- yes, the report path depends on those IDs staying intact

### 3. Rehydration exists in two places

Branch-level rehydration:

- true numeric-ID-to-full-chunk rehydration happens in `rehydrate()`: `extraction/batch.py:139`
- `run_branch()` uses it after scan selection: `extraction/branch_runner.py:439`

Structured-report rehydration:

- `_run_structured_report_pass()` builds an `item_map` keyed by `source_chunk_id`: `extraction/project_runner.py:655`
- section assignments are later mapped back to full evidence lines through that map: `extraction/project_runner.py:698`

Important nuance:

- the report stage does not go back to the raw DB chunk rows
- it rehydrates to formatted extracted evidence lines, not raw source chunks

Because branch items are currently verbatim chunk text, this still preserves grounding. But the report engine is implicitly depending on branch extraction staying verbatim forever.

### 4. The branch "synthesis" stage does not write bullets anymore

This is one of the biggest architecture clarifications from the audit.

Current code:

- `build_synthesis_prompts()` explicitly says the synthesis pass selects chunk IDs and "does NOT generate any text": `extraction/prompts.py:75-109`
- `run_branch()` collects `selected_synth_ids`, resolves them to `final_chunks`, then creates `ExtractionItem`s directly from chunk text: `extraction/branch_runner.py:455-496`

That means:

- the LLM is not writing extracted bullets in the branch pipeline
- the LLM is performing a second selection pass
- the final "bullets" are just verbatim chunk text rendered later by the report builder

This is good for grounding, but several surrounding comments/docs still describe the older behavior.

### 5. Structured report generation is chunked and assembled in stages

Guided reports compile into a structured-report post pass:

- prompt construction: `extraction/guided_reports.py:664`
- compile step: `extraction/guided_reports.py:719`
- runtime pass mode: `structured_report`

Runtime execution:

- post passes are compiled and run from `compiled_post_passes()` plus `_run_post_passes()`: `extraction/guided_reports.py:740`, `extraction/project_runner.py:795`
- structured report mode is dispatched explicitly: `extraction/project_runner.py:872-874`
- `_run_structured_report_pass()` is the core structured report engine: `extraction/project_runner.py:641`

What it does:

1. infer target section headings from the prompt
2. gather branch-item evidence
3. split evidence into batches with `_split_items_text()`
4. route IDs into sections batch by batch
5. merge and dedupe section assignments
6. rehydrate assigned IDs to evidence lines
7. split oversized section evidence into batches again
8. write each section
9. reduce multi-batch section partials if needed
10. concatenate ordered section outputs into one markdown block

Evidence:

- section batching: `extraction/project_runner.py:665`
- section assignment merge: `extraction/project_runner.py:657-689`
- section evidence batching: `extraction/project_runner.py:704`
- section reduce: `extraction/project_runner.py:724-730`
- final section concatenation: `extraction/project_runner.py:734`

Conclusion:

- yes, the report is written in chunks
- yes, it is concatenated together afterward
- yes, the final assembly is section-based rather than prose-on-prose chaining

### 6. Final report assembly is separate and ordered

The final markdown report is assembled by `assemble_report()`:

- entry point: `extraction/report_builder.py:105`
- TOC: `extraction/report_builder.py:30`
- post-pass visibility filter: `extraction/report_builder.py:52`
- post-pass insertion into final markdown: `extraction/report_builder.py:251`
- save to disk: `extraction/report_builder.py:278`

What gets concatenated:

- branch sections
- optional source-override appendix
- visible post-pass sections

This is a clean separation: structured-report generation builds the report content, and `report_builder.py` places it into the final deliverable.

### 7. Report-only reruns are grounded in checkpointed branch outputs

The rerun path is valid and shares the same reporting machinery.

Evidence:

- checkpoint write: `extraction/project_runner.py:65`
- checkpoint branch reload: `extraction/project_runner.py:72`
- report-only checkpoint seed: `extraction/project_runner.py:90`
- rerun entry point: `extraction/project_runner.py:1075`

Flow:

1. load prior branch results from checkpoint
2. seed a fresh rerun checkpoint
3. rerun post-passes only
4. rebuild the report

This is modular and correct. It does not appear to bypass the grounded report engine.

## What Is Working Well

### Grounded extraction core

The branch extractor is solid conceptually:

- it scales via batching
- it uses IDs rather than sending all chunk text blindly
- it rehydrates selected evidence before the second pass
- it outputs verbatim extracted items

This is a good generic primitive for large documents.

### Clean separation of concerns

The code is reasonably well separated:

- `branch_config.py`: datamodels
- `batch.py`: ID assignment / rehydration utilities
- `prompts.py`: prompt construction
- `branch_runner.py`: branch-level extraction runtime
- `guided_reports.py`: UI-friendly report definition compiler
- `project_runner.py`: orchestration
- `report_builder.py`: final markdown assembly

That modular split is a strength.

### Guided report compiler keeps domain logic mostly outside the runtime

The structured report runtime itself is generic. Domain specialization is mostly pushed into:

- branch definitions
- guided report presets
- domain lens / focus notes
- report headings

That is the right direction if the goal is "bring your own domain."

## Modularity / Genericity Assessment

## Verdict

For "someone with a large document can write a script and build a report":

- Mostly yes.

For "generic enough that someone can tie their domain into it":

- Yes at the extraction/runtime level.
- Partially at the report API level.

### Why I say yes

There is already a scriptable public shape:

- `ProjectConfig` holds documents, branches, guided reports, and post passes
- `run_project()` runs the whole workflow
- `scripts/ops/run_extraction.py` provides a CLI wrapper
- `server/routes/extraction.py` provides API endpoints

A new domain can plug in by changing:

- branch list
- branch keywords / semantic prompts
- document type / domain lens
- guided report sections / audience / goal / focus notes

That is a reasonably generic integration story.

### Why I only say partially

There is not yet a clean standalone library API that says:

- "here are my already-extracted items"
- "here are my report sections"
- "build me a grounded report"

Today, the structured report engine is buried inside `_run_post_passes()` / `_run_structured_report_pass()` and expects branch-style evidence flow. That is usable, but it is not a polished reusable reporting API.

If you want the engine to be maximally reusable by outside scripts, I would eventually expose something like:

- `build_structured_report(items_by_category, report_config, llm_fn) -> markdown`

Right now an external integrator can still use it, but they need to understand the internal `ProjectConfig` / `BranchResult` conventions.

## Findings and Risks

### 1. The structured-report JSON selector path appears to be dead runtime code

Severity: Medium

Evidence:

- `_parse_section_id_map()`: `extraction/project_runner.py:360`
- `_parse_or_repair_section_id_map()`: `extraction/project_runner.py:520`
- `_selector_prompt_for_structured_report()`: `extraction/project_runner.py:572`
- `_run_structured_report_pass()` does not call them; it calls `_fallback_section_id_map_via_scan()` directly: `extraction/project_runner.py:670`
- project-wide search only shows those helpers referenced in docs/tests, not in runtime callers

What this means:

- the old JSON selector / repair pipeline is no longer part of the live structured-report path
- tests and docs still exercise / describe that path
- the runtime is scan-only for routing

Recommendation:

- if scan-only routing is the intended permanent design, remove the dormant JSON selector helpers and update tests/docs accordingly
- if you want to keep them as an optional future strategy, wire them back in explicitly behind a config flag instead of leaving them half-alive

Removal candidates I am comfortable flagging:

- `_parse_section_id_map()`
- `_selector_repair_prompt()`
- `_parse_or_repair_section_id_map()`
- `_selector_prompt_for_structured_report()`

I would only remove them together, because they form one obsolete selector path.

### 2. Several comments/docs still describe an older branch-synthesis architecture

Severity: Medium

Evidence:

- `extraction/branch_runner.py:9` says stage 4 writes bullet output
- `LargeDocIngest.md` still describes "writes the actual output bullets"
- `configs/extraction.yaml` still says `synthesis_pass_temperature` is "slightly warmer for bullet generation"
- current runtime actually does second-pass ID selection and then creates `ExtractionItem`s from chunk text: `extraction/prompts.py:75-109`, `extraction/branch_runner.py:455-496`

Why it matters:

- new contributors will misunderstand the extraction model
- they may assume hallucination risk where the current pipeline actually avoids it
- they may accidentally reintroduce freeform synthesis in the wrong place

Recommendation:

- update comments/docs/config comments to match the current runtime truth

### 3. Structured-report rehydration depends on branch items being verbatim forever

Severity: Medium

Evidence:

- report-stage rehydration uses `item_map[source_chunk_id] = formatted_branch_line`: `extraction/project_runner.py:655-669`
- it does not reload original chunk text from the DB

Risk:

- today this is safe because branch items are verbatim chunk text
- if branch extraction ever becomes abstractive or trimmed, the report stage will silently stop being "source-evidence only" and become "derived-evidence only"

Recommendation:

- either keep the "branch items are verbatim" contract explicit and documented
- or make structured-report rehydration fetch the original chunk text by `source_chunk_id` instead of relying on formatted branch evidence lines

### 4. Same `source_chunk_id` selected in multiple branches is collapsed to one evidence line

Severity: Medium

Evidence:

- `item_map[match.group(1)] = line.strip()` overwrites prior entries for the same chunk ID: `extraction/project_runner.py:669`

Risk:

- if the same underlying chunk appears in multiple categories/branches, only the last formatted branch line survives in the map
- the report writer loses branch-context multiplicity for that chunk
- section evidence may show the wrong branch label or hide that the evidence supported multiple categories

Recommendation:

- consider storing `chunk_id -> list[evidence_lines]` or `chunk_id -> best canonical raw evidence`
- if branch labels matter, do not overwrite silently

### 5. Report discovery endpoints and default save paths are inconsistent

Severity: Medium

Evidence:

- `ProjectConfig.report_output_path` defaults to `data/reports`: `extraction/branch_config.py:311`
- `save_report()` writes to `project.report_output_path`: `extraction/report_builder.py:280`
- route fallbacks look in `data/extraction_reports`: `server/routes/extraction.py:360`, `:372`, `:387`
- CLI ad-hoc runs also default to `data/extraction_reports`: `scripts/ops/run_extraction.py:169`

Risk:

- a report can be saved successfully but not show up in the API history/list endpoints
- saved-project runs and ad-hoc/route expectations are not aligned

Recommendation:

- unify on one default
- ideally source that default from one place only

### 6. `_extract_report_headings()` is functional but brittle

Severity: Low to Medium

Evidence:

- heading inference starts only after a phrase containing `following headings`: `extraction/project_runner.py:327`

Impact:

- guided reports are safe because the compiler emits the expected phrase
- advanced/custom users must unknowingly follow a specific prompt shape or the pass fails

Recommendation:

- either formalize section headings as structured config on `PostBranchPass`
- or support a broader set of heading-schema markers than one exact phrase family

### 7. `_split_items_text()` enforces a soft limit, not a hard limit

Severity: Low

Evidence:

- split logic only starts a new batch when `current_parts` already has content: `extraction/project_runner.py:263-272`

Impact:

- a single very long evidence line can exceed `max_chars_per_batch`
- this is more likely now that `scan_chunk_preview_chars` is `0` in config, meaning full chunk text is used in scan directories

Recommendation:

- if strict token budgeting matters, add a single-line oversize splitter or truncation strategy

## Documentation Drift

These docs are not all telling the same story anymore.

### Accurate or mostly accurate

- `README.md` is broadly aligned on the important points:
  - branch extraction is grounded and verbatim
  - structured report uses route IDs -> rehydrate -> write sections

### Needs refresh

- `LargeDocIngest.md`
  - still describes synthesis writing bullets
  - still reflects an older implementation model

- `docs/grounded_structured_report_mode.md`
  - describes the one-shot JSON selector as the internal pipeline
  - does not reflect that the runtime now uses per-section scan routing directly

- `docs/structured_report_pipeline_walkthrough.md`
  - closer to reality than the design doc
  - but still talks about the older JSON selector / repair helpers as a live recovery path even though current runtime no longer calls them

## Stale / Leftover Code and Artifacts

These are the strongest candidates for cleanup.

### Strong removal candidates

- `extraction/project_runner.py`
  - `_parse_section_id_map()`
  - `_selector_repair_prompt()`
  - `_parse_or_repair_section_id_map()`
  - `_selector_prompt_for_structured_report()`

Reason:

- they are not in the live runtime path anymore
- they are only referenced by tests/docs
- they represent an older routing strategy

### Strong stale-test candidates

- tests in `tests/test_extraction.py` that validate the JSON selector / repair path

Reason:

- they are testing a non-runtime strategy
- they create false confidence that a dormant path is still a supported path

### Strong stale-doc candidates

- `LargeDocIngest.md` sections describing branch synthesis as bullet writing
- `docs/grounded_structured_report_mode.md` sections describing JSON routing as the internal path

## Overall Assessment Against Your Desired Design

You asked whether each path is doing what it is designed to do:

- organize bullets into categories
- use them to write a report in chunks
- concatenate it together
- involve bullet IDs
- include rehydration

My answer:

- Yes on categorization: branches are the categories.
- Yes on ID involvement: IDs are foundational in both extraction and reporting.
- Yes on rehydration: true at branch level, and logically true again at report level.
- Yes on chunked report writing: structured reports batch routing and batch section synthesis.
- Yes on concatenation: section outputs and final report sections are concatenated deterministically.

The main "but" is this:

- the branch pipeline does not literally create LLM-authored bullets anymore
- it creates branch-organized verbatim extracted items

That is not a flaw. It is actually a better, safer design. But the docs and comments need to catch up to it.

## Recommended Next Steps

1. Remove or explicitly re-enable the dormant JSON selector path.
2. Unify report output path defaults across `ProjectConfig`, CLI, and API listing routes.
3. Update stale comments/docs to reflect "selection-only branch synthesis."
4. Decide whether structured-report rehydration should remain "from extracted evidence lines" or be upgraded to "from raw source chunks by `source_chunk_id`."
5. Expose a clean public reporting helper if you want outside scripts to use the report engine without knowing internal checkpoint or branch-runner details.

## Bottom Line

The extraction pipeline is substantially modular and grounded, and the core architecture is good. The code already supports a generic large-document workflow where someone can define categories, extract evidence, and build a sectioned report from that evidence.

The biggest issues are not that the pipeline is missing the intended flow. They are:

- stale selector code
- stale tests/docs
- a couple of integration inconsistencies
- one important architectural dependency on branch items staying verbatim

If those are cleaned up, this becomes a much clearer and more reusable extraction/reporting engine.
