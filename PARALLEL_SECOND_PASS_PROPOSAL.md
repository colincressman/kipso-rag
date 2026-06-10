# Parallel Second-Pass Proposal

## Goal

Reduce wall-clock time for the second-pass pipeline by parallelizing independent work while keeping the report grounded, deterministic, and easy to recover from after partial failures.

This proposal does **not** change the current runtime. It describes a future architecture for environments with enough memory to run multiple workers of the same model in parallel.

## Current Problem

The current second-pass flow is mostly serial:

1. organize evidence into categories
2. summarize categories
3. write executive summary
4. write key findings
5. write next actions
6. assemble report

Inside those passes, many calls are also serial:

- category routing batches
- category summaries
- reduction chains

This creates long wall-clock runs even when much of the work is independent.

## Core Idea

Run the same model in multiple parallel workers for each independent stage.

Instead of:

- one worker doing all category routing
- then one worker doing all category summaries
- then one worker doing all report sections

Use:

- multiple identical workers
- stage barriers between dependent phases
- deterministic artifact handoff between stages

Workers can spin up for a stage, process their assigned jobs, then idle or spin down.

## Proposed 4-Stage Flow

### Stage 1: Organize By Category

Input:

- branch evidence lines
- report categories

Jobs:

- one routing job per report category
- each routing job may itself process multiple directory batches

Output:

- one shared `category_organizer_v1` artifact

Notes:

- all routing stays ID-based
- rehydration happens after ID selection
- this is the main structural artifact reused by later passes

### Stage 2: Summarize By Category

Input:

- `category_organizer_v1`

Jobs:

- one summary job per organized category
- each category may still batch internally if it is too large

Output:

- `category_summaries_v1`

Notes:

- this stage should summarize from grounded category evidence
- not from prior synthesized prose

### Stage 3: Report Synthesis

Input:

- preferably `category_organizer_v1`
- optionally category summaries where appropriate for display, but not as the primary evidence source

Jobs:

- `Executive Summary`
- `Key Findings`
- `Next Actions`

These can run in parallel once organization is complete.

Output:

- `executive_summary_v1`
- `key_findings_v1`
- `next_actions_v1`

Notes:

- these should remain grounded in organized evidence
- they should not become summary-on-summary synthesis by default

### Stage 4: Assemble Report

Input:

- all completed second-pass artifacts

Jobs:

- one deterministic assembly job

Output:

- final markdown report

Notes:

- this stage should remain serial
- it is cheap compared to the others

## Why This Helps

This changes the wall-clock shape from many sequential calls to a few synchronized waves:

1. route all categories in parallel
2. summarize all categories in parallel
3. write final report sections in parallel
4. assemble

This should reduce end-to-end time substantially on hardware that can host multiple model workers simultaneously.

## Why This Is Safe

The proposal reuses only **structural** work, not synthesized prose, as the main evidence source.

That means:

- organization can be shared
- summaries, findings, and actions still read grounded evidence
- final prose remains independently generated where it matters

This is consistent with the current design goal:

- reuse the sorting
- do not rely on synthesized text as the primary evidence input for later synthesis

## Worker Model

Use multiple workers of the **same model**.

Each worker:

- loads the same model
- accepts one job at a time
- returns a normal pass artifact

At the stage level:

- create a worker pool of size `N`
- queue all jobs for the stage
- wait for all jobs to finish
- then advance to the next stage

Possible pool sizes:

- `2`
- `4`
- `8`
- configurable

The best pool size depends on:

- RAM
- VRAM
- model size
- backend behavior
- whether the inference stack truly runs requests concurrently

## Scheduling Model

This should be modeled as a stage scheduler, not a serial pass runner.

Suggested structure:

```text
build_stage_plan(project)
  -> stage_1_jobs
  -> stage_2_jobs
  -> stage_3_jobs
  -> stage_4_jobs
```

Then:

```text
for each stage:
  run all independent jobs in parallel
  checkpoint each completed job
  wait for stage completion
```

## Artifact Requirements

To support this well, every stage should emit durable, independently recoverable artifacts.

Minimum artifacts:

- `category_organizer_v1`
- `category_summaries_v1`
- `executive_summary_v1`
- `key_findings_v1`
- `next_actions_v1`
- `assembled_report_v1`

Each artifact should include:

- artifact type
- pass/job name
- completion status
- timing
- warnings/errors
- enough data to avoid rerunning completed jobs unnecessarily

## Checkpointing Requirements

Parallelization is only worth it if partial work is recoverable.

Recommended behavior:

- checkpoint after each category-routing job
- checkpoint after each category-summary job
- checkpoint after each stage-3 synthesis job
- assemble only from completed artifacts

If a worker fails:

- retry that job
- if retries fail, preserve completed sibling jobs
- allow rerun to resume from the checkpointed stage state

## Failure Handling

### Routing Failures

Current direction is already moving toward this:

- retry failed batch
- if it still fails, continue and keep completed work

Future stage scheduler should extend this:

- failed category routing job does not kill the whole stage immediately
- stage completes with warnings if enough work succeeded

### Summary Failures

If one category summary fails:

- other category summaries should still complete
- rerun should only need to redo the failed category

### Stage 3 Failures

If `Executive Summary` fails:

- `Key Findings` and `Next Actions` should still be kept if they succeeded

## Configuration Proposal

Suggested future config options:

```yaml
second_pass_parallel:
  enabled: false
  worker_count: 4
  max_workers: 4
  stage_barrier: true
  max_parallel_routing_jobs: 4
  max_parallel_summary_jobs: 4
  max_parallel_report_jobs: 3
```

Optional:

```yaml
second_pass_parallel:
  keep_workers_warm_between_stages: true
```

This may reduce reload cost if the inference backend supports it cleanly.

## Worker Limits In Settings

The system should expose a user-configurable maximum worker setting.

This is important because different machines will have very different practical concurrency limits even if the software architecture is the same.

Suggested setting:

```yaml
second_pass_parallel:
  max_workers: 4
```

This should act as the hard upper bound for:

- category-routing workers
- category-summary workers
- stage-3 synthesis workers

The scheduler can then choose:

- `min(max_workers, jobs_in_stage)`

rather than assuming a fixed or unlimited pool size.

This matters because:

- some users may only be able to run `2` or `3` workers safely
- others may be able to run `8`, `10`, or more
- available memory is the real bottleneck

### Example Hardware Constraint

For example, a machine like an Nvidia Spark with `128 GB` of unified memory might only realistically support about `10` simultaneous workers for this pipeline.

Rough budget:

- main LLM worker: about `9.8 GB` each
- HyDE model: about `2 GB`
- embedding model: about `6.6 GB`
- Hugging Face support models: about `2 GB`

At `10` workers:

- main workers: about `98 GB`
- HyDE: about `2 GB`
- embeddings: about `6.6 GB`
- Hugging Face models: about `2 GB`

Total:

- about `108.6 GB`

That leaves roughly:

- about `19.4 GB`

for the operating system and runtime overhead.

So even on a large-memory machine, concurrency still needs to be capped deliberately.

Because of that, `max_workers` should be treated as a first-class deployment setting, not a hidden implementation detail.

## UI / UX Impact

The current UI can stay mostly the same.

Potential additions later:

- stage progress display
- active worker count
- per-stage timing
- per-job status:
  - queued
  - running
  - done
  - failed

The important product behavior should be:

- users still configure second passes normally
- the scheduler decides how to parallelize them under the hood

## Expected Speedup

The exact speedup depends on the backend, but the intended wall-clock reduction is:

- from many small serial steps
- to roughly 4 major synchronized stages

Approximate stage count compression:

- current effective serial flow: many category jobs + many summaries + many final sections
- proposed parallel flow: `4` main wall-clock stages

This does **not** reduce total compute.

It reduces elapsed time by executing independent work concurrently.

## Constraints

This cannot be validated well on the current limited hardware setup because:

- multiple parallel model instances are difficult to host locally
- local testing may introduce CPU fallback or model churn
- backend concurrency behavior may differ from the high-memory work server

So this proposal is intentionally design-first, not implementation-first.

## Recommended Future Implementation Order

1. Add per-job stage artifacts and durable checkpointing.
2. Add a stage planner for second passes.
3. Parallelize Stage 1 category routing.
4. Parallelize Stage 2 category summaries.
5. Parallelize Stage 3 executive/findings/actions.
6. Add worker-pool configuration and monitoring.
7. Test on the high-memory server with smaller load first.

## Recommendation

This is a strong direction for the work server.

The key architectural rule should be:

- parallelize independent grounded work
- reuse organizer artifacts
- keep final synthesis grounded in evidence
- checkpoint aggressively so failures do not waste entire stages

That should produce the best combination of:

- speed
- recoverability
- groundedness
- predictable report structure
