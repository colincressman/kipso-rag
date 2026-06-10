# Extraction Studio Redesign

Note: this document describes an older deliverables-plus-advanced-pipeline concept. The current product has been simplified back to branch extraction plus direct report assembly.

## Goal

Redesign Extraction Studio so that:

- branch extraction remains generic and user-defined
- report generation feels like a product, not a prompt console
- normal users do not need to understand `{items_text}`, `{branch_names}`, chunk IDs, or pass chaining
- the backend can still support grounded, multi-section report generation for many document types
- advanced users still have an escape hatch for custom pipelines

This redesign does **not** replace the generic extraction engine. It adds a
better product layer on top of it.

## Product Position

Extraction Studio should be framed as:

1. `Extract evidence` from documents using branches
2. `Turn that evidence into a deliverable` using a guided report builder

That is different from the current experience, which effectively asks the user
to program a post-processing pipeline.

## Core Principles

1. Users should describe outputs in human terms, not prompt-template terms.
2. Branches stay generic and flexible.
3. Report generation defaults to grounded structured synthesis.
4. Raw prompt mechanics move to advanced mode.
5. The same product shape must work for specs, textbooks, laws, manuals, and
   mixed document sets.

## New UX Model

Extraction Studio should expose three levels:

1. `Preset Reports`
2. `Custom Report Builder`
3. `Advanced Pipeline`

These are modes, not separate products.

## Mode 1: Preset Reports

This is the default path for most users.

The user chooses:

- `Document Type`
- `Optional Domain Lens`
- `Report Type`
- `Source Branches`
- `Audience`
- `Grounding Strictness`
- `Optional Special Instructions`

The guided path should also provide:

- suggested evidence packs
- live report preview
- preflight coverage warnings

The user should **not** see:

- system prompt
- user prompt template
- `{items_text}`
- `{branch_names}`
- chunk ID language
- input source
- execution mode
- post-pass chaining

### Example Document Types

- Public bid / spec package
- Textbook / course material
- Policy / law / regulation
- Technical manual / SOP
- Financial / accounting material
- General document set

### Example Report Types

- Executive Summary
- Checklist
- Risk Review
- Compliance Obligations
- Study Guide
- Key Concepts by Topic
- Comparison Report
- Custom Structured Report

### Example Preset Pairings

Public bid / spec package:

- Proposal Readiness Summary
- Bid / No-Bid Review
- Proposal Writer Checklist

Textbook / course material:

- Study Guide
- Definitions and Key Concepts
- Exam Review Checklist

Policy / law / regulation:

- Compliance Obligations Summary
- Deadlines and Filing Requirements
- Ambiguities Requiring Counsel Review

Technical manual / SOP:

- Operating Procedure Summary
- Maintenance Checklist
- Training Guide

Financial / accounting material:

- Reporting Requirements Summary
- Control / Risk Review
- Exception Checklist

## Mode 2: Custom Report Builder

This is the flexible default for users whose use case does not match a preset.

Instead of writing prompts directly, the user fills out a guided form:

- `Report Name`
- `What are you trying to produce?`
- `Who is this for?`
- `Which branches should be used?`
- `Section Headings`
- `Optional Domain Lens`
- `Optional Special Instructions`
- `Grounding Strictness`
- `Show empty sections or hide them`

### Example Inputs

`What are you trying to produce?`

- Help an executive understand the main decision points
- Create a study guide for final exam review
- Identify compliance duties and deadlines
- Build a practical handoff summary for engineers

`Who is this for?`

- Executive
- Project manager
- Engineer
- Student
- Compliance reviewer
- General reader
- Custom

`Section Headings`

This should be a simple reorderable list, not a prompt textarea.

Example:

- Scope Summary
- Required Actions
- Deadlines
- Risks
- Open Questions

The system then generates the internal structured report config automatically.

## Mode 3: Advanced Pipeline

This preserves the current power-user behavior.

Advanced mode can continue exposing:

- multiple passes
- selected pass outputs
- previous pass chaining
- custom system prompts
- custom user prompt templates
- raw execution modes
- report visibility toggles

This mode should be clearly labeled as expert-oriented.

Recommended label:

`Advanced Pipeline (for custom prompt and multi-pass workflows)`

## Recommended Main Screen Layout

### Step 1: Choose Documents and Branches

This remains close to the current branch workflow.

The user:

- selects documents or collection
- creates branches
- previews branch plan

### Step 2: Choose Output Mode

Three large cards:

- `Preset Report`
- `Custom Report Builder`
- `Advanced Pipeline`

### Step 3A: Preset Report Form

Fields:

- Document Type
- Optional Domain Lens
- Report Type
- Source Branches
- Audience
- Grounding Strictness
- Special Instructions
- Live report preview

### Step 3B: Custom Report Builder Form

Fields:

- Report Name
- Goal
- Audience
- Source Branches
- Section Headings
- Optional Domain Lens
- Special Instructions
- Grounding Strictness
- Empty Section Behavior

### Step 3C: Advanced Pipeline Form

This can reuse much of the current post-pass editor with cleaner copy.

## Suggested Copy Changes

Current terms like `Post-Branch Pass` are too implementation-heavy.

Recommended replacements:

- `Post-Branch Pass` -> `Report Step`
- `Input Source` -> `What this step reads`
- `Execution Mode` -> `How this step runs`
- `Selected branch items` -> `Only these branches`
- `Previous pass output` -> `Output from the prior step`
- `Selected pass outputs` -> `Outputs from chosen earlier steps`

For non-advanced modes, remove these fields entirely.

## Preset Report Configuration Model

Internally, each preset should map to:

- section list
- internal selection instructions
- writing instructions
- empty-section behavior
- heading overflow strategy
- recommended grounding strictness

Example:

`Bid / No-Bid Review`

- sections:
  - Positive Fit Indicators
  - Potential No-Bid or Caution Items
  - Scope Clarity Issues
  - Technical Risk Items
  - Commercial / Procurement Risk Items
  - Schedule or Staffing Risk Items
  - Items Requiring Clarification
  - Overall Bid Readiness Assessment
- assessment labels:
  - Strong Fit Based on Extracted Text
  - Possible Fit, Needs Human Review
  - High Risk / Needs Clarification
  - Insufficient Extracted Information
- grounding: strict
- overflow strategy: bucket and assemble

## Custom Report Configuration Model

Internally, a custom report should compile into:

- report name
- audience
- goal
- section list
- branch scope
- focus notes
- grounding strictness
- empty-section behavior

The application should then create the internal structured-report prompt/schema
without making the user author that prompt directly.

## Grounding and Overflow Strategy

Normal users should not configure this directly.

The engine default for preset/custom reports should be:

1. select evidence IDs into headings
2. deduplicate IDs
3. rehydrate original evidence
4. if a heading is too large, bucket evidence within that heading
5. write one subsection at a time from original evidence only
6. assemble the heading

Avoid prose-on-prose reduction by default.

Advanced mode may still expose older map-reduce behavior where needed.

## How Users Inject Specificity

Users still need a way to guide the report without writing raw prompts.

Recommended inputs:

- `Audience`
- `Goal`
- `Focus Notes`
- `Section Headings`
- `Grounding Strictness`

This gives enough control for different document types without requiring prompt
engineering.

### Example Focus Notes

- Focus on deadlines, forms, and owner questions
- Prioritize formulas and likely exam material
- Emphasize reporting obligations and penalties
- Highlight operational risks and startup constraints

## Example User Flows

### Public Spec Package

User does:

- create branches for scope, procurement, schedule, risks, controls
- choose `Preset Report`
- document type: `Public bid / spec package`
- report type: `Bid / No-Bid Review`
- audience: `Project manager`
- focus note: `Be conservative and highlight owner questions`

System does:

- run grounded structured report using preset schema

### Textbook

User does:

- create branches for chapter concepts, definitions, formulas, examples
- choose `Custom Report Builder`
- goal: `Create a study guide for exam prep`
- audience: `Student`
- sections:
  - Core Concepts
  - Definitions
  - Formulas
  - Common Mistakes
  - Review Questions

System does:

- run the same structured-report engine with a different schema

### Accounting Law

User does:

- create branches for filing requirements, deadlines, penalties, exceptions
- choose `Preset Report` or `Custom Report Builder`
- goal: `Summarize compliance duties`
- audience: `Compliance reviewer`

System does:

- generate a compliance-focused structured report

## Backend Mapping

The redesign can map onto the current backend in phases.

### Phase 1

Keep:

- branch extraction model
- structured report mode
- advanced pass config

Add:

- report templates
- custom report builder form
- compiler from guided form -> internal structured report config

### Phase 2

Improve:

- section overflow handling to avoid prose reduction
- clearer result summaries
- preview of generated internal report schema

### Phase 3

Optional:

- save reusable report templates
- domain-specific starter packs
- report comparison / rerun tools

## Recommended Defaults

Default experience:

- branches visible
- report builder visible
- report preview visible
- advanced hidden behind an expert toggle

Default report engine:

- grounded structured report
- strict evidence use
- show empty sections
- conservative synthesis

## Open Questions

These should be resolved before implementation:

1. Should `Document Type` be required or optional?
2. Should preset reports be domain-specific or mixed in one library?
3. Should users be able to save custom report builders as templates?
4. How much of advanced mode should remain editable in the first release?
5. Should branches themselves eventually get presets too?

## Recommendation

Build the redesign around this split:

- generic branch extraction
- guided report builder for most users
- advanced pipeline for power users

That preserves the flexibility that makes Extraction Studio differentiated while
removing the developer-facing UX that currently makes post-processing confusing.
