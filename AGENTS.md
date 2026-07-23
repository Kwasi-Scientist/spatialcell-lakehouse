# SpatialCell Lakehouse Agent Instructions

## Project Purpose

SpatialCell Lakehouse is a bioinformatics data-engineering project for reproducible spatial-transcriptomics processing.
AnnSQL provides SQL-oriented AnnData access, PySpark supports scalable transformations, and Airflow will orchestrate workflows.

## Current Milestone

The active milestone is a reliable Bronze-to-Silver pipeline:

1. Load and validate configuration.
2. Read the Bronze H5AD without modifying it.
3. Preserve Ensembl identifiers and annotate gene symbols.
4. match and extract curated liver marker genes.
5. Write normalized Silver Parquet tables.
6. Validate schemas, identifiers, spatial coordinates, and audit metrics.

Do not introduce them into unrelated work unless the task explicitly requires them.

## Repository Map

- `configs/`: declarative dataset and pipeline configuration.
- `data/bronze/`: immutable raw source data; never commit large source files.
- `data/silver/`: normalized, typed, validated intermediate tables.
- `data/gold/`: reproducible aggregates and analytical products.
- `src/ingest/`: dataset discovery, download, and source verification.
- `src/extract/`: AnnData annotation, marker selection, and table extraction.
- `src/transform/`: PySpark transformations and analytical table construction.
- `src/validate/`: schema, integrity, and biological validation.
- `src/report/`: reports, figures, and run summaries.
- `tests/`: unit tests, synthetic fixtures, and lightweight integration tests.

## Codex Working Protocol

- Read this file, `README.md`, relevant configuration, and affected source files before editing.
- Inspect the current implementation and tests; do not assume the README reflects completed code.
- For multi-file, architectural, or ambiguous changes, provide a concise plan before editing.
- Keep each task narrowly scoped and modify only files required for the requested outcome.
- Avoid unrelated refactors, formatting churn, and speculative abstractions.
- Preserve user-authored biological annotations and curated marker data.
- Add or update focused tests with behavioral changes.
- Run targeted tests first, then the broader available suite when practical.
- Never claim completion when required tests fail or requested work remains.

## Python Engineering Rules

- Use `pathlib.Path` for filesystem paths.
- Add type hints to public functions and methods.
- Add concise docstrings to public modules, classes, and functions.
- Prefer small, single-purpose functions with explicit inputs and outputs.
- Separate pure transformations from filesystem, network, and orchestration side effects.
- Perform no file I/O, Spark startup, AnnSQL setup, or Airflow initialization at import time.
- Use structured logging in production modules rather than ad hoc `print()` calls.
- Raise informative exceptions for invalid configuration or scientific data.
- State whether AnnData functions copy or mutate; default to avoiding unexpected mutation.

## Bioinformatics Invariants

- Preserve original Ensembl feature identifiers in a dedicated column.
- Prefer dataset-provided `feature_name` annotations before external identifier services.
- Treat gene symbols as potentially missing, duplicated, renamed, or non-unique.
- Audit symbol collisions before making `var_names` unique.
- Never silently replace unmatched identifiers with nulls or discard them.
- Preserve alignment among `adata.X`, `adata.obs`, and `adata.var`.
- Validate `adata.n_obs == len(adata.obs)` and `adata.n_vars == len(adata.var)`.
- Copy before destructive subsetting unless explicit in-place behavior is requested and documented.
- Preserve sparse matrices wherever practical.
- Never densify the full transcriptome matrix without a justified memory estimate.
- Do not assume `adata.X` contains raw counts; inspect and document the expression source or layer.
- Separate pipeline-quality validation from biological interpretation.
- Do not describe exploratory summaries as formal differential-expression results.
- Do not claim statistical significance without a defined test and multiple-testing correction.

## Visium-Specific Rules

- Call observations `spots`, not cells, unless single-cell segmentation is explicitly documented.
- Do not treat a Visium spot as a pure cell population.
- Describe marker-derived results as compartment, lineage, or cell-type enrichment signals.
- Preserve stable spot identifiers and spatial coordinates through every transformation.
- Validate the configured spatial key before extraction.
- Do not assume array coordinates and image pixel coordinates are interchangeable.
- Record coordinate names, units, orientation, scale factors, and source when available.

## Marker-Gene Rules

- Treat literature-derived marker dictionaries as curated, versioned source data.
- Preserve general and specific cell-type groupings.
- Do not add, rename, merge, or remove curated markers without explicit approval.
- Match markers against gene symbols only after identifier annotation is audited.
- Report requested, matched, unmatched, duplicated, and ambiguous marker genes.
- Never silently remove unmatched markers; exclude them from computation but retain them in audit outputs.
- Do not infer a compartment from a single marker when compatible multi-marker evidence is available.
- Keep full marker dictionaries out of this file; use the repository's marker source of truth.

## Data-Engineering Invariants

- Treat Bronze inputs as immutable and make ingestion idempotent.
- Write normalized, typed, validated records to Silver.
- Write reproducible aggregates and analytical products to Gold.
- Use Parquet as the standard tabular storage format.
- Define explicit schemas for durable outputs.
- Preserve stable identifiers required for joins; never use row order as a relational key.
- Validate key uniqueness, required columns, nullability, value ranges, and row counts before publication.
- Avoid materializing zero-valued entries when converting sparse expression matrices to long form.
- Do not collect large Spark DataFrames into Pandas; aggregate or filter first.
- Do not commit raw H5AD files, generated Parquet datasets, credentials, or large transient outputs.

## Configuration Rules

- `configs/dataset_config.yaml` is the source of truth for dataset and pipeline settings.
- Resolve the repository root from `src/config.py`, not the process working directory.
- Resolve configured relative paths against that repository root.
- Never hardcode usernames, drive letters, Google Drive mounts, or local absolute paths.
- Load YAML with safe parsing and validate required sections and values.
- Configuration loading must not silently create files or directories.
- Validate the input H5AD path before pipeline execution; never fabricate an absent input.

## Schema Stability

- Treat published Silver and Gold column names, types, nullability, and keys as contracts.
- Core identifiers should remain explicit, including `spot_id`, `ensembl_id`, and `gene_symbol`.
- Expression and spatial outputs should use documented fields such as `expression`, `spatial_x`, and `spatial_y`.
- Schema changes require a stated migration, updated tests, updated documentation, and downstream impact review.

## Testing and Verification

- Use `pytest` for unit and integration tests.
- Test configuration loading, validation, path resolution, and missing-file behavior.
- Test gene annotation, missing symbols, duplicate symbols, and Ensembl fallback behavior.
- Test marker flattening, matching, ambiguity reporting, and sparse subsetting.
- Test sparse-to-long conversion without densifying the full expression matrix.
- Use tiny synthetic AnnData fixtures instead of the full liver dataset in routine tests and CI.
- Synthetic fixtures should include sparse values, spatial coordinates, duplicate and missing symbols, and an unavailable marker.

## Code-Review Rules

- Flag any change that discards Ensembl IDs or overwrites source feature annotations.
- Flag unconditional dense conversion of a full AnnData expression matrix.
- Flag writes that modify or overwrite Bronze source data.
- Flag hardcoded machine-specific paths or configuration derived from the working directory.
- Flag silent marker loss, silent schema changes, or unsupported biological claims.

## Prohibited Actions

- Do not edit, overwrite, rename, or delete raw files in `data/bronze/`.
- Do not commit secrets, credentials, private URLs, personal paths, or protected biomedical data.
- Do not commit large source datasets or generated tables unless explicitly approved as tiny fixtures.
- Do not fabricate biological metadata, marker provenance, validation results, or test outcomes.
- Do not bypass failing validation to produce outputs.
- Do not run destructive Git or filesystem commands without explicit authorization.

## Sources of Truth

- `README.md` defines project intent, architecture, setup, and roadmap.
- `configs/dataset_config.yaml` defines dataset and pipeline configuration.
- The marker registry module or marker-panel configuration defines curated marker genes and provenance.
- The source H5AD and its `obs`, `var`, `obsm`, `layers`, and `uns` define biological source metadata.
- GitHub Issues or the current user request define task-specific scope.
- When sources conflict, stop, describe the conflict, and request direction rather than guessing.
