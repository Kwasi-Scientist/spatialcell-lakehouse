# SpatialCell Lakehouse

SpatialCell Lakehouse is a bioinformatics data engineering project that transforms liver Visium H5AD data into analysis-ready Parquet tables. It uses AnnSQL for SQL-based extraction, PySpark for scalable transformations, and Airflow for workflow orchestration, with reproducible QC, marker analysis, and GitHub-ready pipelines.

## Project Overview

Spatial transcriptomics datasets are commonly distributed as `.h5ad` files containing expression matrices, spatial coordinates, feature metadata, observation metadata, and unstructured annotations. These files are biologically rich, but they are not always optimized for reusable analytical pipelines, SQL-based exploration, or distributed data processing.

This project converts a public liver Visium dataset into a layered analytical data model:

- **Bronze layer:** raw `.h5ad` source data
- **Silver layer:** normalized Parquet tables for observations, genes, marker expression, spatial coordinates, and quality-control metrics
- **Gold layer:** analysis-ready aggregates, marker summaries, spatial enrichment scores, and pipeline validation outputs

The project is designed as a portfolio demonstration of how modern data engineering practices can be applied to spatial transcriptomics.

## Objectives

- Ingest a public liver Visium dataset from the Chan Zuckerberg CELLxGENE data ecosystem
- Preserve original Ensembl gene identifiers while annotating genes with symbols
- Extract selected AnnData components using AnnSQL and Python
- Convert processed tables to Apache Parquet
- Use PySpark for transformations, joins, aggregations, and feature engineering
- Use Apache Airflow to orchestrate the end-to-end workflow
- Produce reproducible spatial marker analyses and quality-control outputs
- Maintain a modular codebase suitable for local development, testing, and deployment

## Planned Technology Stack

| Tool | Role |
|---|---|
| Python | Core programming language |
| AnnData / Scanpy | Single-cell and spatial transcriptomics data structures |
| AnnSQL | SQL-style querying of AnnData and H5AD data |
| Apache Parquet | Persistent columnar storage |
| PyArrow | Parquet serialization |
| PySpark | Scalable transformations and analytical aggregations |
| Apache Airflow | Workflow orchestration |
| DuckDB | Embedded analytical SQL engine |
| Pandas | Lightweight local inspection and interoperability |
| Google Colab | Reproducibility and optional cloud notebook execution |
| Google Drive | Persistent storage for Colab workflows |
| GitHub | Version control and project documentation |

## Proposed Architecture

```text
Chan Zuckerberg CELLxGENE
            |
            v
    Raw liver Visium H5AD
            |
            v
       Bronze Layer
      data/bronze/
            |
            v
 Gene annotation and AnnSQL extraction
            |
            v
       Silver Layer
      data/silver/
            |
            v
 PySpark transformations and validation
            |
            v
        Gold Layer
       data/gold/
            |
            v
 Reports, plots, and analytical outputs
```

## Initial Analytical Scope

The first version of the project will focus on a curated liver marker-gene panel rather than the complete transcriptome. This keeps the pipeline computationally manageable while preserving biological interpretability.

Planned analyses include:

- Spot-level quality-control summaries
- Marker-gene expression by spatial location
- General cell-type marker scoring
- Specific cell-type marker scoring
- Hepatocyte, endothelial, macrophage, cholangiocyte, and stromal compartment enrichment
- Marker detection rates
- Marker co-expression summaries
- Spatial-bin aggregation
- Pipeline-level data quality checks

## Gene Annotation Strategy

The source `.h5ad` file may use Ensembl gene identifiers as `adata.var_names`. When available, gene symbols are read from `adata.var["feature_name"]`.

The workflow:

1. Preserves the original Ensembl identifiers in `adata.var["ensembl_id"]`
2. Cleans gene symbols from `feature_name`
3. Uses Ensembl identifiers as fallbacks for missing symbols
4. Sets gene symbols as `adata.var_names`
5. Makes duplicated gene symbols unique
6. Preserves the original annotation metadata for traceability

## Marker Gene Strategy

Marker genes were curated from published differential-expression results and organized into two dictionaries:

```python
CELLTYPE_MARKER_GENES
SPECIFIC_CELLTYPE_MARKER_GENES
```

The general dictionary groups marker genes by broad biological category. The specific dictionary groups marker genes by more granular cell types or cell states.

The two dictionaries can be flattened into a unique marker panel for expression extraction while retaining their original group mappings for downstream scoring.

## Proposed Repository Structure

```text
spatialcell-lakehouse/
├── README.md
├── requirements.txt
├── pyproject.toml
├── docker-compose.yml
├── .gitignore
├── configs/
│   └── dataset_config.yaml
├── notebooks/
│   └── 01_colab_pipeline_demo.ipynb
├── dags/
│   └── cellxgene_annsql_pipeline.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest/
│   │   ├── __init__.py
│   │   └── download_cellxgene.py
│   ├── extract/
│   │   ├── __init__.py
│   │   ├── gene_annotations.py
│   │   ├── marker_matrix.py
│   │   └── annsql_extract.py
│   ├── transform/
│   │   ├── __init__.py
│   │   └── spark_transforms.py
│   ├── validate/
│   │   ├── __init__.py
│   │   └── data_quality.py
│   └── report/
│       ├── __init__.py
│       └── build_report.py
├── sql/
│   ├── extract_obs.sql
│   ├── extract_var.sql
│   ├── marker_expression.sql
│   └── qc_metrics.sql
├── data/
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── tests/
│   ├── test_schema.py
│   └── test_transforms.py
└── reports/
```

## Data Layers

### Bronze

The Bronze layer contains the original downloaded dataset.

```text
data/bronze/
└── liver_visium.h5ad
```

The raw `.h5ad` file should not be committed to GitHub.

### Silver

The Silver layer contains normalized, reusable Parquet tables.

Planned outputs:

```text
data/silver/
├── obs.parquet
├── var.parquet
├── spatial_coordinates.parquet
├── qc_metrics.parquet
├── marker_expression_long.parquet
└── gene_id_mapping.parquet
```

### Gold

The Gold layer contains analysis-ready outputs.

Planned outputs:

```text
data/gold/
├── marker_expression_by_region.parquet
├── marker_expression_by_spatial_bin.parquet
├── spot_compartment_scores.parquet
├── marker_gene_coexpression.parquet
├── marker_detection_rates.parquet
└── pipeline_run_metrics.parquet
```

## Local Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/spatialcell-lakehouse.git
cd spatialcell-lakehouse
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure local paths

The project should use relative paths or environment variables instead of hardcoded Google Drive paths.

Example:

```bash
export SPATIALCELL_PROJECT_ROOT="$HOME/projects/spatialcell-lakehouse"
export SPATIALCELL_DATA_ROOT="$HOME/spatialcell-data"
```

### 5. Add the source H5AD file

Place the raw dataset in:

```text
data/bronze/liver_visium.h5ad
```

The file should remain excluded from Git.

## Example Gene Annotation Usage

```python
from src.extract.gene_annotations import set_gene_symbols_from_feature_name

adata = set_gene_symbols_from_feature_name(adata)
```

## Example Marker Extraction

```python
marker_genes = sorted({
    gene
    for genes in CELLTYPE_MARKER_GENES.values()
    for gene in genes
})

available_markers = [
    gene
    for gene in marker_genes
    if gene in adata.var_names
]

marker_adata = adata[:, available_markers].copy()
```

## Example Parquet Workflow

```python
obs_df = adata.obs.reset_index(names="spot_id")
obs_df.to_parquet("data/silver/obs.parquet", index=False)

var_df = adata.var.reset_index(names="gene_symbol")
var_df.to_parquet("data/silver/var.parquet", index=False)
```

## Example PySpark Workflow

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("SpatialCellLakehouse")
    .getOrCreate()
)

expression_df = spark.read.parquet(
    "data/silver/marker_expression_long.parquet"
)

marker_summary = (
    expression_df
    .groupBy("gene")
    .agg(
        F.avg("expression").alias("mean_expression"),
        F.countDistinct("spot_id").alias("n_spots"),
    )
)

marker_summary.write.mode("overwrite").parquet(
    "data/gold/marker_expression_summary.parquet"
)
```

## Planned Airflow DAG

The Airflow DAG will orchestrate the following tasks:

```text
select_dataset
      |
      v
download_h5ad
      |
      v
annotate_genes
      |
      v
extract_marker_matrix
      |
      v
write_silver_parquet
      |
      v
run_spark_transformations
      |
      v
validate_gold_tables
      |
      v
build_report
```

## Data Quality Checks

Planned validation rules include:

- No duplicate spot identifiers
- No duplicate canonical Ensembl identifiers
- Unique AnnData variable names
- Marker expression values are numeric
- Spatial coordinates are present
- Required metadata columns exist
- Parquet schemas match expectations
- No unexpected null values in primary identifiers
- Requested marker genes are audited against available genes
- Output row counts are recorded for each pipeline run

## Git and Data Management

Large biological data files should not be committed to GitHub.

Recommended `.gitignore` entries:

```gitignore
data/bronze/*
data/silver/*
data/gold/*

!data/bronze/.gitkeep
!data/silver/.gitkeep
!data/gold/.gitkeep

*.h5ad
*.h5
*.loom
*.mtx
*.parquet

__pycache__/
*.py[cod]
.venv/
.ipynb_checkpoints/

airflow.db
airflow.cfg
logs/

.DS_Store
.vscode/
.idea/
```

## Development Roadmap

### Phase 1: Baseline extraction

- [ ] Add liver Visium dataset
- [ ] Annotate gene symbols
- [ ] Preserve Ensembl identifiers
- [ ] Add marker dictionaries
- [ ] Extract marker-gene expression matrix
- [ ] Write initial Parquet tables

### Phase 2: PySpark transformations

- [ ] Read Silver Parquet tables with PySpark
- [ ] Build marker-expression summaries
- [ ] Compute marker detection rates
- [ ] Compute compartment scores
- [ ] Create spatial-bin aggregations
- [ ] Write Gold Parquet tables

### Phase 3: Validation and testing

- [ ] Add schema tests
- [ ] Add marker mapping tests
- [ ] Add row-count validation
- [ ] Add null and duplicate checks
- [ ] Add small mock AnnData test fixtures

### Phase 4: Airflow orchestration

- [ ] Create Airflow DAG
- [ ] Add task dependencies
- [ ] Add configurable dataset paths
- [ ] Add pipeline logging
- [ ] Add retry behavior
- [ ] Add data quality gates

### Phase 5: Reporting and application layer

- [ ] Generate marker-expression plots
- [ ] Generate spatial marker maps
- [ ] Build an analytical report
- [ ] Add a lightweight application or dashboard
- [ ] Add a public Colab demonstration
- [ ] Add architecture and data-flow diagrams

## Reproducibility

The repository will retain a Colab notebook as a reproducibility and demonstration environment. The local Python modules will remain the source of truth.

The intended workflow is:

```text
Local development
      |
      v
Git commits
      |
      v
GitHub repository
      |
      v
Colab clones repository
      |
      v
Colab executes the same source modules
```

## Project Status

This project is currently under active development.

Current focus:

- Local project initialization
- Gene-symbol annotation
- Marker-gene dictionary integration
- Marker expression extraction
- Parquet layer design

PySpark and Airflow components are planned for subsequent development phases.

## License

A license has not yet been selected. Consider adding an MIT License if the project is intended to be openly reusable.

## Acknowledgments

This project uses public single-cell and spatial transcriptomics resources from the Chan Zuckerberg CELLxGENE ecosystem and builds on the AnnData, Scanpy, AnnSQL, Apache Parquet, PySpark, and Apache Airflow communities.
