"""Tests for provenance-preserving AnnData gene annotation."""

import builtins
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import src.extract.gene_annotations as gene_annotations_module
from src.extract.gene_annotations import (
    GeneAnnotationAudit,
    annotate_gene_names,
    clean_gene_symbols,
    set_gene_symbols_from_feature_name,
)


SOURCE_IDS = tuple(f"ENSG{i:03d}" for i in range(8))
RAW_SYMBOLS = (
    " ALB ",
    "PECAM1",
    "PECAM1",
    None,
    "",
    "   ",
    "KRT19",
    pd.NA,
)
EXPECTED_BASE_LABELS = (
    "ALB",
    "PECAM1",
    "PECAM1",
    "ENSG003",
    "ENSG004",
    "ENSG005",
    "KRT19",
    "ENSG007",
)
EXPECTED_FINAL_NAMES = (
    "ALB",
    "PECAM1",
    "PECAM1-1",
    "ENSG003",
    "ENSG004",
    "ENSG005",
    "KRT19",
    "ENSG007",
)


@pytest.fixture
def spatial_adata() -> AnnData:
    """Build an annotated-input candidate with sparse spatial structures."""
    values = np.arange(1, 25, dtype=np.float64).reshape(3, 8)
    obs = pd.DataFrame(
        {
            "sample": ["liver-a", "liver-a", "liver-a"],
            "region": ["portal", "mid", "central"],
        },
        index=["spot-3", "spot-1", "spot-2"],
    )
    var = pd.DataFrame(
        {
            "feature_name": pd.Series(RAW_SYMBOLS, dtype="object").array,
            "source_rank": list(range(8)),
        },
        index=SOURCE_IDS,
    )
    adata = AnnData(X=sparse.csr_matrix(values), obs=obs, var=var)
    adata.obsm["spatial"] = np.array(
        [[30.0, 31.0], [10.0, 11.0], [20.0, 21.0]]
    )
    adata.layers["counts"] = sparse.csr_matrix(values * 2)
    adata.obsp["spot_graph"] = sparse.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )
    adata.uns["spatial_metadata"] = {
        "coordinate_units": "pixels",
        "orientation": "source-defined",
    }
    return adata


def assert_sparse_equal(left: object, right: object) -> None:
    assert sparse.issparse(left)
    assert sparse.issparse(right)
    assert left.shape == right.shape
    assert (left != right).nnz == 0


def test_clean_gene_symbols_strips_strings_and_preserves_case() -> None:
    raw = pd.Series(
        ["  Alb  ", "PECAM1", None, "", " \t "],
        index=["a", "b", "c", "d", "e"],
        name="feature_name",
    )

    cleaned = clean_gene_symbols(raw)

    assert cleaned.index.equals(raw.index)
    assert cleaned.name == "feature_name"
    assert cleaned.tolist()[:2] == ["Alb", "PECAM1"]
    assert cleaned.isna().tolist() == [False, False, True, True, True]


def test_clean_gene_symbols_does_not_create_literal_missing_strings() -> None:
    cleaned = clean_gene_symbols([None, np.nan, pd.NA, "", "   "])

    assert cleaned.isna().all()
    assert not {"nan", "None", "<NA>"} & set(cleaned.dropna())


def test_clean_gene_symbols_rejects_non_string_values() -> None:
    with pytest.raises(TypeError, match="strings or null.*position\\(s\\): 1"):
        clean_gene_symbols(["ALB", 42])


def test_original_feature_identifiers_are_preserved(
    spatial_adata: AnnData,
) -> None:
    annotated, audit = annotate_gene_names(spatial_adata)

    assert annotated.var["ensembl_id"].tolist() == list(SOURCE_IDS)
    assert audit.preserved_identifier_column == "ensembl_id"


def test_valid_symbols_and_whitespace_cleaning_form_final_labels(
    spatial_adata: AnnData,
) -> None:
    annotated, _ = annotate_gene_names(spatial_adata)

    assert annotated.var_names.tolist()[:3] == [
        "ALB",
        "PECAM1",
        "PECAM1-1",
    ]
    assert annotated.var["gene_symbol"].tolist()[:3] == [
        "ALB",
        "PECAM1",
        "PECAM1",
    ]


@pytest.mark.parametrize("position", [3, 4, 5, 7])
def test_missing_symbols_fall_back_to_source_identifiers(
    spatial_adata: AnnData,
    position: int,
) -> None:
    annotated, _ = annotate_gene_names(spatial_adata)

    assert pd.isna(annotated.var["gene_symbol"].iloc[position])
    assert annotated.var_names[position] == SOURCE_IDS[position]
    assert annotated.var["var_name_base"].iloc[position] == SOURCE_IDS[position]


def test_missing_symbols_never_become_literal_strings(
    spatial_adata: AnnData,
) -> None:
    annotated, _ = annotate_gene_names(spatial_adata)

    assert not {"nan", "None", "<NA>"} & set(annotated.var_names)
    assert annotated.var["gene_symbol"].isna().sum() == 4


def test_duplicate_symbols_are_unique_without_feature_loss(
    spatial_adata: AnnData,
) -> None:
    annotated, audit = annotate_gene_names(spatial_adata)

    assert annotated.var_names.tolist() == list(EXPECTED_FINAL_NAMES)
    assert annotated.n_vars == spatial_adata.n_vars == 8
    assert audit.duplicated_symbols == ("PECAM1",)
    assert audit.distinct_duplicated_symbol_count == 1
    assert audit.duplicated_symbol_feature_count == 2


def test_duplicate_features_are_not_aggregated(
    spatial_adata: AnnData,
) -> None:
    annotated, _ = annotate_gene_names(spatial_adata)

    assert annotated.var_names[1:3].tolist() == ["PECAM1", "PECAM1-1"]
    assert_sparse_equal(annotated.X[:, 1:3], spatial_adata.X[:, 1:3])


def test_cleaned_symbol_and_base_label_provenance_remain_unsuffixed(
    spatial_adata: AnnData,
) -> None:
    annotated, _ = annotate_gene_names(spatial_adata)

    pd.testing.assert_series_equal(
        annotated.var["feature_name"].reset_index(drop=True),
        spatial_adata.var["feature_name"].reset_index(drop=True),
    )
    assert annotated.var["gene_symbol"].tolist()[1:3] == ["PECAM1", "PECAM1"]
    assert annotated.var["var_name_base"].tolist() == list(EXPECTED_BASE_LABELS)


def test_variable_and_observation_order_are_preserved(
    spatial_adata: AnnData,
) -> None:
    annotated, audit = annotate_gene_names(spatial_adata)

    assert annotated.var["ensembl_id"].tolist() == list(SOURCE_IDS)
    assert annotated.obs_names.tolist() == ["spot-3", "spot-1", "spot-2"]
    assert audit.variable_order_preserved is True
    assert audit.observation_order_preserved is True


def test_matrix_dimensions_and_expression_values_are_unchanged(
    spatial_adata: AnnData,
) -> None:
    original_x = spatial_adata.X.copy()

    annotated, audit = annotate_gene_names(spatial_adata)

    assert annotated.shape == spatial_adata.shape == (3, 8)
    assert_sparse_equal(annotated.X, original_x)
    assert audit.input_spot_count == audit.output_spot_count == 3
    assert audit.input_variable_count == audit.output_variable_count == 8


def test_sparse_input_remains_sparse(spatial_adata: AnnData) -> None:
    annotated, audit = annotate_gene_names(spatial_adata)

    assert sparse.issparse(annotated.X)
    assert audit.input_matrix_was_sparse is True
    assert audit.output_matrix_is_sparse is True


def test_obs_and_var_remain_aligned_with_expression(
    spatial_adata: AnnData,
) -> None:
    annotated, _ = annotate_gene_names(spatial_adata)

    assert annotated.X.shape[0] == len(annotated.obs)
    assert annotated.X.shape[1] == len(annotated.var)
    assert annotated.var["source_rank"].tolist() == list(range(8))
    pd.testing.assert_frame_equal(annotated.obs, spatial_adata.obs)


def test_spatial_layers_uns_and_obsp_are_preserved(
    spatial_adata: AnnData,
) -> None:
    original_spatial = spatial_adata.obsm["spatial"].copy()
    original_counts = spatial_adata.layers["counts"].copy()
    original_graph = spatial_adata.obsp["spot_graph"].copy()
    original_uns = deepcopy(spatial_adata.uns)

    annotated, _ = annotate_gene_names(spatial_adata)

    np.testing.assert_array_equal(annotated.obsm["spatial"], original_spatial)
    assert_sparse_equal(annotated.layers["counts"], original_counts)
    assert_sparse_equal(annotated.obsp["spot_graph"], original_graph)
    assert annotated.uns == original_uns


def test_copy_true_leaves_source_unchanged(spatial_adata: AnnData) -> None:
    original = spatial_adata.copy()

    annotated, audit = annotate_gene_names(spatial_adata)

    assert annotated is not spatial_adata
    assert audit.operated_on_copy is True
    assert spatial_adata.var_names.equals(original.var_names)
    pd.testing.assert_frame_equal(spatial_adata.var, original.var)
    pd.testing.assert_frame_equal(spatial_adata.obs, original.obs)
    assert_sparse_equal(spatial_adata.X, original.X)
    assert "ensembl_id" not in spatial_adata.var


def test_copy_false_modifies_and_returns_source(spatial_adata: AnnData) -> None:
    annotated, audit = annotate_gene_names(spatial_adata, copy=False)

    assert annotated is spatial_adata
    assert annotated.var_names.tolist() == list(EXPECTED_FINAL_NAMES)
    assert annotated.var["ensembl_id"].tolist() == list(SOURCE_IDS)
    assert audit.operated_on_copy is False


def test_missing_gene_symbol_column_raises(spatial_adata: AnnData) -> None:
    with pytest.raises(KeyError, match="missing_symbol.*missing from adata.var"):
        annotate_gene_names(
            spatial_adata,
            gene_symbol_column="missing_symbol",
        )


def test_duplicate_source_var_names_raise(spatial_adata: AnnData) -> None:
    spatial_adata.var_names = [
        "ENSG000",
        "ENSG001",
        "ENSG001",
        "ENSG003",
        "ENSG004",
        "ENSG005",
        "ENSG006",
        "ENSG007",
    ]

    with pytest.raises(ValueError, match="var_names must be unique.*ENSG001"):
        annotate_gene_names(spatial_adata)


@pytest.mark.parametrize("invalid_id", ["", "   "])
def test_empty_source_identifiers_raise(
    spatial_adata: AnnData,
    invalid_id: str,
) -> None:
    names = list(SOURCE_IDS)
    names[2] = invalid_id
    spatial_adata.var_names = names

    with pytest.raises(ValueError, match="var_names must be non-empty"):
        annotate_gene_names(spatial_adata)


def test_incompatible_existing_identifier_column_raises_without_mutation(
    spatial_adata: AnnData,
) -> None:
    spatial_adata.var["ensembl_id"] = list(SOURCE_IDS[:-1]) + ["WRONG"]
    original_var = spatial_adata.var.copy(deep=True)

    with pytest.raises(
        ValueError,
        match="does not exactly match current adata.var_names",
    ):
        annotate_gene_names(spatial_adata, copy=False)

    pd.testing.assert_frame_equal(spatial_adata.var, original_var)
    assert spatial_adata.var_names.tolist() == list(SOURCE_IDS)


def test_identical_existing_identifier_column_is_allowed(
    spatial_adata: AnnData,
) -> None:
    spatial_adata.var["ensembl_id"] = list(SOURCE_IDS)

    annotated, _ = annotate_gene_names(spatial_adata)

    assert annotated.var["ensembl_id"].tolist() == list(SOURCE_IDS)
    assert spatial_adata.var["ensembl_id"].tolist() == list(SOURCE_IDS)


@pytest.mark.parametrize(
    ("argument_name", "invalid_value"),
    [
        ("gene_symbol_column", ""),
        ("ensembl_id_column", "   "),
        ("cleaned_symbol_column", None),
        ("final_label_column", 42),
    ],
)
def test_invalid_column_names_raise_without_mutation(
    spatial_adata: AnnData,
    argument_name: str,
    invalid_value: object,
) -> None:
    original_var = spatial_adata.var.copy(deep=True)

    with pytest.raises(ValueError, match="non-empty strings"):
        annotate_gene_names(
            spatial_adata,
            copy=False,
            **{argument_name: invalid_value},
        )

    pd.testing.assert_frame_equal(spatial_adata.var, original_var)


def test_destination_columns_must_not_overwrite_source_column(
    spatial_adata: AnnData,
) -> None:
    with pytest.raises(ValueError, match="pairwise distinct.*feature_name"):
        annotate_gene_names(
            spatial_adata,
            cleaned_symbol_column="feature_name",
        )


def test_custom_columns_and_uniqueness_separator(
    spatial_adata: AnnData,
) -> None:
    spatial_adata.var["symbol_source"] = spatial_adata.var["feature_name"].copy()

    annotated, audit = annotate_gene_names(
        spatial_adata,
        gene_symbol_column="symbol_source",
        ensembl_id_column="source_id",
        cleaned_symbol_column="symbol_clean",
        final_label_column="label_base",
        uniqueness_separator="__",
    )

    assert annotated.var_names[1:3].tolist() == ["PECAM1", "PECAM1__1"]
    assert annotated.var["source_id"].tolist() == list(SOURCE_IDS)
    assert annotated.var["symbol_clean"].tolist()[1:3] == ["PECAM1", "PECAM1"]
    assert annotated.var["label_base"].tolist() == list(EXPECTED_BASE_LABELS)
    assert audit.source_gene_symbol_column == "symbol_source"
    assert audit.preserved_identifier_column == "source_id"
    assert audit.cleaned_symbol_column == "symbol_clean"
    assert audit.final_label_provenance_column == "label_base"


def test_audit_counts_are_exact(spatial_adata: AnnData) -> None:
    _, audit = annotate_gene_names(spatial_adata)

    assert audit.valid_cleaned_symbol_count == 4
    assert audit.missing_symbol_count == 4
    assert audit.identifier_fallback_count == 4
    assert audit.unique_cleaned_symbol_count == 3
    assert audit.distinct_duplicated_symbol_count == 1
    assert audit.duplicated_symbol_feature_count == 2
    assert audit.unique_final_var_name_count == 8
    assert audit.input_var_names_were_unique is True
    assert audit.final_var_names_are_unique is True


def test_final_var_names_are_unique(spatial_adata: AnnData) -> None:
    annotated, _ = annotate_gene_names(spatial_adata)

    assert annotated.var_names.is_unique
    assert annotated.n_vars == len(annotated.var_names)


def test_compatibility_wrapper_uses_the_new_annotation_api(
    spatial_adata: AnnData,
) -> None:
    annotated, audit = set_gene_symbols_from_feature_name(spatial_adata)

    assert annotated.var_names.tolist() == list(EXPECTED_FINAL_NAMES)
    assert audit.source_gene_symbol_column == "feature_name"


def test_invalid_anndata_input_raises() -> None:
    with pytest.raises(TypeError, match="Expected an AnnData object"):
        annotate_gene_names(object())  # type: ignore[arg-type]


def test_audit_is_immutable(spatial_adata: AnnData) -> None:
    _, audit = annotate_gene_names(spatial_adata)

    with pytest.raises(FrozenInstanceError):
        audit.output_variable_count = 9  # type: ignore[misc]
    assert isinstance(audit, GeneAnnotationAudit)


def test_module_execution_performs_no_file_io_or_external_initialization(
    monkeypatch,
) -> None:
    module_path = Path(gene_annotations_module.__file__)
    source = module_path.read_text(encoding="utf-8")
    code = compile(source, str(module_path), "exec")
    original_import = builtins.__import__
    allowed_import_roots = {
        "anndata",
        "collections",
        "dataclasses",
        "pandas",
        "scipy",
    }

    def reject_file_io(*args, **kwargs):
        raise AssertionError("gene_annotations attempted file I/O")

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", maxsplit=1)[0] not in allowed_import_roots:
            raise AssertionError(
                f"gene_annotations imported prohibited module {name}"
            )
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as guarded_context:
        guarded_context.setattr(builtins, "open", reject_file_io)
        guarded_context.setattr(Path, "open", reject_file_io)
        guarded_context.setattr(Path, "read_text", reject_file_io)
        guarded_context.setattr(Path, "write_text", reject_file_io)
        guarded_context.setattr(builtins, "__import__", restricted_import)

        isolated_namespace = {
            "__builtins__": builtins.__dict__,
            "__file__": str(module_path),
            "__name__": gene_annotations_module.__name__,
        }
        exec(code, isolated_namespace)

    assert isolated_namespace["clean_gene_symbols"]
    assert isolated_namespace["annotate_gene_names"]
