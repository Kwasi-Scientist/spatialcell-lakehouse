"""Tests for AnnData gene-symbol annotation."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from src.extract.gene_annotations import (
    GeneAnnotationAudit,
    set_gene_symbols_from_feature_name,
)


def make_adata(
    symbols: list[object],
    *,
    var_names: list[str] | None = None,
) -> AnnData:
    """Build a tiny sparse spatial AnnData fixture."""
    n_vars = len(symbols)
    values = np.arange(1, (2 * n_vars) + 1, dtype=np.float64).reshape(2, n_vars)
    obs = pd.DataFrame(
        {"sample": ["left", "right"]},
        index=["spot-1", "spot-2"],
    )
    var = pd.DataFrame(
        {"feature_name": pd.Series(symbols, dtype="object").array},
        index=var_names or [f"ENSG{i:03d}" for i in range(n_vars)],
    )
    adata = AnnData(X=sparse.csr_matrix(values), obs=obs, var=var)
    adata.obsm["spatial"] = np.array([[1.0, 2.0], [3.0, 4.0]])
    adata.layers["counts"] = sparse.csr_matrix(values)
    return adata


def assert_sparse_equal(left: object, right: object) -> None:
    assert sparse.issparse(left)
    assert sparse.issparse(right)
    assert (left != right).nnz == 0


def test_successful_annotation_with_unique_symbols() -> None:
    adata = make_adata(["ALB", "KRT19", "PECAM1"])

    annotated, audit = set_gene_symbols_from_feature_name(adata)

    assert annotated.var_names.tolist() == ["ALB", "KRT19", "PECAM1"]
    assert annotated.var["ensembl_id"].tolist() == [
        "ENSG000",
        "ENSG001",
        "ENSG002",
    ]
    assert annotated.var["gene_symbol"].tolist() == ["ALB", "KRT19", "PECAM1"]
    assert audit.total_features == 3
    assert audit.symbols_present == 3
    assert audit.symbols_missing == 0
    assert audit.unique_canonical_symbols == 3
    assert audit.final_var_names_unique is True


def test_whitespace_trimming_and_literal_missing_like_strings() -> None:
    adata = make_adata(["  ALB  ", " NA ", "None", " nan "])

    annotated, _ = set_gene_symbols_from_feature_name(adata)

    assert annotated.var["gene_symbol"].tolist() == ["ALB", "NA", "None", "nan"]
    assert annotated.var_names.tolist() == ["ALB", "NA", "None", "nan"]


def test_missing_symbols_use_ensembl_fallbacks() -> None:
    adata = make_adata(["ALB", None, "   ", pd.NA])

    annotated, audit = set_gene_symbols_from_feature_name(adata)

    assert annotated.var_names.tolist() == [
        "ALB",
        "ENSG001",
        "ENSG002",
        "ENSG003",
    ]
    assert annotated.var["gene_symbol"].isna().tolist() == [
        False,
        True,
        True,
        True,
    ]
    assert audit.symbols_present == 1
    assert audit.symbols_missing == 3
    assert audit.ensembl_fallbacks_used == 3


def test_duplicate_symbols_have_deterministic_names_and_collision_audit() -> None:
    adata = make_adata(["DUP", "ALB", "DUP", "DUP", None])

    annotated, audit = set_gene_symbols_from_feature_name(adata)

    assert annotated.var_names.tolist() == [
        "DUP",
        "ALB",
        "DUP-1",
        "DUP-2",
        "ENSG004",
    ]
    assert annotated.var["gene_symbol"].tolist()[:4] == [
        "DUP",
        "ALB",
        "DUP",
        "DUP",
    ]
    assert audit.canonical_symbols_with_collisions == ("DUP",)
    assert audit.duplicated_canonical_symbol_names == 1
    assert audit.features_in_symbol_collisions == 3
    assert audit.unique_canonical_symbols == 2
    assert audit.duplicated_ensembl_identifiers == 0


@pytest.mark.parametrize(
    (
        "symbol_column",
        "ensembl_id_column",
        "canonical_symbol_column",
        "final_label_column",
    ),
    [
        ("provided_symbol", "stable_id", "canonical", "final_name"),
        ("source_symbol", "original_id", "clean_symbol", "unique_label"),
    ],
)
def test_custom_columns_and_uniqueness_separator(
    symbol_column: str,
    ensembl_id_column: str,
    canonical_symbol_column: str,
    final_label_column: str,
) -> None:
    adata = make_adata(["DUP", "DUP"])
    adata.var[symbol_column] = adata.var.pop("feature_name")

    annotated, audit = set_gene_symbols_from_feature_name(
        adata,
        symbol_column=symbol_column,
        ensembl_id_column=ensembl_id_column,
        canonical_symbol_column=canonical_symbol_column,
        final_label_column=final_label_column,
        uniqueness_separator="__",
    )

    assert annotated.var_names.tolist() == ["DUP", "DUP__1"]
    assert annotated.var[ensembl_id_column].tolist() == ["ENSG000", "ENSG001"]
    assert annotated.var[canonical_symbol_column].tolist() == ["DUP", "DUP"]
    assert annotated.var[final_label_column].tolist() == ["DUP", "DUP__1"]
    assert audit.annotation_source_column == symbol_column


@pytest.mark.parametrize(
    ("argument_name", "invalid_name"),
    [
        (argument_name, invalid_name)
        for argument_name in (
            "symbol_column",
            "ensembl_id_column",
            "canonical_symbol_column",
            "final_label_column",
        )
        for invalid_name in ("", "   ")
    ],
)
def test_metadata_column_names_must_be_non_empty_strings_without_mutation(
    argument_name: str,
    invalid_name: str,
) -> None:
    adata = make_adata([" ALB ", "KRT19"])
    original_var = adata.var.copy(deep=True)
    original_names = adata.var_names.copy()

    with pytest.raises(
        ValueError,
        match=rf"non-empty strings.*{argument_name}",
    ):
        set_gene_symbols_from_feature_name(
            adata,
            copy=False,
            **{argument_name: invalid_name},
        )

    pd.testing.assert_frame_equal(adata.var, original_var)
    assert adata.var_names.equals(original_names)


@pytest.mark.parametrize(
    ("first_argument", "second_argument"),
    [
        ("symbol_column", "ensembl_id_column"),
        ("symbol_column", "canonical_symbol_column"),
        ("symbol_column", "final_label_column"),
        ("ensembl_id_column", "canonical_symbol_column"),
        ("ensembl_id_column", "final_label_column"),
        ("canonical_symbol_column", "final_label_column"),
    ],
)
def test_metadata_column_names_must_be_pairwise_distinct_without_mutation(
    first_argument: str,
    second_argument: str,
) -> None:
    adata = make_adata([" ALB ", "KRT19"])
    original_var = adata.var.copy(deep=True)
    original_names = adata.var_names.copy()
    column_names = {
        "symbol_column": "feature_name",
        "ensembl_id_column": "ensembl_id",
        "canonical_symbol_column": "gene_symbol",
        "final_label_column": "var_name_unique",
    }
    column_names[first_argument] = "conflicting_name"
    column_names[second_argument] = "conflicting_name"

    with pytest.raises(
        ValueError,
        match=(
            rf"pairwise distinct.*'conflicting_name'.*"
            rf"{first_argument}.*{second_argument}"
        ),
    ):
        set_gene_symbols_from_feature_name(
            adata,
            copy=False,
            **column_names,
        )

    pd.testing.assert_frame_equal(adata.var, original_var)
    assert adata.var_names.equals(original_names)


def test_missing_symbol_column_fails() -> None:
    adata = make_adata(["ALB"])
    del adata.var["feature_name"]

    with pytest.raises(KeyError, match="missing from adata.var"):
        set_gene_symbols_from_feature_name(adata)


@pytest.mark.parametrize("invalid_id", ["", "   "])
def test_empty_or_whitespace_original_feature_id_fails(invalid_id: str) -> None:
    adata = make_adata(["ALB", "KRT19"], var_names=["ENSG001", invalid_id])

    with pytest.raises(ValueError, match="non-null and non-empty"):
        set_gene_symbols_from_feature_name(adata)


def test_duplicate_original_ensembl_id_fails() -> None:
    with pytest.warns(UserWarning, match="Variable names are not unique"):
        adata = make_adata(
            ["ALB", "KRT19"],
            var_names=["ENSG001", "ENSG001"],
        )
        with pytest.raises(ValueError, match="duplicated identifier"):
            set_gene_symbols_from_feature_name(adata)


def test_copy_true_leaves_original_unchanged() -> None:
    adata = make_adata([" ALB ", "DUP", "DUP"])
    original_var = adata.var.copy(deep=True)
    original_names = adata.var_names.copy()

    annotated, audit = set_gene_symbols_from_feature_name(adata)

    assert annotated is not adata
    pd.testing.assert_frame_equal(adata.var, original_var)
    assert adata.var_names.equals(original_names)
    assert audit.operated_on_copy is True


def test_copy_false_modifies_original() -> None:
    adata = make_adata([" ALB ", "KRT19"])

    annotated, audit = set_gene_symbols_from_feature_name(adata, copy=False)

    assert annotated is adata
    assert adata.var_names.tolist() == ["ALB", "KRT19"]
    assert {"ensembl_id", "gene_symbol", "var_name_unique"} <= set(adata.var)
    assert audit.operated_on_copy is False


def test_repeated_execution_is_stable_and_preserves_ensembl_ids() -> None:
    adata = make_adata(["DUP", "DUP", None])
    first, first_audit = set_gene_symbols_from_feature_name(adata)

    second, second_audit = set_gene_symbols_from_feature_name(first)

    assert second.var["ensembl_id"].tolist() == [
        "ENSG000",
        "ENSG001",
        "ENSG002",
    ]
    assert second.var["ensembl_id"].equals(first.var["ensembl_id"])
    assert second.var["gene_symbol"].equals(first.var["gene_symbol"])
    assert second.var["var_name_unique"].equals(first.var["var_name_unique"])
    assert second.var_names.equals(first.var_names)
    assert second_audit == first_audit


def test_existing_invalid_preserved_ensembl_ids_fail() -> None:
    adata = make_adata(["ALB", "KRT19"])
    adata.var["ensembl_id"] = pd.Series(
        ["ENSG001", pd.NA],
        index=adata.var.index,
        dtype="string",
    )

    with pytest.raises(ValueError, match="non-null and non-empty"):
        set_gene_symbols_from_feature_name(adata)


def test_sparse_expression_and_anndata_structures_are_unchanged() -> None:
    adata = make_adata(["ALB", None, "DUP"])
    original_x = adata.X.copy()
    original_counts = adata.layers["counts"].copy()
    original_obs = adata.obs.copy(deep=True)
    original_spatial = adata.obsm["spatial"].copy()
    original_shape = adata.shape

    annotated, _ = set_gene_symbols_from_feature_name(adata)

    assert annotated.shape == original_shape
    assert_sparse_equal(annotated.X, original_x)
    assert_sparse_equal(annotated.layers["counts"], original_counts)
    pd.testing.assert_frame_equal(annotated.obs, original_obs)
    np.testing.assert_array_equal(annotated.obsm["spatial"], original_spatial)


def test_final_names_and_metadata_remain_aligned() -> None:
    adata = make_adata(["DUP", "DUP", None, "ALB"])

    annotated, _ = set_gene_symbols_from_feature_name(adata)

    assert annotated.var_names.tolist() == annotated.var[
        "var_name_unique"
    ].tolist()
    assert annotated.n_vars == len(annotated.var) == len(annotated.var_names)


def test_input_must_be_anndata() -> None:
    with pytest.raises(TypeError, match="Expected an AnnData object"):
        set_gene_symbols_from_feature_name(object())  # type: ignore[arg-type]


def test_audit_is_immutable() -> None:
    _, audit = set_gene_symbols_from_feature_name(make_adata(["ALB"]))

    with pytest.raises(FrozenInstanceError):
        audit.total_features = 2  # type: ignore[misc]
    assert isinstance(audit, GeneAnnotationAudit)
