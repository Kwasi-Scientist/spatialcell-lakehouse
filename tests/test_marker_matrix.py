"""Tests for deterministic sparse marker-matrix extraction."""

import builtins
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import src.extract.marker_matrix as marker_matrix_module
from src.extract.marker_matrix import (
    MarkerCollectionAudit,
    MarkerExtractionAudit,
    collect_marker_genes,
    extract_marker_matrix,
    match_marker_genes,
)


GENERAL_MARKERS = {
    "Hepatocyte": ("ALB", "MISSING", "KRT19"),
    "Shared compartment": ("PECAM1", "ALB"),
}
SPECIFIC_MARKERS = {
    "Specific immune": ("CD79A", "PECAM1"),
    "Specific ductal": ("KRT19",),
}
EXPECTED_REQUESTED = ("ALB", "MISSING", "KRT19", "PECAM1", "CD79A")
EXPECTED_MATCHED = ("ALB", "KRT19", "PECAM1", "CD79A")


@pytest.fixture
def spatial_adata() -> AnnData:
    """Build a tiny sparse, annotated Visium-like AnnData object."""
    values = np.array(
        [
            [100.0, 1.0, 2.0, 3.0, 4.0],
            [200.0, 5.0, 6.0, 7.0, 8.0],
            [300.0, 9.0, 10.0, 11.0, 12.0],
        ]
    )
    obs = pd.DataFrame(
        {"region": ["portal", "mid", "central"]},
        index=["spot-3", "spot-1", "spot-2"],
    )
    var_names = ["NONMARKER", "KRT19", "ALB", "PECAM1", "CD79A"]
    var = pd.DataFrame(
        {
            "ensembl_id": [f"ENSG{i:03d}" for i in range(len(var_names))],
            "gene_symbol": var_names,
            "feature_source": ["dataset"] * len(var_names),
        },
        index=var_names,
    )
    adata = AnnData(X=sparse.csr_matrix(values), obs=obs, var=var)
    adata.obsm["spatial"] = np.array(
        [[30.0, 31.0], [10.0, 11.0], [20.0, 21.0]]
    )
    adata.layers["counts"] = sparse.csr_matrix(values)
    return adata


def assert_sparse_equal(left: object, right: object) -> None:
    assert sparse.issparse(left)
    assert sparse.issparse(right)
    assert left.shape == right.shape
    assert (left != right).nnz == 0


def collect_test_markers(
    *,
    include_general: bool = True,
    include_specific: bool = True,
) -> tuple[tuple[str, ...], MarkerCollectionAudit]:
    return collect_marker_genes(
        GENERAL_MARKERS,
        SPECIFIC_MARKERS,
        include_general=include_general,
        include_specific=include_specific,
    )


def extract_test_markers(
    adata: AnnData,
    *,
    include_general: bool = True,
    include_specific: bool = True,
) -> tuple[AnnData, MarkerExtractionAudit]:
    return extract_marker_matrix(
        adata,
        GENERAL_MARKERS,
        SPECIFIC_MARKERS,
        include_general=include_general,
        include_specific=include_specific,
    )


def test_collection_combines_general_and_specific_panels() -> None:
    markers, audit = collect_test_markers()

    assert markers == EXPECTED_REQUESTED
    assert audit.included_general_marker_groups == (
        "Hepatocyte",
        "Shared compartment",
    )
    assert audit.included_specific_marker_groups == (
        "Specific immune",
        "Specific ductal",
    )


def test_general_only_extraction_works(spatial_adata: AnnData) -> None:
    extracted, audit = extract_test_markers(
        spatial_adata,
        include_specific=False,
    )

    assert extracted.var_names.tolist() == ["ALB", "KRT19", "PECAM1"]
    assert audit.matched_markers == ("ALB", "KRT19", "PECAM1")
    assert audit.unmatched_markers == ("MISSING",)
    assert audit.included_specific_marker_groups == ()


def test_specific_only_extraction_works(spatial_adata: AnnData) -> None:
    extracted, audit = extract_test_markers(
        spatial_adata,
        include_general=False,
    )

    assert extracted.var_names.tolist() == ["CD79A", "PECAM1", "KRT19"]
    assert audit.matched_markers == ("CD79A", "PECAM1", "KRT19")
    assert audit.unmatched_markers == ()
    assert audit.included_general_marker_groups == ()


def test_disabling_both_panels_raises(spatial_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="At least one marker panel"):
        extract_test_markers(
            spatial_adata,
            include_general=False,
            include_specific=False,
        )


def test_duplicate_markers_are_removed_deterministically() -> None:
    markers, audit = collect_test_markers()

    assert markers == EXPECTED_REQUESTED
    assert audit.requested_marker_count_before_deduplication == 8
    assert audit.unique_requested_marker_count == 5
    assert audit.duplicate_marker_occurrence_count == 3
    assert audit.duplicate_marker_occurrences_removed == (
        "ALB",
        "PECAM1",
        "KRT19",
    )


def test_first_seen_marker_order_is_preserved() -> None:
    markers, _ = collect_test_markers()

    assert markers == ("ALB", "MISSING", "KRT19", "PECAM1", "CD79A")


def test_matched_and_unmatched_markers_are_reported_in_order(
    spatial_adata: AnnData,
) -> None:
    markers, _ = collect_test_markers()

    audit = match_marker_genes(
        markers,
        spatial_adata.var_names,
        general_marker_genes=GENERAL_MARKERS,
        specific_marker_genes=SPECIFIC_MARKERS,
    )

    assert audit.matched_markers == EXPECTED_MATCHED
    assert audit.unmatched_markers == ("MISSING",)
    assert audit.matched_marker_count == 4
    assert audit.unmatched_marker_count == 1


def test_match_percentage_is_calculated_from_unique_requests(
    spatial_adata: AnnData,
) -> None:
    markers, _ = collect_test_markers()

    audit = match_marker_genes(
        markers,
        spatial_adata.var_names,
        general_marker_genes=GENERAL_MARKERS,
        specific_marker_genes=SPECIFIC_MARKERS,
    )

    assert audit.requested_marker_count == 5
    assert audit.match_percentage == pytest.approx(80.0)


def test_markers_in_multiple_groups_are_audited() -> None:
    _, audit = collect_test_markers()

    assert audit.markers_in_multiple_groups == ("ALB", "KRT19", "PECAM1")


def test_extracted_variables_follow_requested_marker_order(
    spatial_adata: AnnData,
) -> None:
    extracted, audit = extract_test_markers(spatial_adata)

    assert extracted.var_names.tolist() == list(EXPECTED_MATCHED)
    assert audit.matched_markers == EXPECTED_MATCHED


def test_spot_order_and_observation_alignment_are_preserved(
    spatial_adata: AnnData,
) -> None:
    extracted, audit = extract_test_markers(spatial_adata)

    assert extracted.obs_names.tolist() == ["spot-3", "spot-1", "spot-2"]
    pd.testing.assert_frame_equal(extracted.obs, spatial_adata.obs)
    assert extracted.X.shape[0] == len(extracted.obs)
    assert audit.input_spot_count == audit.output_spot_count == 3


def test_variable_metadata_remains_aligned_with_expression(
    spatial_adata: AnnData,
) -> None:
    extracted, _ = extract_test_markers(spatial_adata)

    assert extracted.X.shape[1] == len(extracted.var)
    assert extracted.var.index.tolist() == list(EXPECTED_MATCHED)
    assert extracted.var["gene_symbol"].tolist() == list(EXPECTED_MATCHED)


def test_ensembl_identifiers_are_preserved(spatial_adata: AnnData) -> None:
    extracted, _ = extract_test_markers(spatial_adata)

    assert extracted.var["ensembl_id"].tolist() == [
        "ENSG002",
        "ENSG001",
        "ENSG003",
        "ENSG004",
    ]


def test_spatial_coordinates_are_preserved(spatial_adata: AnnData) -> None:
    original_spatial = spatial_adata.obsm["spatial"].copy()

    extracted, _ = extract_test_markers(spatial_adata)

    np.testing.assert_array_equal(extracted.obsm["spatial"], original_spatial)


def test_sparse_input_produces_sparse_output(spatial_adata: AnnData) -> None:
    extracted, audit = extract_test_markers(spatial_adata)
    expected_values = spatial_adata.X[:, [2, 1, 3, 4]]

    assert_sparse_equal(extracted.X, expected_values)
    assert audit.input_matrix_was_sparse is True
    assert audit.output_matrix_is_sparse is True


def test_original_anndata_is_not_mutated(spatial_adata: AnnData) -> None:
    original = spatial_adata.copy()

    extracted, audit = extract_test_markers(spatial_adata)

    assert extracted is not spatial_adata
    assert extracted.is_view is False
    assert audit.operated_on_copy is True
    assert spatial_adata.var_names.equals(original.var_names)
    pd.testing.assert_frame_equal(spatial_adata.obs, original.obs)
    pd.testing.assert_frame_equal(spatial_adata.var, original.var)
    assert_sparse_equal(spatial_adata.X, original.X)
    assert_sparse_equal(spatial_adata.layers["counts"], original.layers["counts"])
    np.testing.assert_array_equal(
        spatial_adata.obsm["spatial"],
        original.obsm["spatial"],
    )


def test_input_marker_dictionaries_are_not_mutated(
    spatial_adata: AnnData,
) -> None:
    general = {
        group: list(markers) for group, markers in GENERAL_MARKERS.items()
    }
    specific = {
        group: list(markers) for group, markers in SPECIFIC_MARKERS.items()
    }
    original_general = deepcopy(general)
    original_specific = deepcopy(specific)

    extract_marker_matrix(spatial_adata, general, specific)

    assert general == original_general
    assert specific == original_specific


def test_non_marker_genes_are_excluded(spatial_adata: AnnData) -> None:
    extracted, _ = extract_test_markers(spatial_adata)

    assert "NONMARKER" not in extracted.var_names
    assert extracted.n_vars == 4


def test_non_unique_var_names_raise(spatial_adata: AnnData) -> None:
    spatial_adata.var_names = [
        "NONMARKER",
        "KRT19",
        "ALB",
        "ALB",
        "CD79A",
    ]

    with pytest.raises(ValueError, match="var_names must be unique"):
        extract_test_markers(spatial_adata)


def test_zero_matched_markers_raise(spatial_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="No requested marker genes matched"):
        extract_marker_matrix(
            spatial_adata,
            {"Unavailable": ("NOT_PRESENT",)},
            {},
            include_specific=False,
        )


def test_matching_is_exact_and_case_sensitive(spatial_adata: AnnData) -> None:
    audit = match_marker_genes(
        ("ALB", "alb"),
        spatial_adata.var_names,
        general_marker_genes={"Test": ("ALB", "alb")},
        specific_marker_genes={},
    )

    assert audit.matched_markers == ("ALB",)
    assert audit.unmatched_markers == ("alb",)


def test_invalid_anndata_input_raises() -> None:
    with pytest.raises(TypeError, match="Expected an AnnData object"):
        extract_marker_matrix(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("general_markers", "error_type", "message"),
    [
        (["ALB"], TypeError, "must be a mapping"),
        ({"": ("ALB",)}, ValueError, "group names must be non-empty"),
        ({"Group": "ALB"}, TypeError, "iterable of marker strings"),
        ({"Group": ()}, ValueError, "at least one marker"),
        ({"Group": ("",)}, ValueError, "empty marker"),
        ({"Group": ("ALB", 3)}, TypeError, "contain only strings"),
    ],
)
def test_invalid_marker_dictionaries_raise(
    general_markers: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        collect_marker_genes(
            general_markers,  # type: ignore[arg-type]
            {},
            include_specific=False,
        )


def test_available_feature_names_must_be_unique() -> None:
    with pytest.raises(ValueError, match="available_var_names must be unique"):
        match_marker_genes(
            ("ALB",),
            ("ALB", "ALB"),
            general_marker_genes={"Test": ("ALB",)},
            specific_marker_genes={},
        )


def test_audit_structures_are_immutable() -> None:
    _, collection_audit = collect_test_markers()

    with pytest.raises(FrozenInstanceError):
        collection_audit.unique_requested_marker_count = 6  # type: ignore[misc]


def test_module_execution_performs_no_file_io_or_pipeline_imports(
    monkeypatch,
) -> None:
    module_path = Path(marker_matrix_module.__file__)
    source = module_path.read_text(encoding="utf-8")
    code = compile(source, str(module_path), "exec")
    original_import = builtins.__import__
    allowed_import_roots = {"anndata", "collections", "dataclasses", "scipy", "src"}

    def reject_file_io(*args, **kwargs):
        raise AssertionError("marker_matrix attempted file I/O")

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", maxsplit=1)[0] not in allowed_import_roots:
            raise AssertionError(f"marker_matrix imported prohibited module {name}")
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
            "__name__": marker_matrix_module.__name__,
        }
        exec(code, isolated_namespace)

    assert isolated_namespace["collect_marker_genes"]
    assert isolated_namespace["match_marker_genes"]
    assert isolated_namespace["extract_marker_matrix"]
