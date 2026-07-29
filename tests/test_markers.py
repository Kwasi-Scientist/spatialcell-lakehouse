"""Tests for the curated human liver marker registry."""

import builtins
from pathlib import Path

import src.markers as marker_module
from src.markers import (
    CELLTYPE_MARKER_GENES,
    MARKER_PANEL_METADATA,
    SPECIFIC_CELLTYPE_MARKER_GENES,
    SPECIFIC_TO_GENERAL_TYPE,
)


MARKER_PANELS = (
    CELLTYPE_MARKER_GENES,
    SPECIFIC_CELLTYPE_MARKER_GENES,
)


def test_general_marker_registry_is_not_empty() -> None:
    assert CELLTYPE_MARKER_GENES


def test_specific_marker_registry_is_not_empty() -> None:
    assert SPECIFIC_CELLTYPE_MARKER_GENES


def test_general_cell_type_keys_are_non_empty_clean_strings() -> None:
    for cell_type in CELLTYPE_MARKER_GENES:
        assert isinstance(cell_type, str)
        assert cell_type
        assert cell_type == cell_type.strip()


def test_specific_cell_type_keys_are_non_empty_clean_strings() -> None:
    for cell_type in SPECIFIC_CELLTYPE_MARKER_GENES:
        assert isinstance(cell_type, str)
        assert cell_type
        assert cell_type == cell_type.strip()


def test_marker_genes_are_non_empty_clean_strings() -> None:
    for panel in MARKER_PANELS:
        for markers in panel.values():
            assert markers
            for gene in markers:
                assert isinstance(gene, str)
                assert gene
                assert gene == gene.strip()


def test_marker_collections_have_no_exact_duplicates() -> None:
    for panel in MARKER_PANELS:
        for markers in panel.values():
            assert len(markers) == len(set(markers))


def test_every_specific_type_has_an_explicit_general_mapping() -> None:
    assert set(SPECIFIC_CELLTYPE_MARKER_GENES) == set(SPECIFIC_TO_GENERAL_TYPE)


def test_every_mapped_general_type_exists() -> None:
    assert set(SPECIFIC_TO_GENERAL_TYPE.values()) <= set(CELLTYPE_MARKER_GENES)


def test_endothelial_general_category_is_retained() -> None:
    assert "Endothelial" in CELLTYPE_MARKER_GENES


def test_endothelial_specific_categories_are_retained() -> None:
    assert {"Arterial", "cvEndo", "cvLSEC", "ppLSEC"} <= set(
        SPECIFIC_CELLTYPE_MARKER_GENES
    )


def test_major_liver_categories_from_the_audited_source_are_retained() -> None:
    assert {"Hepatocyte", "Macrophage/Myeloid"} <= set(CELLTYPE_MARKER_GENES)
    assert {"Cholangio", "Kupffer", "P-Hepato"} <= set(
        SPECIFIC_CELLTYPE_MARKER_GENES
    )


def test_marker_collections_are_immutable_tuples() -> None:
    for panel in MARKER_PANELS:
        assert all(isinstance(markers, tuple) for markers in panel.values())


def test_metadata_contains_provenance_and_interpretation_fields() -> None:
    required_fields = {
        "panel_name",
        "organism",
        "tissue",
        "source_type",
        "source_publication",
        "source_table_or_worksheet",
        "curation_method",
        "curator",
        "curation_date",
        "biological_interpretation_notes",
    }

    assert required_fields <= set(MARKER_PANEL_METADATA)
    assert all(MARKER_PANEL_METADATA[field] for field in required_fields)


def test_module_execution_performs_no_file_io_or_external_imports(
    monkeypatch,
) -> None:
    module_path = Path(marker_module.__file__)
    source = module_path.read_text(encoding="utf-8")
    code = compile(source, str(module_path), "exec")

    def reject_side_effect(*args, **kwargs):
        raise AssertionError("Marker module attempted file I/O or an import")

    with monkeypatch.context() as guarded_context:
        guarded_context.setattr(builtins, "open", reject_side_effect)
        guarded_context.setattr(Path, "open", reject_side_effect)
        guarded_context.setattr(Path, "read_text", reject_side_effect)
        guarded_context.setattr(Path, "write_text", reject_side_effect)
        guarded_context.setattr(builtins, "__import__", reject_side_effect)

        isolated_namespace = {
            "__builtins__": builtins.__dict__,
            "__file__": str(module_path),
            "__name__": "isolated_marker_registry",
        }
        exec(code, isolated_namespace)

    assert isolated_namespace["CELLTYPE_MARKER_GENES"]
    assert isolated_namespace["SPECIFIC_CELLTYPE_MARKER_GENES"]
