"""Collect and extract curated marker expression from annotated AnnData spots.

Matching is exact and case-sensitive. The extracted values are not interpreted
as raw counts or normalized expression, and marker presence supports enrichment
interpretation rather than definitive cell identity for individual Visium spots.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from anndata import AnnData
from scipy import sparse

from src.markers import CELLTYPE_MARKER_GENES, SPECIFIC_CELLTYPE_MARKER_GENES


MarkerDictionary = Mapping[str, Iterable[str]]
MaterializedMarkerGroups = tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class MarkerCollectionAudit:
    """Immutable summary of marker-panel collection and deduplication."""

    requested_marker_count_before_deduplication: int
    unique_requested_marker_count: int
    duplicate_marker_occurrence_count: int
    duplicate_marker_occurrences_removed: tuple[str, ...]
    markers_in_multiple_groups: tuple[str, ...]
    included_general_marker_groups: tuple[str, ...]
    included_specific_marker_groups: tuple[str, ...]


@dataclass(frozen=True)
class MarkerMatchAudit:
    """Immutable summary of exact marker matching against feature names."""

    requested_marker_count: int
    matched_marker_count: int
    unmatched_marker_count: int
    match_percentage: float
    matched_markers: tuple[str, ...]
    unmatched_markers: tuple[str, ...]
    markers_in_multiple_groups: tuple[str, ...]


@dataclass(frozen=True)
class MarkerExtractionAudit:
    """Immutable marker-extraction metrics without expression-matrix values."""

    requested_marker_count_before_deduplication: int
    unique_requested_marker_count: int
    matched_marker_count: int
    unmatched_marker_count: int
    match_percentage: float
    matched_markers: tuple[str, ...]
    unmatched_markers: tuple[str, ...]
    duplicate_marker_occurrence_count: int
    duplicate_marker_occurrences_removed: tuple[str, ...]
    markers_in_multiple_groups: tuple[str, ...]
    included_general_marker_groups: tuple[str, ...]
    included_specific_marker_groups: tuple[str, ...]
    input_spot_count: int
    input_variable_count: int
    output_spot_count: int
    output_variable_count: int
    input_matrix_was_sparse: bool
    output_matrix_is_sparse: bool
    operated_on_copy: bool


def collect_marker_genes(
    general_marker_genes: MarkerDictionary = CELLTYPE_MARKER_GENES,
    specific_marker_genes: MarkerDictionary = SPECIFIC_CELLTYPE_MARKER_GENES,
    *,
    include_general: bool = True,
    include_specific: bool = True,
) -> tuple[tuple[str, ...], MarkerCollectionAudit]:
    """Collect enabled marker panels in first-seen order without mutation.

    Exact repeated symbols are retained only at their first occurrence. General
    groups are traversed before specific groups, and both group and marker order
    follow the supplied mappings.
    """
    _validate_panel_flags(
        include_general=include_general,
        include_specific=include_specific,
    )
    general_groups = _materialize_marker_groups(
        general_marker_genes,
        panel_name="general_marker_genes",
    )
    specific_groups = _materialize_marker_groups(
        specific_marker_genes,
        panel_name="specific_marker_genes",
    )

    enabled_groups = (
        (general_groups if include_general else ())
        + (specific_groups if include_specific else ())
    )
    requested_count = sum(
        len(markers) for _, markers in enabled_groups
    )
    if requested_count == 0:
        raise ValueError("Enabled marker panels contain no requested markers")

    collected: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for _, markers in enabled_groups:
        for marker in markers:
            if marker in seen:
                duplicates.append(marker)
                continue
            seen.add(marker)
            collected.append(marker)

    markers = tuple(collected)
    multi_group_markers = _find_markers_in_multiple_groups(
        markers,
        general_groups=general_groups if include_general else (),
        specific_groups=specific_groups if include_specific else (),
    )
    audit = MarkerCollectionAudit(
        requested_marker_count_before_deduplication=requested_count,
        unique_requested_marker_count=len(markers),
        duplicate_marker_occurrence_count=len(duplicates),
        duplicate_marker_occurrences_removed=tuple(duplicates),
        markers_in_multiple_groups=multi_group_markers,
        included_general_marker_groups=(
            tuple(group for group, _ in general_groups)
            if include_general
            else ()
        ),
        included_specific_marker_groups=(
            tuple(group for group, _ in specific_groups)
            if include_specific
            else ()
        ),
    )
    return markers, audit


def match_marker_genes(
    requested_markers: Iterable[str],
    available_var_names: Iterable[str],
    *,
    general_marker_genes: MarkerDictionary = CELLTYPE_MARKER_GENES,
    specific_marker_genes: MarkerDictionary = SPECIFIC_CELLTYPE_MARKER_GENES,
) -> MarkerMatchAudit:
    """Match requested markers exactly against unique available feature names."""
    requested = _materialize_symbol_sequence(
        requested_markers,
        argument_name="requested_markers",
        allow_empty=False,
    )
    available = _materialize_symbol_sequence(
        available_var_names,
        argument_name="available_var_names",
        allow_empty=True,
    )
    if len(available) != len(set(available)):
        raise ValueError(
            "available_var_names must be unique for deterministic marker matching"
        )

    general_groups = _materialize_marker_groups(
        general_marker_genes,
        panel_name="general_marker_genes",
    )
    specific_groups = _materialize_marker_groups(
        specific_marker_genes,
        panel_name="specific_marker_genes",
    )
    available_set = set(available)
    matched = tuple(marker for marker in requested if marker in available_set)
    unmatched = tuple(marker for marker in requested if marker not in available_set)
    requested_count = len(requested)

    return MarkerMatchAudit(
        requested_marker_count=requested_count,
        matched_marker_count=len(matched),
        unmatched_marker_count=len(unmatched),
        match_percentage=(len(matched) / requested_count) * 100.0,
        matched_markers=matched,
        unmatched_markers=unmatched,
        markers_in_multiple_groups=_find_markers_in_multiple_groups(
            requested,
            general_groups=general_groups,
            specific_groups=specific_groups,
        ),
    )


def extract_marker_matrix(
    adata: AnnData,
    general_marker_genes: MarkerDictionary = CELLTYPE_MARKER_GENES,
    specific_marker_genes: MarkerDictionary = SPECIFIC_CELLTYPE_MARKER_GENES,
    *,
    include_general: bool = True,
    include_specific: bool = True,
    copy: bool = True,
) -> tuple[AnnData, MarkerExtractionAudit]:
    """Return marker-only AnnData spots and a structured extraction audit.

    The input is never modified. By default, the returned marker subset is an
    independent copy; with ``copy=False`` it is an AnnData view. Explicit
    positional indexing preserves first-seen requested-marker order without
    densifying ``adata.X``.
    """
    _validate_annotated_adata(adata)
    if not isinstance(copy, bool):
        raise TypeError(f"copy must be a bool, got {type(copy).__name__}")

    requested, collection_audit = collect_marker_genes(
        general_marker_genes,
        specific_marker_genes,
        include_general=include_general,
        include_specific=include_specific,
    )
    active_general: MarkerDictionary = (
        general_marker_genes if include_general else {}
    )
    active_specific: MarkerDictionary = (
        specific_marker_genes if include_specific else {}
    )
    match_audit = match_marker_genes(
        requested,
        adata.var_names,
        general_marker_genes=active_general,
        specific_marker_genes=active_specific,
    )
    if match_audit.matched_marker_count == 0:
        raise ValueError(
            "No requested marker genes matched adata.var_names using exact, "
            "case-sensitive matching"
        )

    position_by_name = {
        str(variable_name): position
        for position, variable_name in enumerate(adata.var_names)
    }
    variable_positions = [
        position_by_name[marker] for marker in match_audit.matched_markers
    ]
    marker_view = adata[:, variable_positions]
    marker_adata = marker_view.copy() if copy else marker_view
    _validate_alignment(marker_adata)

    input_was_sparse = sparse.issparse(adata.X)
    output_is_sparse = sparse.issparse(marker_adata.X)
    audit = MarkerExtractionAudit(
        requested_marker_count_before_deduplication=(
            collection_audit.requested_marker_count_before_deduplication
        ),
        unique_requested_marker_count=(
            collection_audit.unique_requested_marker_count
        ),
        matched_marker_count=match_audit.matched_marker_count,
        unmatched_marker_count=match_audit.unmatched_marker_count,
        match_percentage=match_audit.match_percentage,
        matched_markers=match_audit.matched_markers,
        unmatched_markers=match_audit.unmatched_markers,
        duplicate_marker_occurrence_count=(
            collection_audit.duplicate_marker_occurrence_count
        ),
        duplicate_marker_occurrences_removed=(
            collection_audit.duplicate_marker_occurrences_removed
        ),
        markers_in_multiple_groups=match_audit.markers_in_multiple_groups,
        included_general_marker_groups=(
            collection_audit.included_general_marker_groups
        ),
        included_specific_marker_groups=(
            collection_audit.included_specific_marker_groups
        ),
        input_spot_count=adata.n_obs,
        input_variable_count=adata.n_vars,
        output_spot_count=marker_adata.n_obs,
        output_variable_count=marker_adata.n_vars,
        input_matrix_was_sparse=input_was_sparse,
        output_matrix_is_sparse=output_is_sparse,
        operated_on_copy=copy,
    )
    return marker_adata, audit


def _validate_panel_flags(
    *,
    include_general: bool,
    include_specific: bool,
) -> None:
    for argument_name, value in (
        ("include_general", include_general),
        ("include_specific", include_specific),
    ):
        if not isinstance(value, bool):
            raise TypeError(
                f"{argument_name} must be a bool, got {type(value).__name__}"
            )
    if not include_general and not include_specific:
        raise ValueError(
            "At least one marker panel must be enabled: include_general or "
            "include_specific"
        )


def _materialize_marker_groups(
    marker_groups: MarkerDictionary,
    *,
    panel_name: str,
) -> MaterializedMarkerGroups:
    if not isinstance(marker_groups, Mapping):
        raise TypeError(
            f"{panel_name} must be a mapping of group names to marker collections"
        )

    materialized: list[tuple[str, tuple[str, ...]]] = []
    for group_name, marker_collection in marker_groups.items():
        if not isinstance(group_name, str) or not group_name.strip():
            raise ValueError(
                f"{panel_name} group names must be non-empty strings; "
                f"got {group_name!r}"
            )
        markers = _materialize_symbol_sequence(
            marker_collection,
            argument_name=f"{panel_name}[{group_name!r}]",
            allow_empty=False,
        )
        materialized.append((group_name, markers))
    return tuple(materialized)


def _materialize_symbol_sequence(
    symbols: Iterable[str],
    *,
    argument_name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(symbols, str) or not isinstance(symbols, Iterable):
        raise TypeError(f"{argument_name} must be an iterable of marker strings")

    materialized = tuple(symbols)
    if not allow_empty and not materialized:
        raise ValueError(f"{argument_name} must contain at least one marker")
    for position, symbol in enumerate(materialized):
        if not isinstance(symbol, str):
            raise TypeError(
                f"{argument_name} must contain only strings; value at position "
                f"{position} has type {type(symbol).__name__}"
            )
        if not symbol.strip():
            raise ValueError(
                f"{argument_name} contains an empty marker at position {position}"
            )
    return materialized


def _find_markers_in_multiple_groups(
    requested_markers: tuple[str, ...],
    *,
    general_groups: MaterializedMarkerGroups,
    specific_groups: MaterializedMarkerGroups,
) -> tuple[str, ...]:
    memberships: dict[str, set[tuple[str, str]]] = {}
    for panel_name, groups in (
        ("general", general_groups),
        ("specific", specific_groups),
    ):
        for group_name, markers in groups:
            group_reference = (panel_name, group_name)
            for marker in markers:
                memberships.setdefault(marker, set()).add(group_reference)

    multiple_group_markers: list[str] = []
    seen: set[str] = set()
    for marker in requested_markers:
        if (
            marker not in seen
            and len(memberships.get(marker, set())) > 1
        ):
            multiple_group_markers.append(marker)
            seen.add(marker)
    return tuple(multiple_group_markers)


def _validate_annotated_adata(adata: AnnData) -> None:
    if not isinstance(adata, AnnData):
        raise TypeError(
            f"Expected an AnnData object, got {type(adata).__name__}"
        )
    _validate_alignment(adata)
    if adata.X is None:
        raise ValueError("adata.X must contain an expression matrix")
    if not adata.var_names.is_unique:
        raise ValueError(
            "adata.var_names must be unique before marker extraction; run the "
            "gene-annotation stage first"
        )
    _materialize_symbol_sequence(
        adata.var_names,
        argument_name="adata.var_names",
        allow_empty=True,
    )


def _validate_alignment(adata: AnnData) -> None:
    if not (
        adata.n_vars == len(adata.var)
        and adata.n_vars == len(adata.var_names)
    ):
        raise ValueError(
            "AnnData feature alignment is invalid: n_vars, var rows, and "
            "var_names length must agree"
        )
    if adata.n_obs != len(adata.obs):
        raise ValueError(
            "AnnData observation alignment is invalid: n_obs and obs rows "
            "must agree"
        )
    if adata.X is not None and adata.X.shape != adata.shape:
        raise ValueError(
            "AnnData expression alignment is invalid: X shape must match "
            "(n_obs, n_vars)"
        )
