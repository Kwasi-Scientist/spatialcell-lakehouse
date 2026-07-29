"""In-memory gene-symbol annotation for spatial AnnData feature metadata.

The module preserves source feature identifiers and expression values. It does
not infer cell identity, interpret the expression source, or query external
annotation services.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
from anndata import AnnData
from scipy import sparse


@dataclass(frozen=True)
class GeneAnnotationAudit:
    """Immutable summary of gene annotation and preservation checks."""

    input_spot_count: int
    input_variable_count: int
    output_spot_count: int
    output_variable_count: int
    source_gene_symbol_column: str
    preserved_identifier_column: str
    cleaned_symbol_column: str
    final_label_provenance_column: str
    valid_cleaned_symbol_count: int
    missing_symbol_count: int
    identifier_fallback_count: int
    unique_cleaned_symbol_count: int
    duplicated_symbols: tuple[str, ...]
    distinct_duplicated_symbol_count: int
    duplicated_symbol_feature_count: int
    unique_final_var_name_count: int
    input_var_names_were_unique: bool
    final_var_names_are_unique: bool
    input_matrix_was_sparse: bool
    output_matrix_is_sparse: bool
    operated_on_copy: bool
    observation_order_preserved: bool
    variable_order_preserved: bool

    @property
    def annotation_source_column(self) -> str:
        """Return the legacy name for the source symbol column."""
        return self.source_gene_symbol_column

    @property
    def total_features(self) -> int:
        """Return the legacy total-feature metric."""
        return self.output_variable_count

    @property
    def symbols_present(self) -> int:
        """Return the legacy valid-symbol metric."""
        return self.valid_cleaned_symbol_count

    @property
    def symbols_missing(self) -> int:
        """Return the legacy missing-symbol metric."""
        return self.missing_symbol_count

    @property
    def ensembl_fallbacks_used(self) -> int:
        """Return the legacy fallback metric."""
        return self.identifier_fallback_count

    @property
    def unique_canonical_symbols(self) -> int:
        """Return the legacy unique cleaned-symbol metric."""
        return self.unique_cleaned_symbol_count

    @property
    def canonical_symbols_with_collisions(self) -> tuple[str, ...]:
        """Return the legacy duplicated-symbol sequence."""
        return self.duplicated_symbols

    @property
    def duplicated_canonical_symbol_names(self) -> int:
        """Return the legacy distinct duplicated-symbol count."""
        return self.distinct_duplicated_symbol_count

    @property
    def features_in_symbol_collisions(self) -> int:
        """Return the legacy duplicated-symbol affected-row count."""
        return self.duplicated_symbol_feature_count

    @property
    def duplicated_ensembl_identifiers(self) -> int:
        """Return zero because duplicate source identifiers are rejected."""
        return 0

    @property
    def final_var_names_unique(self) -> bool:
        """Return the legacy final uniqueness flag."""
        return self.final_var_names_are_unique


def clean_gene_symbols(
    raw_symbols: Iterable[object] | pd.Series,
) -> pd.Series:
    """Return cleaned gene symbols while retaining true missing values.

    Valid strings keep their capitalization and have surrounding whitespace
    removed. Null, empty, and whitespace-only values become ``pd.NA``.
    Non-null, non-string values are rejected rather than silently converted.
    A Series input retains its index and name.
    """
    if isinstance(raw_symbols, (str, bytes)) or not isinstance(
        raw_symbols,
        Iterable,
    ):
        raise TypeError("raw_symbols must be an iterable or pandas Series")

    if isinstance(raw_symbols, pd.Series):
        raw = raw_symbols.copy(deep=True)
    else:
        raw = pd.Series(tuple(raw_symbols), dtype="object")

    invalid_positions = [
        position
        for position, value in enumerate(raw)
        if not _is_missing(value) and not isinstance(value, str)
    ]
    if invalid_positions:
        formatted_positions = ", ".join(str(value) for value in invalid_positions)
        raise TypeError(
            "Gene symbols must be strings or null; non-string value(s) at "
            f"position(s): {formatted_positions}"
        )

    cleaned = raw.astype("string").str.strip()
    cleaned = cleaned.mask(cleaned.eq(""), pd.NA)
    cleaned.name = raw.name
    return cleaned


def annotate_gene_names(
    adata: AnnData,
    gene_symbol_column: str = "feature_name",
    ensembl_id_column: str = "ensembl_id",
    cleaned_symbol_column: str = "gene_symbol",
    final_label_column: str = "var_name_base",
    *,
    copy: bool = True,
    uniqueness_separator: str = "-",
) -> tuple[AnnData, GeneAnnotationAudit]:
    """Annotate feature names while preserving source IDs and AnnData alignment.

    ``final_label_column`` stores symbol-or-identifier labels before uniqueness
    suffixes are added. With ``copy=True`` the source object is unchanged; with
    ``copy=False`` the supplied AnnData object is modified and returned.

    An existing identifier column is accepted only when its values exactly match
    the current source ``adata.var_names`` in row order.
    """
    _validate_column_names(
        gene_symbol_column=gene_symbol_column,
        ensembl_id_column=ensembl_id_column,
        cleaned_symbol_column=cleaned_symbol_column,
        final_label_column=final_label_column,
    )
    if not isinstance(copy, bool):
        raise TypeError(f"copy must be a bool, got {type(copy).__name__}")
    if (
        not isinstance(uniqueness_separator, str)
        or not uniqueness_separator
    ):
        raise ValueError("uniqueness_separator must be a non-empty string")
    if not isinstance(adata, AnnData):
        raise TypeError(
            f"Expected an AnnData object, got {type(adata).__name__}"
        )

    _validate_alignment(adata)
    if gene_symbol_column not in adata.var.columns:
        raise KeyError(
            f"Gene-symbol column '{gene_symbol_column}' is missing from adata.var"
        )

    original_identifiers = _validate_source_identifiers(adata)
    _validate_existing_identifier_column(
        adata,
        ensembl_id_column=ensembl_id_column,
        original_identifiers=original_identifiers,
    )
    cleaned_symbols = clean_gene_symbols(adata.var[gene_symbol_column])

    input_obs_names = tuple(str(value) for value in adata.obs_names)
    input_variable_identifiers = tuple(original_identifiers)
    input_was_sparse = sparse.issparse(adata.X)
    input_spot_count = adata.n_obs
    input_variable_count = adata.n_vars

    annotated = adata.copy() if copy else adata
    preserved_identifiers = pd.Series(
        original_identifiers,
        index=annotated.var.index,
        dtype="string",
    )
    annotated.var[ensembl_id_column] = preserved_identifiers
    annotated.var[cleaned_symbol_column] = pd.Series(
        cleaned_symbols.array.copy(),
        index=annotated.var.index,
        dtype="string",
    )

    base_labels = annotated.var[cleaned_symbol_column].fillna(
        annotated.var[ensembl_id_column]
    )
    annotated.var[final_label_column] = pd.Series(
        base_labels.array.copy(),
        index=annotated.var.index,
        dtype="string",
    )

    symbol_counts = annotated.var[cleaned_symbol_column].dropna().value_counts()
    duplicated_symbol_counts = symbol_counts[symbol_counts > 1]
    duplicated_symbols = tuple(
        sorted(str(value) for value in duplicated_symbol_counts.index)
    )

    annotated.var_names = pd.Index(base_labels.astype(str))
    annotated.var_names_make_unique(join=uniqueness_separator)
    _validate_alignment(annotated)
    if not annotated.var_names.is_unique:
        raise ValueError("Unable to construct unique AnnData variable names")

    valid_symbol_count = int(
        annotated.var[cleaned_symbol_column].notna().sum()
    )
    missing_symbol_count = annotated.n_vars - valid_symbol_count
    output_is_sparse = sparse.issparse(annotated.X)
    observation_order_preserved = (
        tuple(str(value) for value in annotated.obs_names) == input_obs_names
    )
    variable_order_preserved = (
        tuple(str(value) for value in annotated.var[ensembl_id_column])
        == input_variable_identifiers
    )

    audit = GeneAnnotationAudit(
        input_spot_count=input_spot_count,
        input_variable_count=input_variable_count,
        output_spot_count=annotated.n_obs,
        output_variable_count=annotated.n_vars,
        source_gene_symbol_column=gene_symbol_column,
        preserved_identifier_column=ensembl_id_column,
        cleaned_symbol_column=cleaned_symbol_column,
        final_label_provenance_column=final_label_column,
        valid_cleaned_symbol_count=valid_symbol_count,
        missing_symbol_count=missing_symbol_count,
        identifier_fallback_count=missing_symbol_count,
        unique_cleaned_symbol_count=int(
            annotated.var[cleaned_symbol_column].nunique(dropna=True)
        ),
        duplicated_symbols=duplicated_symbols,
        distinct_duplicated_symbol_count=len(duplicated_symbols),
        duplicated_symbol_feature_count=int(duplicated_symbol_counts.sum()),
        unique_final_var_name_count=int(annotated.var_names.nunique()),
        input_var_names_were_unique=True,
        final_var_names_are_unique=annotated.var_names.is_unique,
        input_matrix_was_sparse=input_was_sparse,
        output_matrix_is_sparse=output_is_sparse,
        operated_on_copy=copy,
        observation_order_preserved=observation_order_preserved,
        variable_order_preserved=variable_order_preserved,
    )
    return annotated, audit


def set_gene_symbols_from_feature_name(
    adata: AnnData,
    symbol_column: str = "feature_name",
    ensembl_id_column: str = "ensembl_id",
    canonical_symbol_column: str = "gene_symbol",
    final_label_column: str = "var_name_base",
    copy: bool = True,
    uniqueness_separator: str = "-",
) -> tuple[AnnData, GeneAnnotationAudit]:
    """Compatibility wrapper for :func:`annotate_gene_names`."""
    return annotate_gene_names(
        adata,
        gene_symbol_column=symbol_column,
        ensembl_id_column=ensembl_id_column,
        cleaned_symbol_column=canonical_symbol_column,
        final_label_column=final_label_column,
        copy=copy,
        uniqueness_separator=uniqueness_separator,
    )


def _validate_column_names(
    *,
    gene_symbol_column: str,
    ensembl_id_column: str,
    cleaned_symbol_column: str,
    final_label_column: str,
) -> None:
    column_names = {
        "gene_symbol_column": gene_symbol_column,
        "ensembl_id_column": ensembl_id_column,
        "cleaned_symbol_column": cleaned_symbol_column,
        "final_label_column": final_label_column,
    }
    invalid_arguments = [
        f"{argument}={value!r}"
        for argument, value in column_names.items()
        if not isinstance(value, str) or not value.strip()
    ]
    if invalid_arguments:
        raise ValueError(
            "Annotation column names must be non-empty strings; invalid "
            f"argument(s): {', '.join(invalid_arguments)}"
        )

    arguments_by_name: dict[str, list[str]] = {}
    for argument, name in column_names.items():
        arguments_by_name.setdefault(name, []).append(argument)
    conflicts = [
        f"{name!r} is used by {', '.join(arguments)}"
        for name, arguments in arguments_by_name.items()
        if len(arguments) > 1
    ]
    if conflicts:
        raise ValueError(
            "Annotation column names must be pairwise distinct; conflicting "
            f"name(s): {'; '.join(conflicts)}"
        )


def _validate_source_identifiers(adata: AnnData) -> tuple[str, ...]:
    input_names_were_unique = adata.var_names.is_unique
    if not input_names_were_unique:
        duplicated = tuple(
            sorted(
                str(value)
                for value in adata.var_names[
                    adata.var_names.duplicated(keep=False)
                ].unique()
            )
        )
        raise ValueError(
            "Source adata.var_names must be unique before annotation; duplicated "
            f"identifier(s): {', '.join(duplicated)}"
        )

    raw_identifiers = tuple(adata.var_names)
    invalid_type_positions = [
        position
        for position, value in enumerate(raw_identifiers)
        if not isinstance(value, str)
    ]
    if invalid_type_positions:
        formatted_positions = ", ".join(
            str(value) for value in invalid_type_positions
        )
        raise TypeError(
            "Source feature identifiers in adata.var_names must be strings; "
            f"invalid value(s) at position(s): {formatted_positions}"
        )

    identifiers = tuple(raw_identifiers)
    invalid_positions = [
        position
        for position, value in enumerate(identifiers)
        if not value.strip()
    ]
    if invalid_positions:
        formatted_positions = ", ".join(str(value) for value in invalid_positions)
        raise ValueError(
            "Source feature identifiers in adata.var_names must be non-empty; "
            f"invalid value(s) at position(s): {formatted_positions}"
        )
    return identifiers


def _validate_existing_identifier_column(
    adata: AnnData,
    *,
    ensembl_id_column: str,
    original_identifiers: tuple[str, ...],
) -> None:
    if ensembl_id_column not in adata.var.columns:
        return

    existing = adata.var[ensembl_id_column]
    if existing.isna().any():
        raise ValueError(
            f"Existing identifier column '{ensembl_id_column}' contains null "
            "values and cannot be overwritten"
        )
    invalid_type_positions = [
        position
        for position, value in enumerate(existing)
        if not isinstance(value, str)
    ]
    if invalid_type_positions:
        formatted_positions = ", ".join(
            str(value) for value in invalid_type_positions
        )
        raise TypeError(
            f"Existing identifier column '{ensembl_id_column}' must contain "
            "only strings; invalid value(s) at position(s): "
            f"{formatted_positions}"
        )
    existing_identifiers = tuple(existing)
    if any(not value.strip() for value in existing_identifiers):
        raise ValueError(
            f"Existing identifier column '{ensembl_id_column}' contains empty "
            "values and cannot be overwritten"
        )
    if existing_identifiers != original_identifiers:
        raise ValueError(
            f"Existing identifier column '{ensembl_id_column}' does not exactly "
            "match current adata.var_names; refusing to overwrite preserved "
            "source identifiers"
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


def _is_missing(value: object) -> bool:
    missing = pd.isna(value)
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False
