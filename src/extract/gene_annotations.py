"""Gene-symbol annotation utilities for AnnData feature metadata."""

from dataclasses import dataclass

import pandas as pd
from anndata import AnnData


@dataclass(frozen=True)
class GeneAnnotationAudit:
    """Summary metrics from dataset-provided gene-symbol annotation."""

    annotation_source_column: str
    total_features: int
    symbols_present: int
    symbols_missing: int
    ensembl_fallbacks_used: int
    unique_canonical_symbols: int
    canonical_symbols_with_collisions: tuple[str, ...]
    duplicated_canonical_symbol_names: int
    features_in_symbol_collisions: int
    duplicated_ensembl_identifiers: int
    final_var_names_unique: bool
    operated_on_copy: bool


def set_gene_symbols_from_feature_name(
    adata: AnnData,
    symbol_column: str = "feature_name",
    ensembl_id_column: str = "ensembl_id",
    canonical_symbol_column: str = "gene_symbol",
    final_label_column: str = "var_name_unique",
    copy: bool = True,
    uniqueness_separator: str = "-",
) -> tuple[AnnData, GeneAnnotationAudit]:
    """Annotate AnnData variables with cleaned gene symbols and stable IDs.

    Parameters
    ----------
    adata
        AnnData object whose current variable names are original feature IDs on
        the first run.
    symbol_column
        Column in ``adata.var`` containing dataset-provided gene symbols.
    ensembl_id_column
        Column used to preserve stable original feature identifiers.
    canonical_symbol_column
        Column used for cleaned, unsuffixed canonical symbols.
    final_label_column
        Column used for the exact final unique variable names.
    copy
        If ``True``, annotate a copy. If ``False``, modify ``adata`` in place.
    uniqueness_separator
        Separator passed to ``AnnData.var_names_make_unique``.

    Returns
    -------
    annotated, audit
        The annotated AnnData object and its immutable annotation audit.
    """
    _validate_metadata_column_names(
        symbol_column=symbol_column,
        ensembl_id_column=ensembl_id_column,
        canonical_symbol_column=canonical_symbol_column,
        final_label_column=final_label_column,
    )
    if not isinstance(adata, AnnData):
        raise TypeError(
            f"Expected an AnnData object, got {type(adata).__name__}"
        )
    if symbol_column not in adata.var.columns:
        raise KeyError(
            f"Gene-symbol column '{symbol_column}' is missing from adata.var"
        )
    _validate_alignment(adata)
    if not isinstance(uniqueness_separator, str) or not uniqueness_separator:
        raise ValueError("uniqueness_separator must be a non-empty string")

    annotated = adata.copy() if copy else adata
    ensembl_ids = _get_or_preserve_ensembl_ids(
        annotated,
        ensembl_id_column=ensembl_id_column,
    )

    canonical_symbols = (
        annotated.var[symbol_column]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )
    annotated.var[canonical_symbol_column] = canonical_symbols

    symbol_counts = canonical_symbols.dropna().value_counts()
    collision_counts = symbol_counts[symbol_counts > 1]
    collision_symbols = tuple(sorted(str(value) for value in collision_counts.index))

    preliminary_labels = canonical_symbols.fillna(ensembl_ids)
    annotated.var_names = pd.Index(preliminary_labels.astype(str))
    annotated.var_names_make_unique(join=uniqueness_separator)
    annotated.var[final_label_column] = pd.Series(
        annotated.var_names.astype(str),
        index=annotated.var.index,
        dtype="string",
    )

    _validate_alignment(annotated)
    if not annotated.var_names.is_unique:
        raise ValueError("Unable to construct unique AnnData variable names")

    symbols_present = int(canonical_symbols.notna().sum())
    symbols_missing = int(canonical_symbols.isna().sum())
    audit = GeneAnnotationAudit(
        annotation_source_column=symbol_column,
        total_features=annotated.n_vars,
        symbols_present=symbols_present,
        symbols_missing=symbols_missing,
        ensembl_fallbacks_used=symbols_missing,
        unique_canonical_symbols=int(canonical_symbols.nunique(dropna=True)),
        canonical_symbols_with_collisions=collision_symbols,
        duplicated_canonical_symbol_names=len(collision_symbols),
        features_in_symbol_collisions=int(collision_counts.sum()),
        duplicated_ensembl_identifiers=0,
        final_var_names_unique=annotated.var_names.is_unique,
        operated_on_copy=copy,
    )
    return annotated, audit


def _validate_metadata_column_names(
    *,
    symbol_column: str,
    ensembl_id_column: str,
    canonical_symbol_column: str,
    final_label_column: str,
) -> None:
    column_names = {
        "symbol_column": symbol_column,
        "ensembl_id_column": ensembl_id_column,
        "canonical_symbol_column": canonical_symbol_column,
        "final_label_column": final_label_column,
    }
    invalid_arguments = [
        f"{argument}={value!r}"
        for argument, value in column_names.items()
        if not isinstance(value, str) or not value.strip()
    ]
    if invalid_arguments:
        raise ValueError(
            "Metadata column names must be non-empty strings; invalid "
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
            "Metadata column names must be pairwise distinct; conflicting "
            f"name(s): {'; '.join(conflicts)}"
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


def _get_or_preserve_ensembl_ids(
    adata: AnnData,
    *,
    ensembl_id_column: str,
) -> pd.Series:
    if ensembl_id_column in adata.var.columns:
        ensembl_ids = adata.var[ensembl_id_column].astype("string")
    else:
        ensembl_ids = pd.Series(
            adata.var_names,
            index=adata.var.index,
            dtype="string",
        )

    missing_mask = ensembl_ids.isna() | ensembl_ids.str.strip().eq("")
    if missing_mask.any():
        positions = [
            str(position) for position in range(len(ensembl_ids)) if missing_mask.iloc[position]
        ]
        raise ValueError(
            "Original feature identifiers must be non-null and non-empty; "
            f"invalid value(s) at feature position(s): {', '.join(positions)}"
        )

    duplicated_mask = ensembl_ids.duplicated(keep=False)
    if duplicated_mask.any():
        duplicated_ids = tuple(
            sorted(str(value) for value in ensembl_ids[duplicated_mask].unique())
        )
        raise ValueError(
            "Original Ensembl identifiers must be unique; duplicated "
            f"identifier(s): {', '.join(duplicated_ids)}"
        )

    preserved = pd.Series(
        ensembl_ids.array.copy(),
        index=adata.var.index,
        dtype="string",
    )
    adata.var[ensembl_id_column] = preserved
    return preserved
