"""flybrain.connectome: data model, sparse helpers and functional groups (SPEC sections c.2-c.4).

Only the numpy-only foundation modules are imported here (schema, csr, groups). The remaining
E1 modules (patches, synthetic, loaders, cache) import from these and are imported explicitly by
their users.
"""

from .csr import build_csr, in_degree, remap_ids, sum_duplicates, transpose_csr
from .groups import (
    GROUP_REGEX,
    READOUTS,
    SIDED,
    STAR_TYPES,
    fold_sided_counts,
    readout_partition,
    readout_sizes,
    region_of,
    resolve_groups,
    side_of,
    validate_groups,
)
from .schema import (
    CSR,
    NT_VALUES,
    REGION_ID,
    REGIONS,
    SYNTHETIC_BODY_BASE,
    Connectome,
    count_groups,
    count_regions,
    utc_now_iso,
)

__all__ = [
    "CSR",
    "Connectome",
    "GROUP_REGEX",
    "NT_VALUES",
    "READOUTS",
    "REGIONS",
    "REGION_ID",
    "SIDED",
    "STAR_TYPES",
    "SYNTHETIC_BODY_BASE",
    "build_csr",
    "count_groups",
    "count_regions",
    "fold_sided_counts",
    "in_degree",
    "readout_partition",
    "readout_sizes",
    "region_of",
    "remap_ids",
    "resolve_groups",
    "side_of",
    "sum_duplicates",
    "transpose_csr",
    "utc_now_iso",
    "validate_groups",
]
