"""
archives — archival image sourcing for script beats that reference
specific landmarks, years, historical events, people, or statistics.

Public surface:
  - AnchorSpec, ArchivalAsset, PoolEntry, PoolManifest (models)
  - harvest(anchors) -> PoolManifest (harvester)

The harvester fans each AnchorSpec out to the right adapters, normalizes
hits into ArchivalAsset, dedupes, and assigns an evidence_status so the
script writer knows how specific its claims may be.
"""
from .harvester import harvest
from .models import (
    AnchorSpec,
    ArchivalAsset,
    EvidenceStatus,
    NegativeFinding,
    PoolEntry,
    PoolManifest,
)

__all__ = [
    "AnchorSpec",
    "ArchivalAsset",
    "EvidenceStatus",
    "NegativeFinding",
    "PoolEntry",
    "PoolManifest",
    "harvest",
]
