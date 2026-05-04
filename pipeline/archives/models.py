from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class AnchorSpec:
    beat_id: int
    anchor_type: str  # landmark|event|person|stat|vanished_place|map
    subject: str
    geo: str
    era: Optional[List[int]] = None # [start, end]
    queries: List[str] = field(default_factory=list)
    allow_substitute: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnchorSpec":
        return cls(
            beat_id=data.get("beat_id", 0),
            anchor_type=data.get("anchor_type", "landmark"),
            subject=data.get("subject", ""),
            geo=data.get("geo", ""),
            era=data.get("era"),
            queries=data.get("queries") or [],
            allow_substitute=data.get("allow_substitute", True)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beat_id": self.beat_id,
            "anchor_type": self.anchor_type,
            "subject": self.subject,
            "geo": self.geo,
            "era": self.era,
            "queries": self.queries,
            "allow_substitute": self.allow_substitute
        }


class EvidenceStatus:
    DIRECT = "direct"
    CONTEXTUAL = "contextual"
    SUBSTITUTE = "substitute"
    UNSUPPORTED = "unsupported"


@dataclass
class ArchivalAsset:
    source: str
    source_record_id: str
    record_url: str
    image_url: str
    thumbnail_url: str = ""
    title: str = ""
    description: str = ""
    creator: str = ""
    date: str = ""
    year: Optional[int] = None
    place: str = ""
    rights: str = ""
    media_hint: str = "photo"   # photo|map|postcard|aerial|document|unknown
    width: Optional[int] = None
    height: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PoolEntry:
    beat_id: int
    anchor_type: str
    asset: ArchivalAsset
    relevance_score: float
    evidence_status: str
    matched_on: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NegativeFinding:
    beat_id: int
    anchor_type: str
    subject: str
    queried_sources: List[str]
    reason: str  # no_queries|no_hits|all_below_threshold|wrong_geo|wrong_era

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PoolManifest:
    entries: List[PoolEntry] = field(default_factory=list)
    negative: List[NegativeFinding] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def entries_for_beat(self, beat_id: int) -> List[PoolEntry]:
        return [e for e in self.entries if e.beat_id == beat_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "negative": [n.to_dict() for n in self.negative],
            "stats": self.stats,
        }
