"""Data models for stereotype dimensions, groups, and prompt metadata.

Separates:
- Stereotype dimensions/pairs (WGI, O*NET, QS) - defined in Python
- Demographic groups - loaded from CSV files
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from .config import DEMOGRAPHY_TO_DIMENSION


class Demography(str, Enum):
    """Demographic categories for stereotype testing."""
    
    NATIONALITY = "nationality"
    RACE = "race"
    RELIGION = "religion"
    GENDER = "gender"
    ETHNICITY = "ethnicity"
    LEMBAGA = "lembaga"
    MEDIA = "media"
    NAME = "name"
    ORGANISASI_BISNIS = "organisasi_bisnis"
    POLITICAL_PARTY = "political_party"
    UNIVERSITY = "university"


class StereotypeDimension(str, Enum):
    """Dimensions for categorizing stereotype pairs.
    
    - WGI: World Governance Indicators - institutional/governance stereotypes
    - ONET: O*NET Occupational - job/occupation-related stereotypes
    - QS_RANKINGS: QS University Rankings - academic/university stereotypes
    - SP: Stereotype Prevalence - common societal stereotypes
    """
    
    WGI = "wgi"
    ONET = "onet"
    QS_RANKINGS = "qs_rankings"
    SP = "sp"
# =============================================================================
# STEREOTYPE PAIRS
# =============================================================================
class StereotypePair(BaseModel):
    """A pair of positive and negative stereotype labels within a dimension."""
    
    pair_id: str = Field(..., description="Unique identifier for this stereotype pair")
    parent_category: str | None = Field(
        None,
        description="Higher-level category for this pair within the dimension hierarchy",
    )
    positive_label: str = Field(..., description="The positive stereotype phrase")
    negative_label: str = Field(..., description="The negative stereotype phrase")
    description: str | None = Field(None, description="Optional description of the pair")
    
    class Config:
        frozen = True


class DimensionDefinition(BaseModel):
    """Definition of a stereotype dimension with its pairs."""
    
    dimension: StereotypeDimension = Field(..., description="Dimension identifier")
    name: str = Field(..., description="Human-readable name")
    description: str | None = Field(None, description="Description of this dimension")
    pairs: list[StereotypePair] = Field(
        default_factory=list,
        description="Stereotype pairs in this dimension",
    )
    
    def get_pair(self, pair_id: str) -> StereotypePair | None:
        """Get a specific pair by ID."""
        for pair in self.pairs:
            if pair.pair_id == pair_id:
                return pair
        return None

class DimensionCatalog(BaseModel):
    """Catalog of all stereotype dimensions with their pairs.
    
    This is loaded from Python data and contains the reusable stereotype pairs
    organized by dimension (WGI, O*NET, QS).
    """
    
    name: str = Field(default="Stereotype Dimension Catalog", description="Catalog name")
    version: str = Field(default="1.0.0", description="Catalog version")
    description: str | None = Field(None, description="Catalog description")
    dimensions: list[DimensionDefinition] = Field(
        default_factory=list,
        description="List of dimension definitions",
    )
    
    def get_dimension(self, dimension: StereotypeDimension) -> DimensionDefinition | None:
        """Get a dimension by type."""
        for dim in self.dimensions:
            if dim.dimension == dimension:
                return dim
        return None
    
    def get_all_pairs(self) -> list[tuple[StereotypeDimension, StereotypePair]]:
        """Get all pairs with their dimensions."""
        result = []
        for dim in self.dimensions:
            for pair in dim.pairs:
                result.append((dim.dimension, pair))
        return result
    
    def get_pairs_for_dimension(self, dimension: StereotypeDimension) -> list[StereotypePair]:
        """Get all pairs for a specific dimension."""
        dim_def = self.get_dimension(dimension)
        return dim_def.pairs if dim_def else []

# =============================================================================
# GROUPS (from CSV)
# =============================================================================

class Group(BaseModel):
    """A demographic group loaded from CSV.
    
    CSV columns: group_id, group_name, demography, country (optional)
    """
    
    group_id: str = Field(..., description="Unique identifier for this group")
    group_name: str = Field(..., description="Display name (e.g., 'Javanese', 'Muslim')")
    demography: Demography = Field(..., description="Demographic category")
    country: str | None = Field(None, description="Associated country (optional)")
    
    class Config:
        frozen = True
    
    def get_dimension(self) -> StereotypeDimension:
        """Get the stereotype dimension associated with this group's demography."""
        dimension_str = DEMOGRAPHY_TO_DIMENSION.get(self.demography.value)
        if dimension_str is None:
            raise ValueError(f"No dimension mapping found for demography: {self.demography.value}")
        return StereotypeDimension(dimension_str)

class GroupCollection(BaseModel):
    """Collection of groups loaded from CSV files."""
    
    name: str = Field(default="Group Collection", description="Collection name")
    source_file: str | None = Field(None, description="Source CSV file path")
    groups: list[Group] = Field(default_factory=list, description="List of groups")
    
    def get_groups_by_demography(self, demography: Demography) -> list[Group]:
        """Filter groups by demographic category."""
        return [g for g in self.groups if g.demography == demography]
    
    def get_group(self, group_id: str) -> Group | None:
        """Get a specific group by ID."""
        for group in self.groups:
            if group.group_id == group_id:
                return group
        return None

# =============================================================================
# PROMPT METADATA & OUTPUT
# =============================================================================

class PromptMetadata(BaseModel):
    """Metadata for a generated prompt."""
    
    prompt_id: str = Field(..., description="Unique identifier for this prompt")
    demography: Demography = Field(..., description="Demographic category")
    group: str = Field(..., description="The target group name")
    group_id: str = Field(..., description="Group identifier")
    dimension: StereotypeDimension = Field(..., description="Stereotype dimension")
    stereotype_pair: str = Field(..., description="Stereotype pair identifier")
    parent_category: str | None = Field(
        None,
        description="Higher-level category for the selected stereotype pair",
    )
    positive_label: str = Field(..., description="The positive stereotype phrase")
    negative_label: str = Field(..., description="The negative stereotype phrase")
    task_id: int = Field(..., ge=1, le=7, description="Task ID (1-7)")
    task_complexity_level: int = Field(..., ge=1, le=7, description="Complexity level (1-7)")
    task_name: str = Field(..., description="Human-readable task name")
    
    def to_filename(self) -> str:
        """Generate a standardized filename for this prompt."""
        group_safe = self.group.lower().replace(" ", "_")
        pair_safe = self.stereotype_pair.lower().replace(" ", "_")
        return f"{self.demography.value}_{group_safe}_{self.dimension.value}_{pair_safe}_task{self.task_id}.txt"


class GeneratedPrompt(BaseModel):
    """A fully generated prompt with its metadata."""
    
    prompt_text: str = Field(..., description="The full prompt text")
    metadata: PromptMetadata = Field(..., description="Associated metadata")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt": self.prompt_text,
            "metadata": self.metadata.model_dump(),
        }
