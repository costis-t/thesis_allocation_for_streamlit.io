"""
Configuration management for thesis allocation.
Supports both programmatic configuration and file-based (YAML/JSON) configs.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class PreferenceConfig:
    """Preference calculation parameters."""
    allow_unranked: bool = True
    tier2_cost: int = 1
    tier3_cost: int = 5
    unranked_cost: int = 200
    top2_bias: bool = True
    # Ranked choice costs (configurable)
    rank1_cost: int = 0
    rank2_cost: int = 1
    rank3_cost: int = 100
    rank4_cost: int = 101
    rank5_cost: int = 102
    min_acceptable_preference_rank: int | None = None
    max_acceptable_preference_rank: int | None = None
    excluded_preference_ranks: list | None = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> PreferenceConfig:
        """Create from dictionary with defaults."""
        return PreferenceConfig(
            allow_unranked=d.get("allow_unranked", True),
            tier2_cost=d.get("tier2_cost", 1),
            tier3_cost=d.get("tier3_cost", 5),
            unranked_cost=d.get("unranked_cost", 200),
            top2_bias=d.get("top2_bias", True),
            rank1_cost=d.get("rank1_cost", 0),
            rank2_cost=d.get("rank2_cost", 1),
            rank3_cost=d.get("rank3_cost", 100),
            rank4_cost=d.get("rank4_cost", 101),
            rank5_cost=d.get("rank5_cost", 102),
            min_acceptable_preference_rank=d.get("min_acceptable_preference_rank", None),
            max_acceptable_preference_rank=d.get("max_acceptable_preference_rank", None),
            excluded_preference_ranks=d.get("excluded_preference_ranks", None),
        )


@dataclass
class CapacityConfig:
    """Capacity constraint parameters."""
    enable_topic_overflow: bool = False
    enable_coach_overflow: bool = False
    dept_min_mode: str = "soft"  # "soft" or "hard"
    dept_max_mode: str = "soft"  # "soft" or "hard"
    P_dept_shortfall: int = 1000  # penalty for department minimum shortfall
    P_dept_overflow: int = 1200  # penalty for department maximum overflow
    P_topic: int = 800  # penalty for topic overflow
    P_coach: int = 600  # penalty for coach overflow
    
    def validate(self) -> bool:
        """Validate configuration."""
        if self.dept_min_mode not in ("soft", "hard"):
            raise ValueError(f"Invalid dept_min_mode: {self.dept_min_mode}")
        if self.dept_max_mode not in ("soft", "hard"):
            raise ValueError(f"Invalid dept_max_mode: {self.dept_max_mode}")
        if self.P_dept_shortfall <= 0 or self.P_topic <= 0 or self.P_coach <= 0 or self.P_dept_overflow <= 0:
            raise ValueError("Penalties must be positive")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> CapacityConfig:
        """Create from dictionary with defaults."""
        return CapacityConfig(
            enable_topic_overflow=d.get("enable_topic_overflow", True),
            enable_coach_overflow=d.get("enable_coach_overflow", True),
            dept_min_mode=d.get("dept_min_mode", "soft"),
            dept_max_mode=d.get("dept_max_mode", "soft"),
            P_dept_shortfall=d.get("P_dept_shortfall", 1000),
            P_dept_overflow=d.get("P_dept_overflow", 1200),
            P_topic=d.get("P_topic", 800),
            P_coach=d.get("P_coach", 600),
        )


@dataclass
class SolverConfig:
    """Solver parameters."""
    algorithm: str = "ilp"  # "ilp", "flow", "hybrid"
    time_limit_sec: Optional[int] = None
    random_seed: Optional[int] = None
    epsilon_suboptimal: Optional[float] = None  # e.g., 0.05 for 5% tolerance
    
    def validate(self) -> bool:
        """Validate configuration."""
        if self.algorithm not in ("ilp", "flow", "hybrid"):
            raise ValueError(f"Invalid algorithm: {self.algorithm}")
        if self.time_limit_sec is not None and self.time_limit_sec <= 0:
            raise ValueError("time_limit_sec must be positive")
        if self.epsilon_suboptimal is not None and (self.epsilon_suboptimal < 0 or self.epsilon_suboptimal >= 1):
            raise ValueError("epsilon_suboptimal must be in [0, 1)")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> SolverConfig:
        """Create from dictionary with defaults."""
        return SolverConfig(
            algorithm=d.get("algorithm", "ilp"),
            time_limit_sec=d.get("time_limit_sec"),
            random_seed=d.get("random_seed"),
            epsilon_suboptimal=d.get("epsilon_suboptimal"),
        )


@dataclass
class AllocationConfig:
    """Complete allocation configuration."""
    preference: PreferenceConfig = field(default_factory=PreferenceConfig)
    capacity: CapacityConfig = field(default_factory=CapacityConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    
    def validate(self) -> bool:
        """Validate all sub-configurations."""
        self.capacity.validate()
        self.solver.validate()
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "preference": self.preference.to_dict(),
            "capacity": self.capacity.to_dict(),
            "solver": self.solver.to_dict(),
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        Path(path).write_text(self.to_json())
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> AllocationConfig:
        """Create from dictionary with defaults."""
        pref_dict = d.get("preference", {})
        cap_dict = d.get("capacity", {})
        solver_dict = d.get("solver", {})
        
        return AllocationConfig(
            preference=PreferenceConfig.from_dict(pref_dict),
            capacity=CapacityConfig.from_dict(cap_dict),
            solver=SolverConfig.from_dict(solver_dict),
        )
    
    @staticmethod
    def from_json(json_str: str) -> AllocationConfig:
        """Create from JSON string."""
        return AllocationConfig.from_dict(json.loads(json_str))
    
    @staticmethod
    def load_json(path: str) -> AllocationConfig:
        """Load configuration from JSON file."""
        content = Path(path).read_text()
        return AllocationConfig.from_json(content)
    
    @staticmethod
    def default() -> AllocationConfig:
        """Get default configuration."""
        return AllocationConfig()
