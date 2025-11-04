from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional


@dataclass(frozen=True)
class Student:
    student: str
    plan: bool
    tiers: Dict[int, List[str]]  # 1 -> tier1 topics, 2 -> tier2, 3 -> tier3
    ranks: List[str]             # pref1..pref5 canonical topic_ids
    banned: Set[str]             # hard bans (canonical topic_ids)
    forced_topic: Optional[str] = None  # NEW: hard assignment to specific topic
    
    def is_valid(self) -> bool:
        """Validate that student data is internally consistent."""
        # forced_topic and banned should not overlap
        if self.forced_topic and self.banned:
            return self.forced_topic not in self.banned
        return True
    
    def get_admissible_topics(self, all_topic_ids: Set[str]) -> Set[str]:
        """Get topics available to this student based on preferences and constraints."""
        if self.forced_topic:
            return {self.forced_topic}
        
        # Exclude banned topics
        admissible = all_topic_ids - self.banned
        
        # If student has preferences (tiers or ranks), restrict to those
        if self.tiers or self.ranks:
            preferred = set()
            for tier_topics in self.tiers.values():
                preferred.update(tier_topics)
            preferred.update(self.ranks)
            # Intersection with admissible
            admissible = admissible & preferred
        
        return admissible


@dataclass(frozen=True)
class Topic:
    topic_id: str
    coach_id: str
    department_id: str
    topic_cap: int
    
    def is_valid(self) -> bool:
        """Validate topic data."""
        return self.topic_cap > 0 and self.topic_id and self.coach_id and self.department_id


@dataclass(frozen=True)
class Coach:
    coach_id: str
    department_id: str
    coach_cap: int
    
    def is_valid(self) -> bool:
        """Validate coach data."""
        return self.coach_cap > 0 and self.coach_id and self.department_id


@dataclass(frozen=True)
class Department:
    department_id: str
    desired_min: int  # can be 0 if not provided
    desired_max: Optional[int] = None  # can be None if not provided
    
    def is_valid(self) -> bool:
        """Validate department data."""
        if self.desired_min < 0:
            return False
        if self.desired_max is not None and self.desired_max < 0:
            return False
        if self.desired_max is not None and self.desired_min > self.desired_max:
            return False
        return bool(self.department_id)


@dataclass
class AssignmentRow:
    student: str
    assigned_topic: str
    assigned_coach: str
    department_id: str
    preference_rank: int   # 0/1/5 for tiered; 1..5 for ranks; 999 for unranked; -1 for forced
    effective_cost: int
    via_topic_overflow: int
    via_coach_overflow: int
    forced: bool = False  # NEW: indicates this was a forced assignment
    explanation: str = ""  # NEW: verbose explanation for why this allocation was made
