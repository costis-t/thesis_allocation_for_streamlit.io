from __future__ import annotations
from typing import Dict, Tuple, List, Optional
from .entities import Student, Topic


class PreferenceModelConfig:
    def __init__(
        self,
        allow_unranked: bool = True,
        tier2_cost: int = 1,
        tier3_cost: int = 5,
        unranked_cost: int = 200,
        top2_bias: bool = True,
        # Ranked choice costs (configurable)
        rank1_cost: int = 0,
        rank2_cost: int = 1,
        rank3_cost: int = 100,
        rank4_cost: int = 101,
        rank5_cost: int = 102
    ):
        self.allow_unranked = allow_unranked
        self.tier2_cost = tier2_cost
        self.tier3_cost = tier3_cost
        self.unranked_cost = unranked_cost
        self.top2_bias = top2_bias
        self.rank1_cost = rank1_cost
        self.rank2_cost = rank2_cost
        self.rank3_cost = rank3_cost
        self.rank4_cost = rank4_cost
        self.rank5_cost = rank5_cost


class PreferenceModel:
    """
    Computes edge costs per (student, topic).
    Precedence: overrides > tiers > ranks > unranked; banned => no edge.
    """

    def __init__(self, topics: Dict[str, Topic], overrides: Dict[Tuple[str, str], int] | None, cfg: PreferenceModelConfig):
        self.topics = topics
        self.overrides = overrides or {}
        self.cfg = cfg

    def _rank_cost(self, rank: int) -> int:
        """Calculate cost for a specific rank using configurable costs."""
        if self.cfg.top2_bias:
            # Use configurable ranked choice costs
            if rank == 1: return self.cfg.rank1_cost
            if rank == 2: return self.cfg.rank2_cost
            if rank == 3: return self.cfg.rank3_cost
            if rank == 4: return self.cfg.rank4_cost
            if rank == 5: return self.cfg.rank5_cost
            # For ranks beyond 5, use a formula based on rank5_cost
            return self.cfg.rank5_cost + (rank - 5)
        else:
            # Use configurable ranked choice costs even when top2_bias is False
            if rank == 1: return self.cfg.rank1_cost
            if rank == 2: return self.cfg.rank2_cost
            if rank == 3: return self.cfg.rank3_cost
            if rank == 4: return self.cfg.rank4_cost
            if rank == 5: return self.cfg.rank5_cost
            # For ranks beyond 5, use a formula based on rank5_cost
            return self.cfg.rank5_cost + (rank - 5)

    def compute_costs(self, students: Dict[str, Student]) -> Dict[Tuple[str, str], int]:
        costs: Dict[Tuple[str, str], int] = {}
        topic_ids = list(self.topics.keys())

        for student, s in students.items():
            if not s.plan:
                continue

            # NEW: forced_topic takes absolute precedence
            if s.forced_topic:
                if s.forced_topic in topic_ids and s.forced_topic not in s.banned:
                    costs[(student, s.forced_topic)] = -10000  # Very high priority (large negative cost = maximize this)
                continue

            # quick lookup
            rank_index = {t: i+1 for i, t in enumerate(s.ranks)} if s.ranks else {}

            for tid in topic_ids:
                # banned -> skip
                if tid in s.banned:
                    continue
                # override
                if (student, tid) in self.overrides:
                    costs[(student, tid)] = self.overrides[(student, tid)]
                    continue

                # tiers
                if s.tiers:
                    if tid in s.tiers.get(1, []):
                        costs[(student, tid)] = 0
                        continue
                    if tid in s.tiers.get(2, []):
                        costs[(student, tid)] = self.cfg.tier2_cost
                        continue
                    if tid in s.tiers.get(3, []):
                        costs[(student, tid)] = self.cfg.tier3_cost
                        continue

                # ranks
                if tid in rank_index:
                    costs[(student, tid)] = self._rank_cost(rank_index[tid])
                    continue

                # unranked fallback
                if self.cfg.allow_unranked:
                    costs[(student, tid)] = self.cfg.unranked_cost
                # else: no edge

        return costs

    @staticmethod
    def derive_preference_rank(student: Student, topic_id: str) -> int:
        """
        For reporting (non-colliding values):
          - forced: -1
          - tiers: 0 (tier1), 1 (tier2), 2 (tier3)
          - ranks: 10 (1st), 11 (2nd), 12 (3rd), 13 (4th), 14 (5th)
          - unranked: 999
        """
        if student.forced_topic and topic_id == student.forced_topic:
            return -1
        if student.tiers:
            if topic_id in student.tiers.get(1, []): return 0
            if topic_id in student.tiers.get(2, []): return 1
            if topic_id in student.tiers.get(3, []): return 2
        if topic_id in student.ranks:
            rank_idx = student.ranks.index(topic_id) + 1
            return 9 + rank_idx  # 10, 11, 12, 13, 14
        return 999

    @staticmethod
    def analyze_capacity_issues(
        student: Student,
        assigned_topic_id: str,
        higher_preferences: List[str],
        topics: Dict[str, Topic],
        topic_overflow: Dict[str, int] | None = None,
        coach_overflow: Dict[str, int] | None = None,
        dept_shortfall: Dict[str, int] | None = None
    ) -> Dict[str, bool]:
        """
        Analyze which capacity constraints affected higher preferences.
        
        Args:
            student: The student entity
            assigned_topic_id: Topic they were assigned to
            higher_preferences: List of topics with higher preference
            topics: Dict of all topics
            topic_overflow: Dict of topics that went over capacity (topic_id -> overflow amount)
            coach_overflow: Dict of coaches that went over capacity (coach_id -> overflow amount)
            dept_shortfall: Dict of departments that fell short (dept_id -> shortfall amount)
        
        Returns:
            Dict with keys: 'topic_capacity', 'coach_capacity', 'dept_minimum'
            Values: True if that constraint affected higher preferences
        """
        result = {
            'topic_capacity': False,
            'coach_capacity': False,
            'dept_minimum': False
        }
        
        topic_overflow = topic_overflow or {}
        coach_overflow = coach_overflow or {}
        dept_shortfall = dept_shortfall or {}
        
        # If no overflow data is available but higher preferences exist,
        # assume they were unavailable due to capacity (conservative estimate)
        # Check if there's ACTUAL overflow data (values > 0), not just dict presence
        has_any_actual_overflow = (
            any(v > 0 for v in topic_overflow.values()) or
            any(v > 0 for v in coach_overflow.values()) or
            any(v > 0 for v in dept_shortfall.values())
        )
        
        # Check if any higher preference topic had capacity issues
        for higher_topic_id in higher_preferences:
            if higher_topic_id not in topics:
                continue
            
            # Check topic capacity
            if higher_topic_id in topic_overflow and topic_overflow[higher_topic_id] > 0:
                result['topic_capacity'] = True
            
            # Check coach capacity (coach serving this topic)
            topic = topics[higher_topic_id]
            coach_id = topic.coach_id
            if coach_id in coach_overflow and coach_overflow[coach_id] > 0:
                result['coach_capacity'] = True
            
            # Check department shortfall
            dept_id = topic.department_id
            if dept_id in dept_shortfall and dept_shortfall[dept_id] > 0:
                result['dept_minimum'] = True
        
        # Fallback: if no explicit overflow data but student didn't get higher preference,
        # mark all constraints as possible factors
        if not has_any_actual_overflow and higher_preferences:
            # This is a conservative estimate - we know higher prefs weren't available
            # but can't determine exactly why, so we hint at all constraints
            result['topic_capacity'] = True
            result['coach_capacity'] = True
        
        return result

    @staticmethod
    def generate_allocation_explanation(
        student: Student,
        assigned_topic_id: str,
        preference_rank: int,
        topics: Dict[str, Topic],
        costs: Dict[Tuple[str, str], int],
        capacity_issues: Dict[str, bool] | None = None
    ) -> str:
        """
        Generate a verbose explanation for why a student was assigned to their assigned topic
        instead of higher preferences.
        
        Args:
            student: The student entity
            assigned_topic_id: The topic they were actually assigned to
            preference_rank: Their preference rank for the assigned topic
            topics: Dict of all topics
            costs: Dict of computed costs for this student's assignments
            capacity_issues: Dict indicating which capacity limits caused unavailability
                           Keys: 'topic_capacity', 'coach_capacity', 'dept_minimum'
                           Values: True if this constraint was a factor
        
        Returns:
            A verbose, human-readable explanation
        """
        capacity_issues = capacity_issues or {}
        
        # Helper to build capacity constraint message
        def build_capacity_msg():
            constraints = []
            if capacity_issues.get('topic_capacity'):
                constraints.append("topic capacity")
            if capacity_issues.get('coach_capacity'):
                constraints.append("coach capacity")
            if capacity_issues.get('dept_minimum'):
                constraints.append("department minimum constraints")
            
            if not constraints:
                return "capacity constraints"
            elif len(constraints) == 1:
                return f"{constraints[0]} limits"
            else:
                return f"{', '.join(constraints)} limits"
        
        # Case 1: Forced assignment
        if student.forced_topic:
            return (f"Forced assignment to {assigned_topic_id}. "
                   f"This student was administratively assigned to this topic regardless of preferences.")
        
        # Case 2: First choice and got it
        if preference_rank == 10:  # 1st choice
            return (f"Successfully assigned to 1st choice preference ({assigned_topic_id}). "
                   f"Student's top preference was available and allocated.")
        
        # Case 3: Second choice or higher preference
        if preference_rank == 11:  # 2nd choice
            higher_prefs = student.ranks[:1] if student.ranks else []
            if higher_prefs:
                capacity_msg = build_capacity_msg()
                return (f"Assigned to 2nd choice preference ({assigned_topic_id}). "
                       f"Higher preference ({higher_prefs[0]}) was unavailable due to {capacity_msg}.")
            return f"Assigned to 2nd choice preference ({assigned_topic_id})."
        
        if preference_rank in [12, 13, 14]:  # 3rd-5th choice
            choice_num = preference_rank - 9
            higher_prefs = student.ranks[:choice_num-1] if student.ranks else []
            if higher_prefs:
                higher_str = ", ".join(higher_prefs)
                capacity_msg = build_capacity_msg()
                suffix = 'st' if choice_num == 1 else 'nd' if choice_num == 2 else 'rd' if choice_num == 3 else 'th'
                return (f"Assigned to {choice_num}{suffix} choice preference ({assigned_topic_id}). "
                       f"Higher preferences ({higher_str}) were unavailable due to {capacity_msg}.")
            suffix = 'st' if choice_num == 1 else 'nd' if choice_num == 2 else 'rd' if choice_num == 3 else 'th'
            return f"Assigned to {choice_num}{suffix} choice preference ({assigned_topic_id})."
        
        # Case 4: Tiered assignment
        if preference_rank == 0:  # Tier 1
            return (f"Assigned to Tier 1 topic ({assigned_topic_id}). "
                   f"This is a primary preference tier for this student.")
        if preference_rank == 1:  # Tier 2
            tier1_topics = student.tiers.get(1, []) if student.tiers else []
            if tier1_topics:
                capacity_msg = build_capacity_msg()
                return (f"Assigned to Tier 2 topic ({assigned_topic_id}). "
                       f"Tier 1 preferences ({', '.join(tier1_topics)}) were unavailable due to {capacity_msg}.")
            return f"Assigned to Tier 2 topic ({assigned_topic_id})."
        if preference_rank == 2:  # Tier 3
            tier1_topics = student.tiers.get(1, []) if student.tiers else []
            tier2_topics = student.tiers.get(2, []) if student.tiers else []
            unavail = []
            if tier1_topics:
                unavail.append(f"Tier 1 ({', '.join(tier1_topics)})")
            if tier2_topics:
                unavail.append(f"Tier 2 ({', '.join(tier2_topics)})")
            if unavail:
                capacity_msg = build_capacity_msg()
                return (f"Assigned to Tier 3 topic ({assigned_topic_id}). "
                       f"{' and '.join(unavail)} were unavailable due to {capacity_msg}.")
            return f"Assigned to Tier 3 topic ({assigned_topic_id})."
        
        # Case 5: Unranked (fallback)
        if preference_rank == 999:
            prefs_str = ", ".join(student.ranks) if student.ranks else "None listed"
            capacity_msg = build_capacity_msg()
            return (f"Assigned to unranked topic ({assigned_topic_id}). "
                   f"Student did not list this topic in their preferences ({prefs_str}). "
                   f"All ranked preferences were unavailable due to {capacity_msg}.")
        
        # Fallback
        return f"Assigned to {assigned_topic_id}."
