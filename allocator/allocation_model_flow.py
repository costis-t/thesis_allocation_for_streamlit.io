"""
Min-Cost Max-Flow solver for thesis allocation.
Models the problem as a flow network where each student must reach exactly one topic.
"""
from __future__ import annotations
from typing import Dict, Tuple, List, Set
from collections import defaultdict
import networkx as nx

from .entities import Student, Topic, Coach, Department, AssignmentRow
from .preference_model import PreferenceModel, PreferenceModelConfig
from .allocation_model_ilp import AllocationConfig


class AllocationModelFlow:
    """
    Min-Cost Max-Flow solver:
    Network: Source → Students → Topics → Coaches → Sink
    Each student must be matched to exactly one topic with constraints.
    """

    def __init__(
        self,
        students: Dict[str, Student],
        topics: Dict[str, Topic],
        coaches: Dict[str, Coach],
        departments: Dict[str, Department],
        pref_model: PreferenceModel,
        cfg: AllocationConfig
    ):
        self.students = {k: v for k, v in students.items() if v.plan}
        self.topics = topics
        self.coaches = coaches
        self.departments = departments
        self.pref_model = pref_model
        self.cfg = cfg

        self.graph: nx.DiGraph | None = None
        self.unassignable_students: List[str] = []
        self.costs: Dict[Tuple[str, str], int] = {}

    def build(self) -> None:
        """Build the flow network."""
        self.costs = self.pref_model.compute_costs(self.students)
        
        # Find unassignable students
        admissible_by_student: Dict[str, List[str]] = defaultdict(list)
        for (student, tid) in self.costs.keys():
            admissible_by_student[student].append(tid)

        self.unassignable_students = [
            s.student for s in self.students.values()
            if s.student not in admissible_by_student
        ]

        # Build network
        self.graph = nx.DiGraph()

        # Nodes: source, sink, students, topics, coaches
        SOURCE = "source"
        SINK = "sink"
        self.graph.add_node(SOURCE)
        self.graph.add_node(SINK)

        # Source → Students (each student must get exactly 1 topic)
        for sid in self.students.keys():
            if sid in self.unassignable_students:
                continue
            self.graph.add_edge(SOURCE, f"student_{sid}", capacity=1, weight=0)

        # Students → Topics (preference costs)
        for (sid, tid), cost in self.costs.items():
            self.graph.add_edge(f"student_{sid}", f"topic_{tid}", capacity=1, weight=cost)

        # Topics → Coaches (topic capacity - NO OVERFLOW for flow)
        # Flow networks use hard constraints, not penalties
        for tid, topic in self.topics.items():
            cap = topic.topic_cap  # Hard limit, no overflow
            self.graph.add_edge(f"topic_{tid}", f"coach_{topic.coach_id}", 
                              capacity=cap, weight=0)

        # Coaches → Departments (coach capacity - NO OVERFLOW for flow)
        for cid, coach in self.coaches.items():
            cap = coach.coach_cap  # Hard limit, no overflow
            self.graph.add_edge(f"coach_{cid}", f"dept_{coach.department_id}", 
                              capacity=cap, weight=0)

        # Departments → Sink (department capacity - NO OVERFLOW for flow)
        for did, dept in self.departments.items():
            # For department maximum, we need to consider both minimum and maximum
            # If there's a maximum constraint, use it as the capacity
            if dept.desired_max is not None:
                cap = dept.desired_max
            else:
                # If no maximum, use a very large capacity
                cap = sum(topic.topic_cap for topic in self.topics.values() 
                         if topic.department_id == did)
            self.graph.add_edge(f"dept_{did}", SINK, capacity=cap, weight=0)

    def solve(self) -> Tuple[List[AssignmentRow], Dict]:
        """Solve using min-cost max-flow."""
        if self.graph is None:
            raise RuntimeError("Model not built. Call build() first.")

        try:
            # Solve min-cost max-flow (returns only flow dict in this version)
            flow_dict = nx.max_flow_min_cost(self.graph, "source", "sink")
        except nx.NetworkXError as e:
            raise RuntimeError(f"Flow solver failed: {e}")

        # Extract assignments from flow
        rows: List[AssignmentRow] = []
        assigned_students: Set[str] = set()

        for sid in self.students.keys():
            if sid in self.unassignable_students:
                continue

            # Find which topic this student was assigned to
            student_node = f"student_{sid}"
            assigned_topic = None

            if student_node in flow_dict:
                for topic_node, flow_amount in flow_dict[student_node].items():
                    if flow_amount > 0 and topic_node.startswith("topic_"):
                        tid = topic_node.replace("topic_", "")
                        assigned_topic = tid
                        break

            if assigned_topic:
                topic = self.topics[assigned_topic]
                student = self.students[sid]
                rank = PreferenceModel.derive_preference_rank(student, assigned_topic)
                eff_cost = self.costs.get((sid, assigned_topic), 999)
                
                # Check minimum preference requirement (if configured)
                if self.cfg.min_acceptable_preference_rank is not None:
                    min_rank = self.cfg.min_acceptable_preference_rank
                    # Allow if: forced (-1), tier 1/2/3 (0/1/2), or ranked at acceptable level
                    is_acceptable = (rank == -1 or rank <= 2 or 
                                   (10 <= rank <= 14 and rank <= min_rank))
                    if not is_acceptable:
                        # Skip this assignment if it doesn't meet minimum (shouldn't happen if ILP works)
                        continue
                
                # Check maximum preference requirement (if configured)
                if self.cfg.max_acceptable_preference_rank is not None:
                    max_rank = self.cfg.max_acceptable_preference_rank
                    # Allow if: forced (-1), tier 1/2/3 (0/1/2), or ranked at acceptable level
                    is_acceptable = (rank == -1 or rank <= 2 or 
                                   (10 <= rank <= 14 and rank >= max_rank))
                    if not is_acceptable:
                        # Skip this assignment if it doesn't meet maximum
                        continue
                
                # Check excluded preference ranks (if configured)
                if self.cfg.excluded_preference_ranks:
                    # Allow if: NOT in excluded list (or tier/forced)
                    is_acceptable = (rank not in self.cfg.excluded_preference_ranks or 
                                   rank <= 2 or rank == -1)
                    if not is_acceptable:
                        # Skip this assignment if it's in excluded list
                        continue
                
                # Get higher preferences for this student
                higher_prefs = []
                if rank == 11 and student.ranks:  # 2nd choice
                    higher_prefs = student.ranks[:1]
                elif rank in [12, 13, 14] and student.ranks:  # 3rd-5th choice
                    choice_num = rank - 9
                    higher_prefs = student.ranks[:choice_num-1]
                
                # For flow model, we don't track individual overflow details
                # So we use a generic capacity_issues dict that indicates all constraints could apply
                capacity_issues_dict = {
                    'topic_capacity': len(higher_prefs) > 0,
                    'coach_capacity': len(higher_prefs) > 0,
                    'dept_minimum': len(higher_prefs) > 0,
                    'dept_maximum': len(higher_prefs) > 0
                }
                
                # Generate explanation
                explanation = PreferenceModel.generate_allocation_explanation(
                    student=student,
                    assigned_topic_id=assigned_topic,
                    preference_rank=rank,
                    topics=self.topics,
                    costs=self.costs,
                    capacity_issues=capacity_issues_dict
                )

                rows.append(AssignmentRow(
                    student=sid,
                    assigned_topic=assigned_topic,
                    assigned_coach=topic.coach_id,
                    department_id=topic.department_id,
                    preference_rank=rank,
                    effective_cost=eff_cost,
                    via_topic_overflow=0,  # Flow doesn't track this easily
                    via_coach_overflow=0,
                    explanation=explanation
                ))
                assigned_students.add(sid)

        # Unassigned students
        unassigned = [
            s.student for s in self.students.values()
            if s.student not in assigned_students
        ]

        # Compute objective value (sum of costs)
        objective_value = sum(
            self.graph[u][v].get("weight", 0) * flow_dict[u][v]
            for u in flow_dict for v in flow_dict[u] if flow_dict[u][v] > 0
        )

        diagnostics = {
            "status": "Optimal" if len(assigned_students) == len([s for s in self.students if s not in self.unassignable_students]) else "Suboptimal",
            "objective_value": objective_value,
            "unassignable_students": self.unassignable_students,
            "unassigned_after_solve": unassigned,
            "algorithm": "min-cost-max-flow",
        }

        return rows, diagnostics
