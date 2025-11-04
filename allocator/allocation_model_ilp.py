from __future__ import annotations
from typing import Dict, Tuple, List, Set
from collections import defaultdict
import itertools

import pulp  # PuLP ILP solver

from .entities import Student, Topic, Coach, Department, AssignmentRow
from .preference_model import PreferenceModel, PreferenceModelConfig


class AllocationConfig:
    def __init__(
        self,
        # preference model
        pref_cfg: PreferenceModelConfig = PreferenceModelConfig(),
        # department minimums: "soft" or "hard"
        dept_min_mode: str = "soft",
        # department maximums: "soft" or "hard"
        dept_max_mode: str = "soft",
        # overflow toggles and penalties
        enable_topic_overflow: bool = True,
        enable_coach_overflow: bool = True,
        P_dept_shortfall: int = 1000,
        P_dept_overflow: int = 1200,
        P_topic: int = 800,
        P_coach: int = 600,
        # solver
        time_limit_sec: int | None = None,
        random_seed: int | None = None,
        # near-optimal solutions
        epsilon_suboptimal: float | None = None,  # e.g., 0.05 for 5% above optimal
        # minimum preference satisfaction (new!)
        min_acceptable_preference_rank: int | None = None,  # e.g., 11 = at least 2nd choice
        # maximum preference satisfaction (new!)
        max_acceptable_preference_rank: int | None = None,  # e.g., 12 = no better than 3rd choice
        # exclude specific preference ranks (new!)
        excluded_preference_ranks: list | None = None,  # e.g., [10, 11] = exclude 1st and 2nd choice
    ):
        assert dept_min_mode in ("soft", "hard")
        assert dept_max_mode in ("soft", "hard")
        self.pref_cfg = pref_cfg
        self.dept_min_mode = dept_min_mode
        self.dept_max_mode = dept_max_mode
        self.enable_topic_overflow = enable_topic_overflow
        self.enable_coach_overflow = enable_coach_overflow
        self.P_dept_shortfall = P_dept_shortfall
        self.P_dept_overflow = P_dept_overflow
        self.P_topic = P_topic
        self.P_coach = P_coach
        self.time_limit_sec = time_limit_sec
        self.random_seed = random_seed
        self.epsilon_suboptimal = epsilon_suboptimal
        self.min_acceptable_preference_rank = min_acceptable_preference_rank
        self.max_acceptable_preference_rank = max_acceptable_preference_rank
        self.excluded_preference_ranks = excluded_preference_ranks or []


class AllocationModelILP:
    """
    ILP:
      min Σ cost[s,t] x[s,t] + penalties
      s.t. one topic per student; topic and coach caps (with overflow); dept mins (soft/hard)
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

        self._model: pulp.LpProblem | None = None
        self._x: Dict[Tuple[str, str], pulp.LpVariable] = {}
        self._overflow_topic: Dict[str, pulp.LpVariable] = {}
        self._overflow_coach: Dict[str, pulp.LpVariable] = {}
        self._shortfall: Dict[str, pulp.LpVariable] = {}
        self.costs: Dict[Tuple[str, str], int] = {}

        # Derived indices
        self.topics_by_coach: Dict[str, List[str]] = defaultdict(list)
        self.topics_by_dept: Dict[str, List[str]] = defaultdict(list)
        for tid, t in topics.items():
            self.topics_by_coach[t.coach_id].append(tid)
            self.topics_by_dept[t.department_id].append(tid)

    def build(self) -> None:
        self.costs = self.pref_model.compute_costs(self.students)
        # track students that have at least one admissible topic
        admissible_by_student: Dict[str, List[str]] = defaultdict(list)
        for (student, tid) in self.costs.keys():
            admissible_by_student[student].append(tid)

        # Students with no admissible topics (e.g., banned all) – exclude and record
        self.unassignable_students: List[str] = [
            s.student for s in self.students.values()
            if s.student not in admissible_by_student
        ]

        # ILP
        self._model = pulp.LpProblem("thesis_allocation", pulp.LpMinimize)

        # Decision vars x[s,t] in {0,1}
        self._x = {
            (s, t): pulp.LpVariable(f"x__{sanitize(s)}__{sanitize(t)}", lowBound=0, upBound=1, cat="Binary")
            for (s, t) in self.costs.keys()
        }

        # Overflow vars (integers >= 0)
        if self.cfg.enable_topic_overflow:
            self._overflow_topic = {
                tid: pulp.LpVariable(f"ov_topic__{sanitize(tid)}", lowBound=0, cat="Integer")
                for tid in self.topics.keys()
            }
        if self.cfg.enable_coach_overflow:
            self._overflow_coach = {
                cid: pulp.LpVariable(f"ov_coach__{sanitize(cid)}", lowBound=0, cat="Integer")
                for cid in self.coaches.keys()
            }

        # Department shortfall (soft) or none (hard)
        if self.cfg.dept_min_mode == "soft":
            self._shortfall = {
                did: pulp.LpVariable(f"shortfall__{sanitize(did)}", lowBound=0, cat="Integer")
                for did in self.departments.keys()
            }
        
        # Department overflow (soft) or none (hard)
        self._overflow_dept: Dict[str, pulp.LpVariable] = {}
        if self.cfg.dept_max_mode == "soft":
            self._overflow_dept = {
                did: pulp.LpVariable(f"ov_dept__{sanitize(did)}", lowBound=0, cat="Integer")
                for did in self.departments.keys()
                if self.departments[did].desired_max is not None
            }

        # Objective
        obj_terms = []
        # preference costs
        for (s, t), c in self.costs.items():
            obj_terms.append(c * self._x[(s, t)])
        # penalties
        if self.cfg.dept_min_mode == "soft":
            for d in self.departments.values():
                obj_terms.append(self.cfg.P_dept_shortfall * self._shortfall[d.department_id])
        if self.cfg.dept_max_mode == "soft":
            for did, var in self._overflow_dept.items():
                obj_terms.append(self.cfg.P_dept_overflow * var)
        if self.cfg.enable_topic_overflow:
            for tid in self.topics.keys():
                obj_terms.append(self.cfg.P_topic * self._overflow_topic[tid])
        if self.cfg.enable_coach_overflow:
            for cid in self.coaches.keys():
                obj_terms.append(self.cfg.P_coach * self._overflow_coach[cid])

        self._model += pulp.lpSum(obj_terms)

        # Constraints

        # 1) One topic per student with admissible options
        for s in self.students.values():
            if s.student in self.unassignable_students:
                continue
            vars_for_s = [self._x[(s.student, t)] for t in admissible_by_student[s.student]]
            self._model += pulp.lpSum(vars_for_s) == 1, f"one_topic_{sanitize(s.student)}"

        # 1b) Minimum acceptable preference level (if configured)
        if self.cfg.min_acceptable_preference_rank is not None:
            min_rank = self.cfg.min_acceptable_preference_rank
            for s in self.students.values():
                if not s.plan:
                    continue
                # Find topics that meet the minimum preference
                acceptable_topics = []
                for t in self.topics.keys():
                    if (s.student, t) not in self.costs:
                        continue  # Not an option for this student
                    pref_rank = PreferenceModel.derive_preference_rank(s, t)
                    # Accept if: forced (-1), tier 1/2/3 (0/1/2), or ranked at acceptable level
                    if pref_rank == -1 or pref_rank <= 2 or (10 <= pref_rank <= 14 and pref_rank <= min_rank):
                        acceptable_topics.append(self._x[(s.student, t)])
                
                # At least one assignment must be from acceptable topics
                if acceptable_topics:
                    self._model += pulp.lpSum(acceptable_topics) >= 1, f"min_pref_{sanitize(s.student)}"

        # 1c) Maximum acceptable preference level (if configured)
        if self.cfg.max_acceptable_preference_rank is not None:
            max_rank = self.cfg.max_acceptable_preference_rank
            for s in self.students.values():
                if not s.plan:
                    continue
                # Find topics that meet the maximum preference
                acceptable_topics = []
                for t in self.topics.keys():
                    if (s.student, t) not in self.costs:
                        continue  # Not an option for this student
                    pref_rank = PreferenceModel.derive_preference_rank(s, t)
                    # Accept if: forced (-1), tier 1/2/3 (0/1/2), or ranked at acceptable level
                    if pref_rank == -1 or pref_rank <= 2 or (10 <= pref_rank <= 14 and pref_rank >= max_rank):
                        acceptable_topics.append(self._x[(s.student, t)])
                
                # At least one assignment must be from acceptable topics
                if acceptable_topics:
                    self._model += pulp.lpSum(acceptable_topics) >= 1, f"max_pref_{sanitize(s.student)}"

        # 1d) Exclude specific preference ranks (if configured)
        if self.cfg.excluded_preference_ranks:
            excluded = self.cfg.excluded_preference_ranks
            for s in self.students.values():
                if not s.plan:
                    continue
                # Find topics that are NOT in excluded ranks
                acceptable_topics = []
                for t in self.topics.keys():
                    if (s.student, t) not in self.costs:
                        continue  # Not an option for this student
                    pref_rank = PreferenceModel.derive_preference_rank(s, t)
                    # Accept if NOT in excluded list (tiers always ok)
                    if pref_rank not in excluded or pref_rank <= 2 or pref_rank == -1:
                        acceptable_topics.append(self._x[(s.student, t)])
                
                # At least one assignment must be from acceptable topics
                if acceptable_topics:
                    self._model += pulp.lpSum(acceptable_topics) >= 1, f"excl_pref_{sanitize(s.student)}"

        # 2) Topic capacities with overflow
        for tid, topic in self.topics.items():
            vars_to_t = [self._x[(s, t)] for (s, t) in self._x.keys() if t == tid]
            lhs = pulp.lpSum(vars_to_t)
            if self.cfg.enable_topic_overflow:
                lhs = lhs - self._overflow_topic[tid]  # sum x ≤ cap + ov  -> sum x - ov ≤ cap
            self._model += lhs <= topic.topic_cap, f"topic_cap_{sanitize(tid)}"

        # 3) Coach capacities with overflow
        for cid, coach in self.coaches.items():
            vars_for_c = [self._x[(s, t)] for (s, t) in self._x.keys() if self.topics[t].coach_id == cid]
            lhs = pulp.lpSum(vars_for_c)
            if self.cfg.enable_coach_overflow:
                lhs = lhs - self._overflow_coach[cid]
            self._model += lhs <= coach.coach_cap, f"coach_cap_{sanitize(cid)}"

        # 4) Department desired minimums (soft or hard)
        for did, dept in self.departments.items():
            if dept.desired_min <= 0:
                continue
            vars_for_d = [self._x[(s, t)] for (s, t) in self._x.keys() if self.topics[t].department_id == did]
            lhs = pulp.lpSum(vars_for_d)
            if self.cfg.dept_min_mode == "soft":
                lhs = lhs + self._shortfall[did]
                self._model += lhs >= dept.desired_min, f"dept_min_soft_{sanitize(did)}"
            else:  # hard
                self._model += lhs >= dept.desired_min, f"dept_min_hard_{sanitize(did)}"

        # 5) Department desired maximums (soft or hard)
        for did, dept in self.departments.items():
            if dept.desired_max is None:
                continue
            vars_for_d = [self._x[(s, t)] for (s, t) in self._x.keys() if self.topics[t].department_id == did]
            lhs = pulp.lpSum(vars_for_d)
            if self.cfg.dept_max_mode == "soft":
                lhs = lhs - self._overflow_dept[did]  # sum x ≤ max + ov  -> sum x - ov ≤ max
                self._model += lhs <= dept.desired_max, f"dept_max_soft_{sanitize(did)}"
            else:  # hard
                self._model += lhs <= dept.desired_max, f"dept_max_hard_{sanitize(did)}"

        # (Optional) solver config
        if self.cfg.time_limit_sec is not None:
            self._model.solver = pulp.PULP_CBC_CMD(timeLimit=self.cfg.time_limit_sec, msg=False)
        else:
            self._model.solver = pulp.PULP_CBC_CMD(msg=False)

    def solve(self) -> Tuple[List[AssignmentRow], Dict]:
        if self._model is None:
            raise RuntimeError("Model not built. Call build() first.")
        status = self._model.solve(self._model.solver)
        if pulp.LpStatus[status] not in ("Optimal", "Not Solved", "Infeasible", "Unbounded", "Undefined"):
            # Still proceed to extract if possible
            pass

        # If epsilon_suboptimal is set, constrain to near-optimal and resolve
        if self.cfg.epsilon_suboptimal is not None and pulp.LpStatus[status] == "Optimal":
            optimal_value = pulp.value(self._model.objective)
            epsilon_bound = optimal_value * (1 + self.cfg.epsilon_suboptimal)
            self._model += self._model.objective <= epsilon_bound, "epsilon_suboptimal_constraint"
            # Re-solve to find alternative near-optimal solution
            status = self._model.solve(self._model.solver)

        # Extract assignments
        x_val = {(s, t): int(round(var.value())) for (s, t), var in self._x.items()}
        topic_over = {tid: int(round(var.value())) for tid, var in self._overflow_topic.items()} if self._overflow_topic else {}
        coach_over = {cid: int(round(var.value())) for cid, var in self._overflow_coach.items()} if self._overflow_coach else {}
        shortfall = {did: int(round(var.value())) for did, var in self._shortfall.items()} if self._shortfall else {}
        dept_over = {did: int(round(var.value())) for did, var in self._overflow_dept.items()} if self._overflow_dept else {}

        # Build lookup for reporting
        by_student = defaultdict(list)
        for (s, t), v in x_val.items():
            if v == 1:
                by_student[s].append(t)

        # Check for potential non-uniqueness: students with tied costs for their assigned topic
        tied_students = []
        for student_id, ts in by_student.items():
            if not ts:
                continue
            tid = ts[0]
            assigned_cost = self.costs.get((student_id, tid), 999)
            # Find all topics with same cost for this student
            same_cost_topics = [t for t, c in self.costs.items() if t[0] == student_id and c == assigned_cost and t[1] != tid]
            if same_cost_topics:
                tied_students.append((student_id, tid, assigned_cost, [t[1] for t in same_cost_topics]))

        rows: List[AssignmentRow] = []
        assigned_students: Set[str] = set()

        for student_id, ts in by_student.items():
            if not ts:
                continue
            # By construction, one topic per student
            tid = ts[0]
            topic = self.topics[tid]
            student = self.students[student_id]
            rank = PreferenceModel.derive_preference_rank(student, tid)
            eff_cost = self.pref_model.overrides.get((student_id, tid), self.pref_model.compute_costs({student_id: student}).get((student_id, tid), 999))
            
            # Get higher preferences for this student
            higher_prefs = []
            if rank == 11 and student.ranks:  # 2nd choice
                higher_prefs = student.ranks[:1]
            elif rank in [12, 13, 14] and student.ranks:  # 3rd-5th choice
                choice_num = rank - 9
                higher_prefs = student.ranks[:choice_num-1]
            
            # Analyze which capacity constraints affected higher preferences
            capacity_issues_dict = PreferenceModel.analyze_capacity_issues(
                student=student,
                assigned_topic_id=tid,
                higher_preferences=higher_prefs,
                topics=self.topics,
                topic_overflow=topic_over,
                coach_overflow=coach_over,
                dept_shortfall=shortfall
            )
            
            # Generate explanation with capacity issue details
            explanation = PreferenceModel.generate_allocation_explanation(
                student=student,
                assigned_topic_id=tid,
                preference_rank=rank,
                topics=self.topics,
                costs=self.costs,
                capacity_issues=capacity_issues_dict
            )

            rows.append(AssignmentRow(
                student=student_id,
                assigned_topic=tid,
                assigned_coach=topic.coach_id,
                department_id=topic.department_id,
                preference_rank=rank,
                effective_cost=eff_cost,
                via_topic_overflow=1 if topic_over.get(tid, 0) > 0 else 0,
                via_coach_overflow=1 if coach_over.get(topic.coach_id, 0) > 0 else 0,
                explanation=explanation
            ))
            assigned_students.add(student_id)

        # Diagnostics
        unassigned = [
            s.student for s in self.students.values()
            if s.student not in assigned_students
        ]

        diagnostics = {
            "status": pulp.LpStatus[status],
            "objective_value": pulp.value(self._model.objective),
            "unassignable_students": getattr(self, "unassignable_students", []),
            "unassigned_after_solve": unassigned,
            "topic_overflow": topic_over,
            "coach_overflow": coach_over,
            "department_shortfall": shortfall,
            "department_overflow": dept_over,
            "tied_students": tied_students,
        }
        return rows, diagnostics


def sanitize(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_]+", "_", s)
