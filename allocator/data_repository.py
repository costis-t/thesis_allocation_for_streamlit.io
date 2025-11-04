from __future__ import annotations
from typing import Dict, List, Tuple, Set
from .entities import Student, Topic, Coach, Department
from .utils import read_csv_rows, normalize_headers, split_pipe, to_int_or_zero


class DataRepository:
    """
    Loads and validates:
      - students.csv
      - capacities.csv
      - overrides.csv (optional)
    Produces canonical entity maps.
    """

    def __init__(self, students_path: str, capacities_path: str, overrides_path: str | None = None):
        self.students_path = students_path
        self.capacities_path = capacities_path
        self.overrides_path = overrides_path

        self.students: Dict[str, Student] = {}
        self.topics: Dict[str, Topic] = {}
        self.coaches: Dict[str, Coach] = {}
        self.departments: Dict[str, Department] = {}
        self.overrides: Dict[Tuple[str, str], int] = {}

    def load(self) -> None:
        self._load_capacities()
        self._load_students()
        if self.overrides_path:
            self._load_overrides()

    # ---------- internal loaders ----------

    def _load_capacities(self) -> None:
        rows = normalize_headers(read_csv_rows(self.capacities_path))
        # expected normalized columns:
        # topic_id, coach_id, maximum_students_per_topic, maximum_students_per_coach,
        # department_id, desired_minimum_by_department, desired_maximum_by_department

        if not rows:
            raise ValueError("capacities.csv is empty")

        # Build topics
        topics: Dict[str, Topic] = {}
        # Dedup for coaches and departments
        coach_caps: Dict[str, int] = {}
        coach_dept: Dict[str, str] = {}
        dept_min: Dict[str, int] = {}
        dept_max: Dict[str, int] = {}

        for r in rows:
            topic_id = r.get("topic_id", "").strip()
            coach_id = r.get("coach_id", "").strip()
            dept_id = r.get("department_id", "").strip()
            cap_topic = to_int_or_zero(r.get("maximum_students_per_topic", 0))
            cap_coach = to_int_or_zero(r.get("maximum_students_per_coach", 0))
            desired_min_dept = r.get("desired_minimum_by_department", "")
            desired_min_dept = to_int_or_zero(desired_min_dept) if str(desired_min_dept).strip() != "" else 0
            desired_max_dept = r.get("desired_maximum_by_department", "")
            desired_max_dept = to_int_or_zero(desired_max_dept) if str(desired_max_dept).strip() != "" else None

            if not topic_id or not coach_id or not dept_id:
                raise ValueError(f"capacities.csv: missing required fields on row: {r}")

            # Topic
            if topic_id in topics:
                # Defensive: topic should not repeat with conflicting data
                t = topics[topic_id]
                if t.coach_id != coach_id or t.department_id != dept_id or t.topic_cap != cap_topic:
                    raise ValueError(f"Inconsistent topic row for topic_id={topic_id}")
            else:
                topics[topic_id] = Topic(topic_id=topic_id, coach_id=coach_id, department_id=dept_id, topic_cap=cap_topic)

            # Coach total cap dedup & check
            if coach_id in coach_caps:
                if coach_caps[coach_id] != cap_coach:
                    raise ValueError(f"Inconsistent maximum_students_per_coach for coach {coach_id}")
            else:
                coach_caps[coach_id] = cap_coach

            # Coach department consistency
            if coach_id in coach_dept:
                if coach_dept[coach_id] != dept_id:
                    raise ValueError(f"Coach {coach_id} appears in multiple departments: {coach_dept[coach_id]} vs {dept_id}")
            else:
                coach_dept[coach_id] = dept_id

            # Department desired min dedup & check
            if dept_id in dept_min:
                if desired_min_dept and dept_min[dept_id] != desired_min_dept:
                    raise ValueError(f"Inconsistent Desired minimum by department for department {dept_id}")
            else:
                dept_min[dept_id] = desired_min_dept

            # Department desired max dedup & check
            if dept_id in dept_max:
                if desired_max_dept and dept_max[dept_id] != desired_max_dept:
                    raise ValueError(f"Inconsistent Desired maximum by department for department {dept_id}")
            else:
                dept_max[dept_id] = desired_max_dept

        # Build Coaches and Departments
        coaches = {cid: Coach(coach_id=cid, department_id=coach_dept[cid], coach_cap=coach_caps[cid]) for cid in coach_caps}
        departments = {did: Department(department_id=did, desired_min=dept_min.get(did, 0), desired_max=dept_max.get(did, None)) for did in dept_min}

        self.topics = topics
        self.coaches = coaches
        self.departments = departments

    def _load_students(self) -> None:
        rows = normalize_headers(read_csv_rows(self.students_path))
        students: Dict[str, Student] = {}
        for r in rows:
            student = (r.get("student_id") or "").strip()
            if not student:
                continue
            plan = (str(r.get("plan_thesis", "")).strip().lower() == "yes")

            # ranked prefs pref1..pref5
            ranks = []
            for i in range(1, 6):
                key = f"pref{i}"
                val = (r.get(key) or "").strip()
                if val:
                    ranks.append(val)

            # tiers
            tier1 = split_pipe(r.get("tier1", ""))
            tier2 = split_pipe(r.get("tier2", ""))
            tier3 = split_pipe(r.get("tier3", ""))
            tiers = {}
            if tier1: tiers[1] = tier1
            if tier2: tiers[2] = tier2
            if tier3: tiers[3] = tier3

            banned = set(split_pipe(r.get("banned", "")))
            
            # NEW: forced_topic
            forced_topic = (r.get("forced_topic", "") or "").strip() or None

            students[student] = Student(
                student=student,
                plan=plan,
                tiers=tiers,
                ranks=ranks,
                banned=banned,
                forced_topic=forced_topic
            )
        self.students = students

    def _load_overrides(self) -> None:
        rows = normalize_headers(read_csv_rows(self.overrides_path))
        overrides = {}
        for r in rows:
            student = (r.get("student_id") or "").strip()
            topic_id = (r.get("topic_id") or "").strip()
            try:
                cost = int(str(r.get("cost")).strip())
            except Exception:
                continue
            if student and topic_id:
                overrides[(student, topic_id)] = cost
        self.overrides = overrides
