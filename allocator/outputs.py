from __future__ import annotations
import csv
from collections import Counter, defaultdict
from typing import List, Dict
from .entities import AssignmentRow, Topic, Coach, Department


def write_allocation_csv(path: str, rows: List[AssignmentRow]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "student", "assigned_topic", "assigned_coach", "department_id",
            "preference_rank", "effective_cost", "via_topic_overflow", "via_coach_overflow", "explanation"
        ])
        for r in rows:
            w.writerow([
                r.student, r.assigned_topic, r.assigned_coach, r.department_id,
                r.preference_rank, r.effective_cost, r.via_topic_overflow, r.via_coach_overflow, r.explanation
            ])


def write_summary_txt(
    path: str,
    rows: List[AssignmentRow],
    topics: Dict[str, Topic],
    coaches: Dict[str, Coach],
    departments: Dict[str, Department],
    diagnostics: Dict
) -> None:
    total_assigned = len(rows)
    pref_counts = Counter(r.preference_rank for r in rows)

    used_per_topic = Counter(r.assigned_topic for r in rows)
    used_per_coach = Counter(r.assigned_coach for r in rows)
    used_per_dept = Counter(r.department_id for r in rows)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Solver status: {diagnostics.get('status')}\n")
        f.write(f"Objective: {diagnostics.get('objective_value')}\n\n")

        unassignable = diagnostics.get("unassignable_students", [])
        unassigned_after = diagnostics.get("unassigned_after_solve", [])
        f.write(f"Unassignable students (no admissible topics): {len(unassignable)}\n")
        if unassignable:
            for e in unassignable:
                f.write(f"  - {e}\n")
        f.write(f"\nUnassigned after solve: {len(unassigned_after)}\n")
        if unassigned_after:
            for e in unassigned_after:
                f.write(f"  - {e}\n")

        # Uniqueness check
        tied = diagnostics.get("tied_students", [])
        f.write(f"\n--- SOLUTION UNIQUENESS ---\n")
        if not tied:
            f.write("✓ Solution appears UNIQUE (no ties in costs).\n")
        else:
            f.write(f"⚠ Solution may NOT be unique: {len(tied)} student(s) have equally-good alternatives:\n")
            for student_id, assigned_tid, cost, alt_tids in tied[:10]:  # Show first 10
                f.write(f"  {student_id}: assigned {assigned_tid} (cost={cost}), could also take: {alt_tids}\n")
            if len(tied) > 10:
                f.write(f"  ... and {len(tied) - 10} more students with tied costs.\n")

        f.write("\nPreference satisfaction:\n")
        f.write(f"  Tier1: {pref_counts.get(0,0)}\n")
        f.write(f"  Tier2: {pref_counts.get(1,0)}\n")
        f.write(f"  Tier3: {pref_counts.get(2,0)}\n")
        
        f.write("\nRanked choice satisfaction:\n")
        f.write(f"  1st choice: {pref_counts.get(10,0)}\n")
        f.write(f"  2nd choice: {pref_counts.get(11,0)}\n")
        f.write(f"  3rd choice: {pref_counts.get(12,0)}\n")
        f.write(f"  4th choice: {pref_counts.get(13,0)}\n")
        f.write(f"  5th choice: {pref_counts.get(14,0)}\n")
        f.write(f"  Unranked : {pref_counts.get(999,0)}\n")

        f.write("\nTopic utilization:\n")
        topic_over = diagnostics.get("topic_overflow", {})
        for tid, t in topics.items():
            used = used_per_topic.get(tid, 0)
            ov = topic_over.get(tid, 0)
            f.write(f"  {tid}: {used} / {t.topic_cap}" + (f"  (overflow={ov})" if ov else "") + "\n")

        f.write("\nCoach utilization:\n")
        coach_over = diagnostics.get("coach_overflow", {})
        for cid, c in coaches.items():
            used = used_per_coach.get(cid, 0)
            ov = coach_over.get(cid, 0)
            f.write(f"  {cid}: {used} / {c.coach_cap}" + (f"  (overflow={ov})" if ov else "") + "\n")

        f.write("\nDepartment totals:\n")
        dept_short = diagnostics.get("department_shortfall", {})
        for did, d in departments.items():
            used = used_per_dept.get(did, 0)
            line = f"  {did}: {used}"
            if d.desired_min:
                line += f" (desired_min={d.desired_min}"
                if dept_short:
                    line += f", shortfall={dept_short.get(did, 0)}"
                line += ")"
            f.write(line + "\n")
