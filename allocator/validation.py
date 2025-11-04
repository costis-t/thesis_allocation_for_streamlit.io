"""
Validation module for thesis allocation inputs.
Provides comprehensive checks and clear error reporting.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from .entities import Student, Topic, Coach, Department


@dataclass
class ValidationError:
    """Represents a validation error with context."""
    severity: str  # "error", "warning"
    message: str
    context: Dict[str, str] = None
    
    def __str__(self) -> str:
        msg = f"[{self.severity.upper()}] {self.message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            msg += f" ({context_str})"
        return msg


class InputValidator:
    """Validates all input data for consistency and integrity."""
    
    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
    
    def clear(self) -> None:
        """Clear previous validation results."""
        self.errors = []
        self.warnings = []
    
    def validate_all(
        self,
        students: Dict[str, Student],
        topics: Dict[str, Topic],
        coaches: Dict[str, Coach],
        departments: Dict[str, Department],
    ) -> Tuple[bool, List[ValidationError]]:
        """
        Run all validation checks.
        Returns (is_valid, list_of_errors_and_warnings)
        """
        self.clear()
        
        # Entity-level validation
        self._validate_entities(students, topics, coaches, departments)
        
        # Cross-entity consistency
        self._validate_consistency(students, topics, coaches, departments)
        
        return len(self.errors) == 0, self.errors + self.warnings
    
    def _validate_entities(
        self,
        students: Dict[str, Student],
        topics: Dict[str, Topic],
        coaches: Dict[str, Coach],
        departments: Dict[str, Department],
    ) -> None:
        """Validate individual entities."""
        # Validate students
        for sid, student in students.items():
            if not student.is_valid():
                self.errors.append(ValidationError(
                    severity="error",
                    message=f"Student has forced_topic in banned list",
                    context={"student_id": sid, "forced_topic": student.forced_topic}
                ))
            
            if student.forced_topic and student.forced_topic not in topics:
                self.errors.append(ValidationError(
                    severity="error",
                    message=f"Student's forced_topic does not exist",
                    context={"student_id": sid, "forced_topic": student.forced_topic}
                ))
        
        # Validate topics
        for tid, topic in topics.items():
            if not topic.is_valid():
                self.errors.append(ValidationError(
                    severity="error",
                    message=f"Topic has invalid data",
                    context={"topic_id": tid, "cap": str(topic.topic_cap)}
                ))
        
        # Validate coaches
        for cid, coach in coaches.items():
            if not coach.is_valid():
                self.errors.append(ValidationError(
                    severity="error",
                    message=f"Coach has invalid data",
                    context={"coach_id": cid, "cap": str(coach.coach_cap)}
                ))
        
        # Validate departments
        for did, dept in departments.items():
            if not dept.is_valid():
                self.errors.append(ValidationError(
                    severity="error",
                    message=f"Department has invalid data",
                    context={"department_id": did}
                ))
    
    def _validate_consistency(
        self,
        students: Dict[str, Student],
        topics: Dict[str, Topic],
        coaches: Dict[str, Coach],
        departments: Dict[str, Department],
    ) -> None:
        """Validate consistency across entities."""
        topic_ids = set(topics.keys())
        coach_ids = set(coaches.keys())
        dept_ids = set(departments.keys())
        
        # Validate topic-coach-department relationships
        for tid, topic in topics.items():
            if topic.coach_id not in coach_ids:
                self.errors.append(ValidationError(
                    severity="error",
                    message=f"Topic references non-existent coach",
                    context={"topic_id": tid, "coach_id": topic.coach_id}
                ))
            else:
                coach = coaches[topic.coach_id]
                if coach.department_id not in dept_ids:
                    self.errors.append(ValidationError(
                        severity="error",
                        message=f"Coach references non-existent department",
                        context={"coach_id": topic.coach_id, "department_id": coach.department_id}
                    ))
        
        # Validate student preferences reference existing topics
        for sid, student in students.items():
            if not student.plan:
                continue
            
            # Check ranks
            for pref_topic in student.ranks:
                if pref_topic not in topic_ids:
                    self.warnings.append(ValidationError(
                        severity="warning",
                        message=f"Student preference references non-existent topic",
                        context={"student_id": sid, "topic_id": pref_topic}
                    ))
            
            # Check tiers
            for tier_topics in student.tiers.values():
                for pref_topic in tier_topics:
                    if pref_topic not in topic_ids:
                        self.warnings.append(ValidationError(
                            severity="warning",
                            message=f"Student tier preference references non-existent topic",
                            context={"student_id": sid, "topic_id": pref_topic}
                        ))
            
            # Check banned
            for banned_topic in student.banned:
                if banned_topic not in topic_ids:
                    self.warnings.append(ValidationError(
                        severity="warning",
                        message=f"Student banned topic does not exist",
                        context={"student_id": sid, "topic_id": banned_topic}
                    ))
    
    def get_summary(self) -> str:
        """Get a summary of validation results."""
        error_count = len(self.errors)
        warning_count = len(self.warnings)
        
        if error_count == 0 and warning_count == 0:
            return "✓ All validations passed"
        
        summary = ""
        if error_count > 0:
            summary += f"✗ {error_count} error(s) found\n"
        if warning_count > 0:
            summary += f"⚠ {warning_count} warning(s) found"
        
        return summary.strip()
