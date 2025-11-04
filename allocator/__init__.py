# thesis_allocator/allocator/__init__.py

"""
Allocator package: provides classes and functions for thesis topic allocation.
"""

from .data_repository import DataRepository
from .preference_model import PreferenceModel, PreferenceModelConfig
from .allocation_model_ilp import AllocationModelILP, AllocationConfig
from .allocation_model_flow import AllocationModelFlow
from .outputs import write_allocation_csv, write_summary_txt

__all__ = [
    "DataRepository",
    "PreferenceModel",
    "PreferenceModelConfig",
    "AllocationModelILP",
    "AllocationModelFlow",
    "AllocationConfig",
    "write_allocation_csv",
    "write_summary_txt",
]
