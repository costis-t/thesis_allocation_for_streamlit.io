#!/usr/bin/env python3
"""
Fast Grid Search with Tier Support (1/4 combinations)

This script implements a faster grid search (roughly 12,500 combinations) that explores
both rank costs AND tier costs. This is a separate tool from comprehensive_grid_search.py
to allow running both when needed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any
import itertools
from dataclasses import dataclass
import time
from datetime import datetime
import warnings
import multiprocessing as mp
from functools import partial
warnings.filterwarnings('ignore')

# Try to import tqdm, fallback to simple progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple progress bar without tqdm
    class tqdm:
        def __init__(self, iterable, desc="", unit="", ncols=100):
            self.iterable = iterable
            self.desc = desc
            self.unit = unit
            self.total = len(iterable) if hasattr(iterable, '__len__') else None
            self.n = 0
            self.start_time = time.time()
            
        def __iter__(self):
            return self
            
        def __next__(self):
            try:
                item = next(iter(self.iterable))
                self.n += 1
                self._print_progress()
                return item
            except StopIteration:
                self._print_final()
                raise
                
        def _print_progress(self):
            if self.n % 10 == 0 or self.n == self.total:
                elapsed = time.time() - self.start_time
                rate = self.n / elapsed if elapsed > 0 else 0
                remaining = (self.total - self.n) / rate if rate > 0 and self.total else 0
                percent = (self.n / self.total * 100) if self.total else 0
                print(f"\r{self.desc}: {self.n}/{self.total} ({percent:.1f}%) | "
                      f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s", end="", flush=True)
        
        def _print_final(self):
            elapsed = time.time() - self.start_time
            print(f"\r{self.desc}: Complete! Processed {self.n} items in {elapsed:.1f}s")
        
        def set_postfix(self, **kwargs):
            # Store for display
            self.postfix = kwargs
            
        def close(self):
            if hasattr(self, 'postfix'):
                print(f" | stats: {self.postfix}")

# Import our existing modules
import sys
sys.path.append('.')
from allocator.data_repository import DataRepository
from allocator.allocation_model_ilp import AllocationConfig as LegacyAllocationConfig
from allocator.preference_model import PreferenceModelConfig, PreferenceModel
from allocator.allocation_model_ilp import AllocationModelILP

# Worker function for multiprocessing
def run_single_allocation(args):
    """Run a single allocation - returns dict with result."""
    i, cost_combo, students_path, capacities_path = args
    
    try:
        # Load data
        repo = DataRepository(students_path, capacities_path)
        repo.load()
        
        # Create preference model configuration
        from allocator.preference_model import PreferenceModelConfig
        pref_cfg = PreferenceModelConfig(
            rank1_cost=cost_combo[0],
            rank2_cost=cost_combo[1],
            rank3_cost=cost_combo[2],
            rank4_cost=cost_combo[3],
            rank5_cost=cost_combo[4],
            tier2_cost=cost_combo[5],
            tier3_cost=cost_combo[6],
            top2_bias=False,
            unranked_cost=200
        )
        
        # Create preference model and solve
        pref_model = PreferenceModel(topics=repo.topics, overrides=None, cfg=pref_cfg)
        allocation_cfg = LegacyAllocationConfig(
            dept_min_mode="soft",
            dept_max_mode="soft",
            enable_topic_overflow=True,
            enable_coach_overflow=True,
            P_dept_shortfall=1000,
            P_dept_overflow=1200,
            P_topic=800,
            P_coach=600,
            time_limit_sec=30,
            epsilon_suboptimal=None,
            pref_cfg=pref_cfg
        )
        
        model = AllocationModelILP(
            students=repo.students,
            topics=repo.topics,
            coaches=repo.coaches,
            departments=repo.departments,
            pref_model=pref_model,
            cfg=allocation_cfg
        )
        
        model.build()
        rows, diagnostics = model.solve()
        
        allocation = {row.student: row.assigned_topic for row in rows}
        
        # Calculate metrics
        pref_satisfaction = _calculate_pref_satisfaction(repo, allocation)
        gini = _calculate_gini_for_allocation(allocation, repo)
        satisfaction_score = _calculate_satisfaction_score(pref_satisfaction)
        
        # Check success
        status = diagnostics.get('status', '')
        is_successful = status not in ['Infeasible', 'Undefined', 'Unbounded']
        
        # Debug: print first failure
        if not is_successful and i < 3:
            print(f"\n  ⚠️  First failure detected!")
            print(f"     Cost combo: {cost_combo}")
            print(f"     Status: {status}")
            print(f"     Diagnostics: {diagnostics.get('objective_value', 'N/A')}")
            print(f"     Allocation size: {len(allocation)}")
        
        return {
            'success': is_successful and len(allocation) > 0,
            'simulation_id': i,
            'cost_combo': cost_combo,
            'allocation': allocation,
            'preference_satisfaction': pref_satisfaction,
            'gini_coefficient': gini,
            'satisfaction_score': satisfaction_score,
            'diagnostics': diagnostics,
            'num_students': len(allocation),
            'timestamp': datetime.now().isoformat(),
            'status': status  # Add status for debugging
        }
    except Exception as e:
        if i < 3:
            print(f"\n  ❌ Exception in first few allocations: {str(e)[:100]}")
        return {
            'success': False,
            'simulation_id': i,
            'cost_combo': cost_combo,
            'error': str(e)[:100]
        }


def _calculate_pref_satisfaction(repo, allocation):
    """Calculate preference satisfaction."""
    satisfaction = defaultdict(int)
    for student_id, topic_id in allocation.items():
        student = repo.students.get(student_id)
        if not student:
            continue
        rank = PreferenceModel.derive_preference_rank(student, topic_id)
        if rank == -1: satisfaction['forced'] += 1
        elif rank == 0: satisfaction['tier1'] += 1
        elif rank == 1: satisfaction['tier2'] += 1
        elif rank == 2: satisfaction['tier3'] += 1
        elif rank == 10: satisfaction['rank1'] += 1
        elif rank == 11: satisfaction['rank2'] += 1
        elif rank == 12: satisfaction['rank3'] += 1
        elif rank == 13: satisfaction['rank4'] += 1
        elif rank == 14: satisfaction['rank5'] += 1
        elif rank == 999: satisfaction['unranked'] += 1
    return dict(satisfaction)


def _calculate_satisfaction_score(pref_satisfaction):
    """Calculate satisfaction score."""
    weights = {
        'forced': 6.0, 'tier1': 5.5, 'rank1': 5.0, 'rank2': 4.0, 'tier2': 3.5,
        'rank3': 3.0, 'tier3': 2.5, 'rank4': 2.0, 'rank5': 1.0, 'unranked': 0.0
    }
    total_weight = sum(weights.get(k, 0) * v for k, v in pref_satisfaction.items())
    total_students = sum(pref_satisfaction.values())
    max_score = total_students * 6.0
    return total_weight / max_score if max_score > 0 else 0.0


def _calculate_gini_for_allocation(allocation, repo):
    """Calculate Gini coefficient for coach load balance."""
    from collections import Counter
    import numpy as np
    
    # Count students per coach
    coach_counts = Counter()
    for student_id, topic_id in allocation.items():
        # Get the topic to find its coach_id
        topic = repo.topics.get(topic_id)
        if topic:
            coach_id = topic.coach_id
            coach_counts[coach_id] += 1
    
    if not coach_counts or len(coach_counts) < 2:
        return 0.0
    
    # Get counts and sort
    counts = sorted(coach_counts.values())
    n = len(counts)
    cumsum = sum(counts)
    
    if cumsum == 0:
        return 0.0
    
    # Calculate Gini coefficient
    gini = (2 * sum((i + 1) * count for i, count in enumerate(counts))) / (n * cumsum) - (n + 1) / n
    return abs(gini)


@dataclass
class GridSearchResult:
    """Container for grid search results with tier support."""
    cost_combination: Tuple[int, int, int, int, int, int, int]  # rank1-5, tier2, tier3
    satisfaction_score: float
    fairness_score: float
    total_cost: float
    preference_satisfaction: Dict[str, int]
    gini_coefficient: float
    num_students: int
    algorithm: str
    timestamp: str
    simulation_id: int

class ComprehensiveGridSearchWithTiers:
    """Implements fast grid search with tier cost support."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.results = []
        self.pareto_frontier = []
        self.simulation_count = 0
        self.start_time = time.time()
        
    def generate_grid_combinations(self) -> List[Tuple[int, int, int, int, int, int, int]]:
        """Generate systematic grid of cost combinations (ultra-fast, ~4-5 minutes)."""
        print("🚀 Generating ultra-fast grid combinations (with tier support, 4-5 min runtime)...")
        
        # Optimized for 4-5 minute execution
        # 6 values each for rank, 5 for tier = ~18-20K combinations
        rank1_range = [0, 5, 12, 25, 40, 50]    # 6 values
        rank2_range = [0, 5, 12, 25, 40, 50]    # 6 values
        rank3_range = [0, 25, 60, 100, 150, 200]  # 6 values
        rank4_range = [0, 25, 60, 100, 150, 200]  # 6 values
        rank5_range = [0, 25, 60, 100, 150, 200]  # 6 values
        
        # Add tier costs with reasonable ranges
        tier2_range = [0, 3, 8, 15, 20]          # 5 values
        tier3_range = [0, 8, 15, 25, 40]         # 5 values
        
        # Generate all combinations
        all_combinations = list(itertools.product(
            rank1_range, rank2_range, rank3_range, rank4_range, rank5_range,
            tier2_range, tier3_range
        ))
        
        # Filter to ensure monotonic costs: rank1 <= rank2 <= rank3 <= rank4 <= rank5
        combinations = [
            combo for combo in all_combinations
            if combo[0] <= combo[1] <= combo[2] <= combo[3] <= combo[4]
        ]
        
        print(f"📊 Generated {len(all_combinations):,} total combinations")
        print(f"   Filtered to {len(combinations):,} monotonic combinations")
        print(f"   • Rank1: {len(rank1_range)} values ({min(rank1_range)}-{max(rank1_range)})")
        print(f"   • Rank2: {len(rank2_range)} values ({min(rank2_range)}-{max(rank2_range)})")
        print(f"   • Rank3: {len(rank3_range)} values ({min(rank3_range)}-{max(rank3_range)})")
        print(f"   • Rank4: {len(rank4_range)} values ({min(rank4_range)}-{max(rank4_range)})")
        print(f"   • Rank5: {len(rank5_range)} values ({min(rank5_range)}-{max(rank5_range)})")
        print(f"   • Tier2: {len(tier2_range)} values ({min(tier2_range)}-{max(tier2_range)})")
        print(f"   • Tier3: {len(tier3_range)} values ({min(tier3_range)}-{max(tier3_range)})")
        
        return combinations
    
    def run_allocation_with_all_costs(
        self, 
        rank1_cost: int, rank2_cost: int, rank3_cost: int, 
        rank4_cost: int, rank5_cost: int, tier2_cost: int, tier3_cost: int,
        students_path: str, capacities_path: str, algorithm: str = "ilp"
    ) -> Dict[str, Any]:
        """Run allocation with both rank and tier costs."""
        try:
            # Load data repository
            repo = DataRepository(students_path, capacities_path)
            repo.load()
            
            # Create preference model configuration WITH tier costs
            pref_cfg = PreferenceModelConfig(
                rank1_cost=rank1_cost,
                rank2_cost=rank2_cost,
                rank3_cost=rank3_cost,
                rank4_cost=rank4_cost,
                rank5_cost=rank5_cost,
                tier2_cost=tier2_cost,
                tier3_cost=tier3_cost,
                top2_bias=False,
                unranked_cost=200
            )
            
            # Create preference model
            pref_model = PreferenceModel(topics=repo.topics, overrides=None, cfg=pref_cfg)
            
            # Create allocation configuration
            allocation_cfg = LegacyAllocationConfig(
                dept_min_mode="soft",
                dept_max_mode="soft",
                enable_topic_overflow=True,
                enable_coach_overflow=True,
                P_dept_shortfall=1000,
                P_dept_overflow=1200,
                P_topic=800,
                P_coach=600,
                time_limit_sec=300,
                epsilon_suboptimal=None,
                pref_cfg=pref_cfg
            )
            
            # Create and solve model
            if algorithm == "ilp":
                model = AllocationModelILP(
                    students=repo.students,
                    topics=repo.topics,
                    coaches=repo.coaches,
                    departments=repo.departments,
                    pref_model=pref_model,
                    cfg=allocation_cfg
                )
            
            model.build()
            rows, diagnostics = model.solve()
            
            # Convert rows to allocation
            allocation = {}
            for row in rows:
                allocation[row.student] = row.assigned_topic
            
            # Calculate metrics
            pref_satisfaction = self._calculate_preference_satisfaction(repo, allocation)
            gini = calculate_gini_coefficient(allocation, repo.coaches)
            
            # Accept any status that's not infeasible or undefined
            status = diagnostics.get('status', '')
            is_successful = status not in ['Infeasible', 'Undefined', 'Unbounded']
            
            return {
                'success': is_successful and len(allocation) > 0,
                'allocation': allocation,
                'preference_satisfaction': pref_satisfaction,
                'gini_coefficient': gini,
                'diagnostics': diagnostics,
                'num_students': len(allocation),
                'algorithm': algorithm,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_preference_satisfaction(self, repo, allocation):
        """Calculate preference satisfaction distribution."""
        satisfaction = defaultdict(int)
        
        for student_id, topic_id in allocation.items():
            student = repo.students.get(student_id)
            if not student:
                continue
            
            # Get preference rank
            rank = PreferenceModel.derive_preference_rank(student, topic_id)
            
            if rank == -1:  # forced
                satisfaction['forced'] += 1
            elif rank == 0:  # tier1
                satisfaction['tier1'] += 1
            elif rank == 1:  # tier2
                satisfaction['tier2'] += 1
            elif rank == 2:  # tier3
                satisfaction['tier3'] += 1
            elif rank == 10:  # rank1
                satisfaction['rank1'] += 1
            elif rank == 11:  # rank2
                satisfaction['rank2'] += 1
            elif rank == 12:  # rank3
                satisfaction['rank3'] += 1
            elif rank == 13:  # rank4
                satisfaction['rank4'] += 1
            elif rank == 14:  # rank5
                satisfaction['rank5'] += 1
            elif rank == 999:  # unranked
                satisfaction['unranked'] += 1
        
        return dict(satisfaction)
    
    def _calculate_satisfaction_score(self, pref_satisfaction: Dict[str, int]) -> float:
        """Calculate a single satisfaction score from preference satisfaction data."""
        if not pref_satisfaction:
            return 0.0
        
        # Weighted satisfaction score
        weights = {
            'forced': 6.0,
            'tier1': 5.5,
            'rank1': 5.0,
            'rank2': 4.0,
            'tier2': 3.5,
            'rank3': 3.0,
            'tier3': 2.5,
            'rank4': 2.0,
            'rank5': 1.0,
            'unranked': 0.0
        }
        
        total_weighted_satisfaction = 0
        total_students = 0
        
        for rank, count in pref_satisfaction.items():
            if rank in weights:
                total_weighted_satisfaction += weights[rank] * count
                total_students += count
        
        if total_students == 0:
            return 0.0
        
        # Normalize to 0-1 scale
        max_possible_score = total_students * weights['forced']
        return total_weighted_satisfaction / max_possible_score if max_possible_score > 0 else 0.0
    
    def run_grid_search(self, combinations: List[Tuple[int, int, int, int, int, int, int]], 
                       students_path: str, capacities_path: str, num_cores: int = 14) -> List[GridSearchResult]:
        """Run grid search with tier support using multiprocessing."""
        print(f"🚀 Starting parallel grid search with {len(combinations):,} combinations using {num_cores} cores...")
        
        # Prepare arguments for multiprocessing
        args = [(i, combo, students_path, capacities_path) for i, combo in enumerate(combinations)]
        
        # Run with multiprocessing
        successful_simulations = 0
        failed_simulations = 0
        
        print(f"   Processing {len(args):,} allocations across {num_cores} cores...")
        
        with mp.Pool(processes=num_cores) as pool:
            results_iter = pool.imap(run_single_allocation, args)
            results_list = []
            
            # Track progress
            for result in tqdm(results_iter, total=len(args), desc="Grid Search", unit=" combos", ncols=100):
                if result.get('success', False):
                    successful_simulations += 1
                    # Convert to GridSearchResult
                    grid_result = GridSearchResult(
                        cost_combination=result['cost_combo'],
                        satisfaction_score=result.get('satisfaction_score', 0.0),
                        fairness_score=1.0 - result.get('gini_coefficient', 0.0),
                        total_cost=0.0,  # Not calculated in worker
                        preference_satisfaction=result.get('preference_satisfaction', {}),
                        gini_coefficient=result.get('gini_coefficient', 0.0),
                        num_students=result.get('num_students', 0),
                        algorithm='ilp',
                        timestamp=result.get('timestamp', ''),
                        simulation_id=result.get('simulation_id', 0)
                    )
                    results_list.append(grid_result)
                else:
                    failed_simulations += 1
        
        self.results = results_list
        print(f"\n✅ Grid search complete!")
        print(f"   Successful: {successful_simulations:,}")
        print(f"   Failed: {failed_simulations:,}")
        if successful_simulations + failed_simulations > 0:
            print(f"   Success rate: {100*successful_simulations/(successful_simulations+failed_simulations):.1f}%")
        
        return results_list
    
    def find_pareto_frontier(self) -> List[GridSearchResult]:
        """Find Pareto-optimal solutions."""
        print("🔍 Finding Pareto frontier...")
        
        if not self.results:
            print("❌ No results to analyze")
            return []
        
        pareto_solutions = []
        
        for i, result_i in enumerate(self.results):
            is_pareto_optimal = True
            
            for j, result_j in enumerate(self.results):
                if i == j:
                    continue
                
                # Check if result_j dominates result_i
                if (result_j.satisfaction_score >= result_i.satisfaction_score and
                    result_j.fairness_score >= result_i.fairness_score and
                    (result_j.satisfaction_score > result_i.satisfaction_score or
                     result_j.fairness_score > result_i.fairness_score)):
                    is_pareto_optimal = False
                    break
            
            if is_pareto_optimal:
                pareto_solutions.append(result_i)
        
        self.pareto_frontier = pareto_solutions
        print(f"✅ Found {len(pareto_solutions)} Pareto-optimal solutions")
        
        # Recommend best solutions
        self._recommend_solutions()
        
        return pareto_solutions
    
    def _recommend_solutions(self):
        """Recommend best Pareto-optimal solutions based on different criteria."""
        if not self.pareto_frontier:
            return
        
        print("\n🎯 Recommended Solutions:")
        print("=" * 60)
        
        # 1. Best Combined Score (product of satisfaction and fairness)
        best_combined = max(self.pareto_frontier, 
                           key=lambda s: s.satisfaction_score * s.fairness_score)
        print(f"\n📊 Best Combined Score:")
        print(f"   Cost: {best_combined.cost_combination}")
        print(f"   → rank1={best_combined.cost_combination[0]}, rank2={best_combined.cost_combination[1]}, rank3={best_combined.cost_combination[2]}, rank4={best_combined.cost_combination[3]}, rank5={best_combined.cost_combination[4]}, tier2={best_combined.cost_combination[5]}, tier3={best_combined.cost_combination[6]}")
        print(f"   Satisfaction × Fairness = {best_combined.satisfaction_score * best_combined.fairness_score:.4f}")
        print(f"   Satisfaction: {best_combined.satisfaction_score:.3f}")
        print(f"   Fairness: {best_combined.fairness_score:.3f}")
        
        # 2. Best Satisfaction (prioritize student happiness)
        best_satisfaction = max(self.pareto_frontier, 
                               key=lambda s: s.satisfaction_score)
        print(f"\n😊 Best Satisfaction:")
        print(f"   Cost: {best_satisfaction.cost_combination}")
        print(f"   → rank1={best_satisfaction.cost_combination[0]}, rank2={best_satisfaction.cost_combination[1]}, rank3={best_satisfaction.cost_combination[2]}, rank4={best_satisfaction.cost_combination[3]}, rank5={best_satisfaction.cost_combination[4]}, tier2={best_satisfaction.cost_combination[5]}, tier3={best_satisfaction.cost_combination[6]}")
        print(f"   Satisfaction: {best_satisfaction.satisfaction_score:.3f}")
        print(f"   Fairness: {best_satisfaction.fairness_score:.3f}")
        
        # 3. Best Fairness (prioritize equality)
        best_fairness = max(self.pareto_frontier, 
                           key=lambda s: s.fairness_score)
        print(f"\n⚖️  Best Fairness (Most Equal):")
        print(f"   Cost: {best_fairness.cost_combination}")
        print(f"   → rank1={best_fairness.cost_combination[0]}, rank2={best_fairness.cost_combination[1]}, rank3={best_fairness.cost_combination[2]}, rank4={best_fairness.cost_combination[3]}, rank5={best_fairness.cost_combination[4]}, tier2={best_fairness.cost_combination[5]}, tier3={best_fairness.cost_combination[6]}")
        print(f"   Satisfaction: {best_fairness.satisfaction_score:.3f}")
        print(f"   Fairness: {best_fairness.fairness_score:.3f}")
        
        # 4. Most Balanced (closest to equal weight)
        best_balanced = min(self.pareto_frontier,
                           key=lambda s: abs(s.satisfaction_score - s.fairness_score))
        print(f"\n⚖️  Most Balanced:")
        print(f"   Cost: {best_balanced.cost_combination}")
        print(f"   → rank1={best_balanced.cost_combination[0]}, rank2={best_balanced.cost_combination[1]}, rank3={best_balanced.cost_combination[2]}, rank4={best_balanced.cost_combination[3]}, rank5={best_balanced.cost_combination[4]}, tier2={best_balanced.cost_combination[5]}, tier3={best_balanced.cost_combination[6]}")
        print(f"   Satisfaction: {best_balanced.satisfaction_score:.3f}")
        print(f"   Fairness: {best_balanced.fairness_score:.3f}")
        print(f"   Difference: {abs(best_balanced.satisfaction_score - best_balanced.fairness_score):.3f}")
        
        print("\n" + "=" * 60)
    
    def save_results(self):
        """Save results to files."""
        print("💾 Saving results...")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert results to DataFrame
        data = []
        for r in self.results:
            row = {
                'simulation_id': r.simulation_id,
                'rank1_cost': r.cost_combination[0],
                'rank2_cost': r.cost_combination[1],
                'rank3_cost': r.cost_combination[2],
                'rank4_cost': r.cost_combination[3],
                'rank5_cost': r.cost_combination[4],
                'tier2_cost': r.cost_combination[5],
                'tier3_cost': r.cost_combination[6],
                'satisfaction_score': r.satisfaction_score,
                'fairness_score': r.fairness_score,
                'gini_coefficient': r.gini_coefficient,
                'num_students': r.num_students,
                'is_pareto': r in self.pareto_frontier,
                **{f'pref_{k}': v for k, v in r.preference_satisfaction.items()}
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        csv_path = self.output_dir / 'grid_search_results_with_tiers.csv'
        df.to_csv(csv_path, index=False)
        print(f"   Saved: {csv_path}")
        
        # Save summary
        summary_path = self.output_dir / 'grid_search_summary_with_tiers.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Fast Grid Search with Tiers - Summary\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Total Simulations: {len(self.results):,}\n")
            f.write(f"Pareto Solutions: {len(self.pareto_frontier):,}\n\n")
            
            # Write recommended solutions at the top
            if self.pareto_frontier:
                f.write(f"🎯 RECOMMENDED SOLUTIONS\n")
                f.write(f"{'='*60}\n")
                
                # 1. Best Combined Score
                best_combined = max(self.pareto_frontier, 
                                   key=lambda s: s.satisfaction_score * s.fairness_score)
                f.write(f"\n📊 BEST COMBINED SCORE (Highest Satisfaction × Fairness)\n")
                f.write(f"   Cost: {best_combined.cost_combination}\n")
                f.write(f"   → rank1={best_combined.cost_combination[0]}, rank2={best_combined.cost_combination[1]}, rank3={best_combined.cost_combination[2]}, rank4={best_combined.cost_combination[3]}, rank5={best_combined.cost_combination[4]}, tier2={best_combined.cost_combination[5]}, tier3={best_combined.cost_combination[6]}\n")
                f.write(f"   Satisfaction: {best_combined.satisfaction_score:.3f}\n")
                f.write(f"   Fairness: {best_combined.fairness_score:.3f}\n")
                f.write(f"   Combined: {best_combined.satisfaction_score * best_combined.fairness_score:.4f}\n")
                f.write(f"   Gini: {best_combined.gini_coefficient:.3f}\n")
                
                # 2. Best Satisfaction
                best_satisfaction = max(self.pareto_frontier, 
                                      key=lambda s: s.satisfaction_score)
                f.write(f"\n😊 BEST SATISFACTION (Maximize Student Happiness)\n")
                f.write(f"   Cost: {best_satisfaction.cost_combination}\n")
                f.write(f"   → rank1={best_satisfaction.cost_combination[0]}, rank2={best_satisfaction.cost_combination[1]}, rank3={best_satisfaction.cost_combination[2]}, rank4={best_satisfaction.cost_combination[3]}, rank5={best_satisfaction.cost_combination[4]}, tier2={best_satisfaction.cost_combination[5]}, tier3={best_satisfaction.cost_combination[6]}\n")
                f.write(f"   Satisfaction: {best_satisfaction.satisfaction_score:.3f}\n")
                f.write(f"   Fairness: {best_satisfaction.fairness_score:.3f}\n")
                f.write(f"   Gini: {best_satisfaction.gini_coefficient:.3f}\n")
                
                # 3. Best Fairness
                best_fairness = max(self.pareto_frontier, 
                                   key=lambda s: s.fairness_score)
                f.write(f"\n⚖️  BEST FAIRNESS (Most Equal Distribution)\n")
                f.write(f"   Cost: {best_fairness.cost_combination}\n")
                f.write(f"   → rank1={best_fairness.cost_combination[0]}, rank2={best_fairness.cost_combination[1]}, rank3={best_fairness.cost_combination[2]}, rank4={best_fairness.cost_combination[3]}, rank5={best_fairness.cost_combination[4]}, tier2={best_fairness.cost_combination[5]}, tier3={best_fairness.cost_combination[6]}\n")
                f.write(f"   Satisfaction: {best_fairness.satisfaction_score:.3f}\n")
                f.write(f"   Fairness: {best_fairness.fairness_score:.3f}\n")
                f.write(f"   Gini: {best_fairness.gini_coefficient:.3f}\n")
                
                # 4. Most Balanced
                best_balanced = min(self.pareto_frontier,
                                   key=lambda s: abs(s.satisfaction_score - s.fairness_score))
                f.write(f"\n⚖️  MOST BALANCED (Smallest Gap Between Metrics)\n")
                f.write(f"   Cost: {best_balanced.cost_combination}\n")
                f.write(f"   → rank1={best_balanced.cost_combination[0]}, rank2={best_balanced.cost_combination[1]}, rank3={best_balanced.cost_combination[2]}, rank4={best_balanced.cost_combination[3]}, rank5={best_balanced.cost_combination[4]}, tier2={best_balanced.cost_combination[5]}, tier3={best_balanced.cost_combination[6]}\n")
                f.write(f"   Satisfaction: {best_balanced.satisfaction_score:.3f}\n")
                f.write(f"   Fairness: {best_balanced.fairness_score:.3f}\n")
                f.write(f"   Difference: {abs(best_balanced.satisfaction_score - best_balanced.fairness_score):.3f}\n")
                f.write(f"   Gini: {best_balanced.gini_coefficient:.3f}\n")
                
                f.write(f"\n{'='*60}\n")
                f.write(f"\nALL PARETO-OPTIMAL SOLUTIONS ({len(self.pareto_frontier):,})\n")
                f.write(f"{'='*60}\n")
                for i, sol in enumerate(self.pareto_frontier, 1):
                    f.write(f"\n{i}. Cost: {sol.cost_combination}\n")
                    f.write(f"   Satisfaction: {sol.satisfaction_score:.3f}\n")
                    f.write(f"   Fairness: {sol.fairness_score:.3f}\n")
                    f.write(f"   Gini: {sol.gini_coefficient:.3f}\n")
        
        print(f"   Saved: {summary_path}")
        
        # Create visualizations
        self.create_visualizations(df)
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create visualizations of results."""
        print("📈 Creating visualizations...")
        
        if df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Fast Grid Search with Tiers - Results', fontsize=16, fontweight='bold')
        
        # Pareto frontier
        ax = axes[0, 0]
        pareto_df = df[df['is_pareto']]
        
        # Debug: Check for duplicate coordinates
        print(f"   📊 Pareto DataFrame size: {len(pareto_df)} rows")
        unique_coords = pareto_df[['satisfaction_score', 'fairness_score']].drop_duplicates()
        print(f"   📊 Unique coordinates in Pareto: {len(unique_coords)}")
        
        ax.scatter(df['satisfaction_score'], df['fairness_score'], 
                   c='lightgray', alpha=0.5, s=20, label='All solutions')
        ax.scatter(pareto_df['satisfaction_score'], pareto_df['fairness_score'],
                   c='red', s=50, marker='*', label=f'Pareto frontier ({len(pareto_df)} solutions, {len(unique_coords)} unique points)')
        ax.set_xlabel('Satisfaction Score')
        ax.set_ylabel('Fairness Score')
        ax.set_title('Pareto Frontier')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cost vs Satisfaction
        ax = axes[0, 1]
        ax.scatter(df['satisfaction_score'], df['rank1_cost'], c=df['tier2_cost'], cmap='viridis', alpha=0.5)
        ax.set_xlabel('Satisfaction Score')
        ax.set_ylabel('Rank1 Cost')
        ax.set_title('Rank1 Cost vs Satisfaction')
        ax.grid(True, alpha=0.3)
        
        # Tier costs distribution
        ax = axes[1, 0]
        ax.hist2d(df['tier2_cost'], df['tier3_cost'], bins=20, cmap='Blues')
        ax.set_xlabel('Tier2 Cost')
        ax.set_ylabel('Tier3 Cost')
        ax.set_title('Tier Costs Distribution')
        
        # Satisfaction distribution
        ax = axes[1, 1]
        ax.hist(df['satisfaction_score'], bins=30, edgecolor='black')
        ax.set_xlabel('Satisfaction Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Satisfaction Score Distribution')
        ax.grid(True, alpha=0.3)
        
        # Save figure
        fig_path = self.output_dir / 'grid_search_with_tiers_analysis.png'
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {fig_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Fast grid search with tier support')
    parser.add_argument('--students', type=str, default='data/input/students.csv',
                       help='Students CSV file')
    parser.add_argument('--capacities', type=str, default='data/input/capacities.csv',
                       help='Capacities CSV file')
    parser.add_argument('--output', type=str, default='test_grid_search/fast_with_tiers',
                       help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    # Create grid search instance
    grid_search = ComprehensiveGridSearchWithTiers(output_dir)
    
    # Generate combinations
    combinations = grid_search.generate_grid_combinations()
    
    # Run grid search
    results = grid_search.run_grid_search(
        combinations, 
        args.students, 
        args.capacities
    )
    
    # Find Pareto frontier
    grid_search.find_pareto_frontier()
    
    # Save results
    grid_search.save_results()
    
    print(f"\n🎉 Complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()

