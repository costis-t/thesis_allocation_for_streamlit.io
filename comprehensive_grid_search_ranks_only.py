#!/usr/bin/env python3
"""
Fast Grid Search - Rank Costs Only

This script implements a grid search that only considers rank costs (rank1-5),
without tier costs. Generates approximately 12,000-14,000 combinations.
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
        
        # Create preference model configuration (ONLY rank costs)
        pref_cfg = PreferenceModelConfig(
            rank1_cost=cost_combo[0],
            rank2_cost=cost_combo[1],
            rank3_cost=cost_combo[2],
            rank4_cost=cost_combo[3],
            rank5_cost=cost_combo[4],
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
            'status': status
        }
    except Exception as e:
        if i < 3:
            print(f"\n  ❌ Exception in allocation {i}: {str(e)[:100]}")
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
    
    # Count students per coach
    coach_counts = Counter()
    for student_id, topic_id in allocation.items():
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
    """Container for grid search results - ranks only."""
    cost_combination: Tuple[int, int, int, int, int]  # rank1-5 only
    satisfaction_score: float
    fairness_score: float
    total_cost: float
    preference_satisfaction: Dict[str, int]
    gini_coefficient: float
    num_students: int
    algorithm: str
    timestamp: str
    simulation_id: int

class ComprehensiveGridSearchRanksOnly:
    """Implements grid search with rank costs only."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.results = []
        self.pareto_frontier = []
        self.simulation_count = 0
        self.start_time = time.time()
        
    def generate_grid_combinations(self, granularity: int = 1) -> List[Tuple[int, int, int, int, int]]:
        """Generate systematic grid of rank cost combinations.
        
        Args:
            granularity: How detailed the search (1=fast ~12K, higher = more combinations)
        """
        print(f"🚀 Generating grid combinations (rank costs only, granularity={granularity})...")
        
        # Define base ranges
        if granularity == 1:
            # Fast (~12,000 combinations, ~5-6 min)
            rank1_range = [0, 5, 12, 25, 40, 60, 80]
            rank2_range = [0, 5, 12, 25, 40, 60, 80]
            rank3_range = [0, 25, 60, 100, 150, 200]
            rank4_range = [0, 25, 60, 100, 150, 200]
            rank5_range = [0, 25, 60, 100, 150, 200]
        elif granularity == 2:
            # Medium (~30,000 combinations, ~12-15 min)
            rank1_range = [0, 5, 10, 12, 20, 25, 35, 40, 50, 60, 80]
            rank2_range = [0, 5, 10, 12, 20, 25, 35, 40, 50, 60, 80]
            rank3_range = [0, 25, 40, 60, 80, 100, 125, 150, 175, 200]
            rank4_range = [0, 25, 40, 60, 80, 100, 125, 150, 175, 200]
            rank5_range = [0, 25, 40, 60, 80, 100, 125, 150, 175, 200]
        elif granularity == 3:
            # Exhaustive (~50,000+ combinations, ~20-25 min)
            rank1_range = [0, 5, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80]
            rank2_range = [0, 5, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80]
            rank3_range = [0, 25, 40, 50, 60, 75, 80, 100, 120, 125, 140, 150, 175, 200]
            rank4_range = [0, 25, 40, 50, 60, 75, 80, 100, 120, 125, 140, 150, 175, 200]
            rank5_range = [0, 25, 40, 50, 60, 75, 80, 100, 120, 125, 140, 150, 175, 200]
        elif granularity <= 6:
            # Generate fine-grained ranges for medium values
            rank1_range = list(range(0, 81, 5))  # 0, 5, 10, ..., 75, 80 (17 values)
            rank2_range = list(range(0, 81, 5))  # Same
            rank3_range = list(range(0, 201, 10))  # 0, 10, 20, ..., 190, 200 (21 values)
            rank4_range = list(range(0, 201, 10))  # Same
            rank5_range = list(range(0, 201, 10))  # Same
        elif granularity <= 8:
            # Even finer for ranks 1-2
            rank1_range = list(range(0, 81, 4))  # 0, 4, 8, ..., 76, 80 (21 values)
            rank2_range = list(range(0, 81, 4))  # Same
            rank3_range = list(range(0, 201, 8))  # 0, 8, 16, ..., 192, 200 (26 values)
            rank4_range = list(range(0, 201, 8))  # Same
            rank5_range = list(range(0, 201, 8))  # Same
        else:  # granularity 9-10
            # Very fine-grained
            rank1_range = list(range(0, 81, 3))  # 0, 3, 6, ..., 78, 81 → cap at 80 (28 values)
            rank2_range = list(range(0, 81, 3))  # Same
            rank3_range = list(range(0, 201, 5))  # 0, 5, 10, ..., 195, 200 (41 values)
            rank4_range = list(range(0, 201, 5))  # Same
            rank5_range = list(range(0, 201, 5))  # Same
            
            # Cap rank1-2 at 80
            rank1_range = [x for x in rank1_range if x <= 80]
            rank2_range = [x for x in rank2_range if x <= 80]
        
        # Generate all combinations
        all_combinations = list(itertools.product(
            rank1_range, rank2_range, rank3_range, rank4_range, rank5_range
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
        
        return combinations
    
    def run_grid_search(self, combinations: List[Tuple[int, int, int, int, int]], 
                       students_path: str, capacities_path: str, num_cores: int = 14, 
                       granularity: int = 1) -> List[GridSearchResult]:
        """Run grid search with parallel execution."""
        
        total_combos = len(combinations)
        self.results = []
        
        print(f"🚀 Starting grid search with {total_combos:,} combinations...")
        print(f"   Using {num_cores} CPU cores")
        print(f"   Granularity: {granularity}")
        estimated_time = {
            1: "~5-6 minutes",
            2: "~12-15 minutes",
            3: "~20-25 minutes",
            4: "~45-60 minutes",
            5: "~45-60 minutes",
            6: "~45-60 minutes",
            7: "~60-75 minutes",
            8: "~60-75 minutes",
            9: "~2-3 hours",
            10: "~2-3 hours"
        }.get(granularity, "~2-3+ hours")
        print(f"   Estimated time: {estimated_time}")
        print(f"   Students: {Path(students_path).name}")
        print(f"   Capacities: {Path(capacities_path).name}")
        print()
        
        # Prepare arguments for multiprocessing
        args_list = [
            (i, combo, students_path, capacities_path)
            for i, combo in enumerate(combinations)
        ]
        
        # Run in parallel
        success_count = 0
        fail_count = 0
        
        with mp.Pool(num_cores) as pool:
            with tqdm(total=total_combos, desc="Grid Search") as pbar:
                for result_dict in pool.imap(run_single_allocation, args_list):
                    if result_dict.get('success'):
                        success_count += 1
                        self.results.append(result_dict)  # Collect successful results
                    else:
                        fail_count += 1
                    
                    pbar.update(1)
                    pbar.set_postfix(success=success_count, fail=fail_count)
        
        print(f"\n✅ Successful: {success_count}, Failed: {fail_count}")
        return self.results
    
    def _recommend_solutions(self):
        """Recommend best solutions from Pareto frontier."""
        if not self.pareto_frontier:
            print("   No Pareto-optimal solutions found")
            return
        
        # 1. Best Combined Score (Satisfaction × Fairness)
        best_combined = max(self.pareto_frontier, 
                           key=lambda s: s.satisfaction_score * s.fairness_score)
        print(f"\n📊 BEST COMBINED SCORE")
        print(f"   Cost: {best_combined.cost_combination}")
        print(f"   → rank1={best_combined.cost_combination[0]}, rank2={best_combined.cost_combination[1]}, rank3={best_combined.cost_combination[2]}, rank4={best_combined.cost_combination[3]}, rank5={best_combined.cost_combination[4]}")
        print(f"   Satisfaction: {best_combined.satisfaction_score:.3f}")
        print(f"   Fairness: {best_combined.fairness_score:.3f}")
        
        # 2. Best Satisfaction
        best_satisfaction = max(self.pareto_frontier, key=lambda s: s.satisfaction_score)
        print(f"\n😊 BEST SATISFACTION")
        print(f"   Cost: {best_satisfaction.cost_combination}")
        print(f"   → rank1={best_satisfaction.cost_combination[0]}, rank2={best_satisfaction.cost_combination[1]}, rank3={best_satisfaction.cost_combination[2]}, rank4={best_satisfaction.cost_combination[3]}, rank5={best_satisfaction.cost_combination[4]}")
        print(f"   Satisfaction: {best_satisfaction.satisfaction_score:.3f}")
        
        # 3. Best Fairness
        best_fairness = max(self.pareto_frontier, key=lambda s: s.fairness_score)
        print(f"\n⚖️  BEST FAIRNESS")
        print(f"   Cost: {best_fairness.cost_combination}")
        print(f"   → rank1={best_fairness.cost_combination[0]}, rank2={best_fairness.cost_combination[1]}, rank3={best_fairness.cost_combination[2]}, rank4={best_fairness.cost_combination[3]}, rank5={best_fairness.cost_combination[4]}")
        print(f"   Fairness: {best_fairness.fairness_score:.3f}")
        
        # 4. Most Balanced
        best_balanced = min(self.pareto_frontier,
                           key=lambda s: abs(s.satisfaction_score - s.fairness_score))
        print(f"\n⚖️  MOST BALANCED")
        print(f"   Cost: {best_balanced.cost_combination}")
        print(f"   → rank1={best_balanced.cost_combination[0]}, rank2={best_balanced.cost_combination[1]}, rank3={best_balanced.cost_combination[2]}, rank4={best_balanced.cost_combination[3]}, rank5={best_balanced.cost_combination[4]}")
        print(f"   Satisfaction: {best_balanced.satisfaction_score:.3f}")
        print(f"   Fairness: {best_balanced.fairness_score:.3f}")
    
    def _identify_pareto_frontier(self, results: List[Dict]) -> List[GridSearchResult]:
        """Identify Pareto-optimal solutions."""
        # Convert to GridSearchResult objects
        grid_results = []
        for r in results:
            if r.get('success'):
                grid_results.append(GridSearchResult(
                    cost_combination=r['cost_combo'],
                    satisfaction_score=r['satisfaction_score'],
                    fairness_score=1 - r['gini_coefficient'],  # Convert Gini to fairness
                    total_cost=0,
                    preference_satisfaction=r['preference_satisfaction'],
                    gini_coefficient=r['gini_coefficient'],
                    num_students=r['num_students'],
                    algorithm='ilp',
                    timestamp=r['timestamp'],
                    simulation_id=r['simulation_id']
                ))
        
        # Find Pareto-optimal solutions
        pareto_frontier = []
        for solution in grid_results:
            is_dominated = False
            for other in grid_results:
                if (other.satisfaction_score > solution.satisfaction_score and 
                    other.fairness_score > solution.fairness_score):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_frontier.append(solution)
        
        return pareto_frontier
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create visualizations of results."""
        print("📈 Creating visualizations...")
        
        if df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Grid Search - Rank Costs Only - Results', fontsize=16, fontweight='bold')
        
        # Pareto frontier
        ax = axes[0, 0]
        pareto_df = df[df['is_pareto']]
        
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
        
        # Rank1 cost vs Satisfaction
        ax = axes[0, 1]
        ax.scatter(df['satisfaction_score'], df['rank1_cost'], alpha=0.5)
        ax.set_xlabel('Satisfaction Score')
        ax.set_ylabel('Rank1 Cost')
        ax.set_title('Rank1 Cost vs Satisfaction')
        ax.grid(True, alpha=0.3)
        
        # Rank4 vs Rank5 cost
        ax = axes[1, 0]
        ax.scatter(df['rank4_cost'], df['rank5_cost'], alpha=0.5)
        ax.set_xlabel('Rank4 Cost')
        ax.set_ylabel('Rank5 Cost')
        ax.set_title('Rank4 vs Rank5 Cost')
        ax.grid(True, alpha=0.3)
        
        # Satisfaction distribution
        ax = axes[1, 1]
        ax.hist(df['satisfaction_score'], bins=30, edgecolor='black')
        ax.set_xlabel('Satisfaction Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Satisfaction Score Distribution')
        ax.grid(True, alpha=0.3)
        
        # Save figure
        fig_path = self.output_dir / 'grid_search_ranks_only_analysis.png'
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: {fig_path}")
        plt.close()

    def save_results(self, results: List[Dict]):
        """Save results to CSV and summary text file."""
        self.pareto_frontier = self._identify_pareto_frontier(results)
        
        # Convert results to DataFrame with proper structure
        data = []
        for r in results:
            if r.get('success'):
                row = {
                    'simulation_id': r['simulation_id'],
                    'rank1_cost': r['cost_combo'][0],
                    'rank2_cost': r['cost_combo'][1],
                    'rank3_cost': r['cost_combo'][2],
                    'rank4_cost': r['cost_combo'][3],
                    'rank5_cost': r['cost_combo'][4],
                    'satisfaction_score': r['satisfaction_score'],
                    'fairness_score': 1 - r['gini_coefficient'],
                    'gini_coefficient': r['gini_coefficient'],
                    'num_students': r['num_students'],
                    'is_pareto': False,  # Will be set below
                    **{f'pref_{k}': v for k, v in r['preference_satisfaction'].items()}
                }
                data.append(row)
        
        df = pd.DataFrame(data)
        
        # Mark Pareto solutions
        if self.pareto_frontier:
            pareto_indices = {sol.simulation_id for sol in self.pareto_frontier}
            df['is_pareto'] = df['simulation_id'].isin(pareto_indices)
        
        # Save CSV
        csv_path = self.output_dir / "grid_search_results_ranks_only.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved results to {csv_path}")
        
        # Create visualizations
        self.create_visualizations(df)
        
        # Save summary
        txt_path = self.output_dir / "grid_search_summary_ranks_only.txt"
        with open(txt_path, 'w') as f:
            f.write(f"Grid Search Results - Rank Costs Only\n")
            f.write(f"{'='*60}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total combinations: {len(results):,}\n")
            f.write(f"Successful: {sum(1 for r in results if r.get('success')):,}\n")
            f.write(f"Failed: {sum(1 for r in results if not r.get('success')):,}\n")
            f.write(f"Pareto-optimal solutions: {len(self.pareto_frontier):,}\n\n")
            
            if self.pareto_frontier:
                # Write recommended solutions at the top
                f.write(f"🎯 RECOMMENDED SOLUTIONS\n")
                f.write(f"{'='*60}\n")
                
                # 1. Best Combined Score
                best_combined = max(self.pareto_frontier, key=lambda s: s.satisfaction_score * s.fairness_score)
                f.write(f"\n📊 BEST COMBINED SCORE\n")
                f.write(f"   Cost: {best_combined.cost_combination}\n")
                f.write(f"   → rank1={best_combined.cost_combination[0]}, rank2={best_combined.cost_combination[1]}, rank3={best_combined.cost_combination[2]}, rank4={best_combined.cost_combination[3]}, rank5={best_combined.cost_combination[4]}\n")
                f.write(f"   Satisfaction: {best_combined.satisfaction_score:.3f}\n")
                f.write(f"   Fairness: {best_combined.fairness_score:.3f}\n\n")
                
                # 2. Best Satisfaction
                best_satisfaction = max(self.pareto_frontier, key=lambda s: s.satisfaction_score)
                f.write(f"😊 BEST SATISFACTION\n")
                f.write(f"   Cost: {best_satisfaction.cost_combination}\n")
                f.write(f"   → rank1={best_satisfaction.cost_combination[0]}, rank2={best_satisfaction.cost_combination[1]}, rank3={best_satisfaction.cost_combination[2]}, rank4={best_satisfaction.cost_combination[3]}, rank5={best_satisfaction.cost_combination[4]}\n")
                f.write(f"   Satisfaction: {best_satisfaction.satisfaction_score:.3f}\n\n")
                
                # 3. Best Fairness
                best_fairness = max(self.pareto_frontier, key=lambda s: s.fairness_score)
                f.write(f"⚖️  BEST FAIRNESS\n")
                f.write(f"   Cost: {best_fairness.cost_combination}\n")
                f.write(f"   → rank1={best_fairness.cost_combination[0]}, rank2={best_fairness.cost_combination[1]}, rank3={best_fairness.cost_combination[2]}, rank4={best_fairness.cost_combination[3]}, rank5={best_fairness.cost_combination[4]}\n")
                f.write(f"   Fairness: {best_fairness.fairness_score:.3f}\n\n")
                
                # 4. Most Balanced
                best_balanced = min(self.pareto_frontier, key=lambda s: abs(s.satisfaction_score - s.fairness_score))
                f.write(f"⚖️  MOST BALANCED\n")
                f.write(f"   Cost: {best_balanced.cost_combination}\n")
                f.write(f"   → rank1={best_balanced.cost_combination[0]}, rank2={best_balanced.cost_combination[1]}, rank3={best_balanced.cost_combination[2]}, rank4={best_balanced.cost_combination[3]}, rank5={best_balanced.cost_combination[4]}\n")
                f.write(f"   Satisfaction: {best_balanced.satisfaction_score:.3f}\n")
                f.write(f"   Fairness: {best_balanced.fairness_score:.3f}\n\n")
                
                # Write ALL Pareto-optimal solutions
                f.write(f"\n{'='*60}\n")
                f.write(f"ALL PARETO-OPTIMAL SOLUTIONS ({len(self.pareto_frontier):,})\n")
                f.write(f"{'='*60}\n\n")
                
                for i, sol in enumerate(sorted(self.pareto_frontier, key=lambda s: (s.satisfaction_score, s.fairness_score)), 1):
                    f.write(f"{i}. Cost: {sol.cost_combination}\n")
                    f.write(f"   → rank1={sol.cost_combination[0]}, rank2={sol.cost_combination[1]}, rank3={sol.cost_combination[2]}, rank4={sol.cost_combination[3]}, rank5={sol.cost_combination[4]}\n")
                    f.write(f"   Satisfaction: {sol.satisfaction_score:.3f}\n")
                    f.write(f"   Fairness: {sol.fairness_score:.3f}\n")
                    f.write(f"   Gini: {sol.gini_coefficient:.3f}\n\n")
        
        print(f"✅ Saved summary to {txt_path}")
        self._recommend_solutions()


def main():
    parser = argparse.ArgumentParser(description='Grid search for rank costs only')
    parser.add_argument('--students', type=str, required=True, help='Path to students CSV')
    parser.add_argument('--capacities', type=str, required=True, help='Path to capacities CSV')
    parser.add_argument('--output', type=str, default='grid_search_ranks_only', help='Output directory')
    parser.add_argument('--cores', type=int, default=14, help='Number of CPU cores')
    parser.add_argument('--granularity', type=int, default=1, 
                       help='Search granularity: 1=fast (~12K), 2=med (~30K), 3-6=detailed (~100K+), 7-10=very detailed (200K+)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize grid search
    grid_search = ComprehensiveGridSearchRanksOnly(output_dir)
    
    # Generate combinations
    combinations = grid_search.generate_grid_combinations(granularity=args.granularity)
    
    # Run grid search
    results = grid_search.run_grid_search(
        combinations, args.students, args.capacities, args.cores, args.granularity
    )
    
    # Save results
    grid_search.save_results(results)
    
    print("\n✅ Grid search complete!")


if __name__ == '__main__':
    main()
