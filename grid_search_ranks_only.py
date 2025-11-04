#!/usr/bin/env python3
"""
Grid Search - Rank Costs Only (No Tiers)

This script performs a grid search over rank costs (rank1-rank5) only,
without tier cost parameters. Use this for simpler parameter optimization.

Usage:
    ./grid_search_ranks_only.py
    
    # Custom files
    ./grid_search_ranks_only.py --students data/input/students.csv --capacities data/input/capacities.csv
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
            self.n = 0
            self.total = len(iterable) if hasattr(iterable, '__len__') else None
            self.start_time = time.time()
        def __iter__(self):
            return self
        def __next__(self):
            try:
                item = next(iter(self.iterable))
                self.n += 1
                if self.n % 50 == 0:
                    elapsed = time.time() - self.start_time
                    rate = self.n / elapsed if elapsed > 0 else 0
                    remaining = (self.total - self.n) / rate if rate > 0 and self.total else 0
                    percent = (self.n / self.total * 100) if self.total else 0
                    print(f"\r{self.desc}: {self.n}/{self.total} ({percent:.1f}%) | ETA: {remaining:.1f}s", end="", flush=True)
                return item
            except StopIteration:
                print(f"\r{self.desc}: Complete! Processed {self.n} items")
                raise
        def set_postfix(self, **kwargs):
            pass
        def close(self):
            pass

# Import modules
import sys
sys.path.append('.')
from allocator.data_repository import DataRepository
from allocator.allocation_model_ilp import AllocationConfig as LegacyAllocationConfig
from allocator.preference_model import PreferenceModelConfig, PreferenceModel
from allocator.allocation_model_ilp import AllocationModelILP

# Worker function
def run_single_allocation(args):
    """Run a single allocation - returns dict with result."""
    i, cost_combo, students_path, capacities_path = args
    
    try:
        repo = DataRepository(students_path, capacities_path)
        repo.load()
        
        pref_cfg = PreferenceModelConfig(
            rank1_cost=cost_combo[0],
            rank2_cost=cost_combo[1],
            rank3_cost=cost_combo[2],
            rank4_cost=cost_combo[3],
            rank5_cost=cost_combo[4],
            top2_bias=False,
            unranked_cost=200
            # No tier costs - uses defaults (tier1=0, tier2=1, tier3=5)
        )
        
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
        pref_satisfaction = _calculate_pref_satisfaction(repo, allocation)
        gini = _calculate_gini_for_allocation(allocation, repo)
        satisfaction_score = _calculate_satisfaction_score(pref_satisfaction)
        
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
    
    coach_counts = Counter()
    for student_id, topic_id in allocation.items():
        topic = repo.topics.get(topic_id)
        if topic:
            coach_id = topic.coach_id
            coach_counts[coach_id] += 1
    
    if not coach_counts or len(coach_counts) < 2:
        return 0.0
    
    counts = sorted(coach_counts.values())
    n = len(counts)
    cumsum = sum(counts)
    
    if cumsum == 0:
        return 0.0
    
    gini = (2 * sum((i + 1) * count for i, count in enumerate(counts))) / (n * cumsum) - (n + 1) / n
    return abs(gini)


@dataclass
class GridSearchResult:
    """Container for grid search results - rank costs only."""
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


class GridSearchRanksOnly:
    """Grid search for rank costs only (no tiers)."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.results = []
        self.pareto_frontier = []
        self.simulation_count = 0
        self.start_time = time.time()
    
    def generate_grid_combinations(self) -> List[Tuple[int, int, int, int, int]]:
        """Generate grid of rank cost combinations only."""
        print("🔧 Generating grid combinations (rank costs only, ~14K combinations, ~4-5 min runtime)...")
        
        # 12 values each to get ~14,000 combinations after monotonic filtering
        rank1_range = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 50]    # 12 values
        rank2_range = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 50]    # 12 values
        rank3_range = [0, 20, 35, 50, 65, 80, 100, 115, 130, 145, 160, 200]  # 12 values
        rank4_range = [0, 20, 35, 50, 65, 80, 100, 115, 130, 145, 160, 200]  # 12 values
        rank5_range = [0, 20, 35, 50, 65, 80, 100, 115, 130, 145, 160, 200]  # 12 values
        
        all_combinations = list(itertools.product(
            rank1_range, rank2_range, rank3_range, rank4_range, rank5_range
        ))
        
        # Filter to ensure monotonic costs
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
                       students_path: str, capacities_path: str, num_cores: int = 14) -> List[GridSearchResult]:
        """Run grid search using multiprocessing."""
        print(f"🚀 Starting parallel grid search with {len(combinations):,} combinations using {num_cores} cores...")
        
        args = [(i, combo, students_path, capacities_path) for i, combo in enumerate(combinations)]
        successful_simulations = 0
        failed_simulations = 0
        
        print(f"   Processing {len(args):,} allocations across {num_cores} cores...")
        
        with mp.Pool(processes=num_cores) as pool:
            results_iter = pool.imap(run_single_allocation, args)
            results_list = []
            
            for result in tqdm(results_iter, total=len(args), desc="Grid Search", unit=" combos", ncols=100):
                if result.get('success', False):
                    successful_simulations += 1
                    grid_result = GridSearchResult(
                        cost_combination=result['cost_combo'],
                        satisfaction_score=result.get('satisfaction_score', 0.0),
                        fairness_score=1.0 - result.get('gini_coefficient', 0.0),
                        total_cost=0.0,
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
        self._recommend_solutions()
        
        return pareto_solutions
    
    def _recommend_solutions(self):
        """Recommend best Pareto-optimal solutions."""
        if not self.pareto_frontier:
            return
        
        print("\n🎯 Recommended Solutions:")
        print("=" * 60)
        
        best_combined = max(self.pareto_frontier, key=lambda s: s.satisfaction_score * s.fairness_score)
        print(f"\n📊 Best Combined Score:")
        print(f"   Cost: {best_combined.cost_combination}")
        print(f"   Satisfaction: {best_combined.satisfaction_score:.3f}")
        print(f"   Fairness: {best_combined.fairness_score:.3f}")
        
        best_satisfaction = max(self.pareto_frontier, key=lambda s: s.satisfaction_score)
        print(f"\n😊 Best Satisfaction:")
        print(f"   Cost: {best_satisfaction.cost_combination}")
        print(f"   Satisfaction: {best_satisfaction.satisfaction_score:.3f}")
        print(f"   Fairness: {best_satisfaction.fairness_score:.3f}")
        
        best_fairness = max(self.pareto_frontier, key=lambda s: s.fairness_score)
        print(f"\n⚖️  Best Fairness:")
        print(f"   Cost: {best_fairness.cost_combination}")
        print(f"   Satisfaction: {best_fairness.satisfaction_score:.3f}")
        print(f"   Fairness: {best_fairness.fairness_score:.3f}")
        
        print("\n" + "=" * 60)
    
    def save_results(self):
        """Save results to files."""
        print("💾 Saving results...")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        data = []
        for r in self.results:
            row = {
                'simulation_id': r.simulation_id,
                'rank1_cost': r.cost_combination[0],
                'rank2_cost': r.cost_combination[1],
                'rank3_cost': r.cost_combination[2],
                'rank4_cost': r.cost_combination[3],
                'rank5_cost': r.cost_combination[4],
                'satisfaction_score': r.satisfaction_score,
                'fairness_score': r.fairness_score,
                'gini_coefficient': r.gini_coefficient,
                'num_students': r.num_students,
                'is_pareto': r in self.pareto_frontier,
                **{f'pref_{k}': v for k, v in r.preference_satisfaction.items()}
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        csv_path = self.output_dir / 'grid_search_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"   Saved: {csv_path}")
        
        summary_path = self.output_dir / 'grid_search_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Grid Search - Rank Costs Only - Summary\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Total Simulations: {len(self.results):,}\n")
            f.write(f"Pareto Solutions: {len(self.pareto_frontier):,}\n\n")
            
            if self.pareto_frontier:
                best_combined = max(self.pareto_frontier, key=lambda s: s.satisfaction_score * s.fairness_score)
                best_satisfaction = max(self.pareto_frontier, key=lambda s: s.satisfaction_score)
                best_fairness = max(self.pareto_frontier, key=lambda s: s.fairness_score)
                best_balanced = min(self.pareto_frontier, key=lambda s: abs(s.satisfaction_score - s.fairness_score))
                
                f.write(f"🎯 RECOMMENDED SOLUTIONS\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"📊 BEST COMBINED SCORE\n")
                f.write(f"   Cost: {best_combined.cost_combination}\n")
                f.write(f"   Satisfaction: {best_combined.satisfaction_score:.3f}\n")
                f.write(f"   Fairness: {best_combined.fairness_score:.3f}\n\n")
                
                f.write(f"😊 BEST SATISFACTION\n")
                f.write(f"   Cost: {best_satisfaction.cost_combination}\n")
                f.write(f"   Satisfaction: {best_satisfaction.satisfaction_score:.3f}\n")
                f.write(f"   Fairness: {best_satisfaction.fairness_score:.3f}\n\n")
                
                f.write(f"⚖️  BEST FAIRNESS\n")
                f.write(f"   Cost: {best_fairness.cost_combination}\n")
                f.write(f"   Satisfaction: {best_fairness.satisfaction_score:.3f}\n")
                f.write(f"   Fairness: {best_fairness.fairness_score:.3f}\n\n")
        
        print(f"   Saved: {summary_path}")
        print(f"\n✅ Complete! Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Grid search for rank costs only')
    parser.add_argument('--students', type=str, default='data/input/students.csv',
                       help='Students CSV file')
    parser.add_argument('--capacities', type=str, default='data/input/capacities.csv',
                       help='Capacities CSV file')
    parser.add_argument('--output', type=str, default='test_grid_search/ranks_only',
                       help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    grid_search = GridSearchRanksOnly(output_dir)
    
    combinations = grid_search.generate_grid_combinations()
    print(f"\n⏱️  Estimated runtime: {len(combinations) * 0.02 / 60:.1f} minutes")
    
    results = grid_search.run_grid_search(combinations, args.students, args.capacities)
    grid_search.find_pareto_frontier()
    grid_search.save_results()


if __name__ == '__main__':
    main()

