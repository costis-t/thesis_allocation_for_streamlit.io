#!/usr/bin/env python3
"""
Fine-Grained Grid Search with Tier Support

This script performs a more granular search either:
1. Full fine-grained search (smaller steps, more combinations)
2. Focused search around promising regions from a previous grid search

Usage:
    # Full fine-grained search
    ./fine_grid_search_with_tiers.py
    
    # Focused search around best solutions
    ./fine_grid_search_with_tiers.py --focus-results previous_results.csv
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
                if self.n % 50 == 0:
                    elapsed = time.time() - self.start_time
                    rate = self.n / elapsed if elapsed > 0 else 0
                    remaining = (self.total - self.n) / rate if rate > 0 and self.total else 0
                    percent = (self.n / self.total * 100) if self.total else 0
                    print(f"\r{self.desc}: {self.n}/{self.total} ({percent:.1f}%) | ETA: {remaining:.1f}s", end="", flush=True)
                return item
            except StopIteration:
                print(f"\r{self.desc}: Complete! Processed {self.n} items in {time.time() - self.start_time:.1f}s")
                raise
        def set_postfix(self, **kwargs):
            pass
        def close(self):
            pass

# Import our existing modules
import sys
sys.path.append('.')
from allocator.data_repository import DataRepository
from allocator.allocation_model_ilp import AllocationConfig as LegacyAllocationConfig
from allocator.preference_model import PreferenceModelConfig, PreferenceModel
from allocator.allocation_model_ilp import AllocationModelILP

# Worker function for multiprocessing (reuse from comprehensive)
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


def _calculate_satisfaction_score_preliminary(result_dict):
    """Calculate satisfaction score from result dict."""
    pref_satisfaction = result_dict.get('preference_satisfaction', {})
    return _calculate_satisfaction_score(pref_satisfaction)


def _calculate_gini_for_allocation(allocation, repo):
    """Calculate Gini coefficient for coach load balance."""
    from collections import Counter
    import numpy as np
    
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
    """Container for grid search results with tier support."""
    cost_combination: Tuple[int, int, int, int, int, int, int]
    satisfaction_score: float
    fairness_score: float
    total_cost: float
    preference_satisfaction: Dict[str, int]
    gini_coefficient: float
    num_students: int
    algorithm: str
    timestamp: str
    simulation_id: int


class FineGridSearchWithTiers:
    """Implements fine-grained grid search with tier cost support."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.results = []
        self.pareto_frontier = []
        self.simulation_count = 0
        self.start_time = time.time()
    
    def generate_grid_combinations(self, focus_file: str = None, 
                                  granularity: int = 3, top_n: int = 20) -> List[Tuple[int, int, int, int, int, int, int]]:
        """
        Generate fine-grained grid combinations.
        
        If focus_file is provided, creates a focused search around the top N solutions.
        Otherwise, does a full fine-grained search.
        
        Args:
            focus_file: CSV file with previous results (optional)
            granularity: Number of steps for focused search (default: 3)
            top_n: Number of top solutions to analyze (default: 20)
        """
        if focus_file:
            print(f"🔍 Loading focus results from: {focus_file}")
            focus_df = pd.read_csv(focus_file)
            
            # Get top N solutions
            focus_df['combined_score'] = focus_df['satisfaction_score'] * focus_df['fairness_score']
            top_solutions = focus_df.nlargest(top_n, 'combined_score')
            
            print(f"   📊 Analyzing {len(top_solutions)} top solutions...")
            print(f"   🎯 Using granularity={granularity} steps per parameter")
            
            # Extract cost ranges from these solutions
            rank1_range = self._create_fine_range(top_solutions['rank1_cost'].min(), top_solutions['rank1_cost'].max(), granularity)
            rank2_range = self._create_fine_range(top_solutions['rank2_cost'].min(), top_solutions['rank2_cost'].max(), granularity)
            rank3_range = self._create_fine_range(top_solutions['rank3_cost'].min(), top_solutions['rank3_cost'].max(), granularity)
            rank4_range = self._create_fine_range(top_solutions['rank4_cost'].min(), top_solutions['rank4_cost'].max(), granularity)
            rank5_range = self._create_fine_range(top_solutions['rank5_cost'].min(), top_solutions['rank5_cost'].max(), granularity)
            tier2_range = self._create_fine_range(top_solutions['tier2_cost'].min(), top_solutions['tier2_cost'].max(), granularity)
            tier3_range = self._create_fine_range(top_solutions['tier3_cost'].min(), top_solutions['tier3_cost'].max(), granularity)
            
            # Debug: Show the ranges being used
            print(f"\n   📊 Parameter ranges from top {len(top_solutions)} solutions:")
            print(f"      Rank1: {top_solutions['rank1_cost'].min()}-{top_solutions['rank1_cost'].max()} → {len(rank1_range)} values")
            print(f"      Rank2: {top_solutions['rank2_cost'].min()}-{top_solutions['rank2_cost'].max()} → {len(rank2_range)} values")
            print(f"      Rank3: {top_solutions['rank3_cost'].min()}-{top_solutions['rank3_cost'].max()} → {len(rank3_range)} values")
            print(f"      Rank4: {top_solutions['rank4_cost'].min()}-{top_solutions['rank4_cost'].max()} → {len(rank4_range)} values")
            print(f"      Rank5: {top_solutions['rank5_cost'].min()}-{top_solutions['rank5_cost'].max()} → {len(rank5_range)} values")
            print(f"      Tier2: {top_solutions['tier2_cost'].min()}-{top_solutions['tier2_cost'].max()} → {len(tier2_range)} values")
            print(f"      Tier3: {top_solutions['tier3_cost'].min()}-{top_solutions['tier3_cost'].max()} → {len(tier3_range)} values")
            
            print(f"   🎯 Creating focused search with finer steps")
        else:
            print("🔧 Generating FULL fine-grained grid combinations...")
            
            # Fine-grained ranges (2x as many values)
            rank1_range = list(range(0, 51, 2))      # 0, 2, 4, ..., 50 (26 values)
            rank2_range = list(range(0, 51, 2))      # 0, 2, 4, ..., 50 (26 values)
            rank3_range = list(range(0, 201, 10))    # 0, 10, 20, ..., 200 (21 values)
            rank4_range = list(range(0, 201, 10))    # 0, 10, 20, ..., 200 (21 values)
            rank5_range = list(range(0, 201, 10))    # 0, 10, 20, ..., 200 (21 values)
            tier2_range = list(range(0, 21, 1))      # 0, 1, 2, ..., 20 (21 values)
            tier3_range = list(range(0, 41, 2))      # 0, 2, 4, ..., 40 (21 values)
        
        # Generate all combinations
        all_combinations = list(itertools.product(
            rank1_range, rank2_range, rank3_range, rank4_range, rank5_range,
            tier2_range, tier3_range
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
        print(f"   • Tier2: {len(tier2_range)} values ({min(tier2_range)}-{max(tier2_range)})")
        print(f"   • Tier3: {len(tier3_range)} values ({min(tier3_range)}-{max(tier3_range)})")
        
        return combinations
    
    def _create_fine_range(self, min_val, max_val, steps):
        """Create a fine-grained range around min/max with specified steps."""
        if min_val == max_val:
            return [min_val]
        
        step = max(1, (max_val - min_val) // steps)
        return list(range(max(0, min_val - step), min(250, max_val + step + 1), step))


def save_fine_search_results(results: List[Dict], output_dir: Path, granularity: int, top_n: int):
    """Save fine-grained search results to files."""
    print("\n💾 Saving fine-grained search results...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    data = []
    for r in results:
        row = {
            'simulation_id': r.get('simulation_id', 0),
            'rank1_cost': r.get('cost_combo', [0]*7)[0],
            'rank2_cost': r.get('cost_combo', [0]*7)[1],
            'rank3_cost': r.get('cost_combo', [0]*7)[2],
            'rank4_cost': r.get('cost_combo', [0]*7)[3],
            'rank5_cost': r.get('cost_combo', [0]*7)[4],
            'tier2_cost': r.get('cost_combo', [0]*7)[5],
            'tier3_cost': r.get('cost_combo', [0]*7)[6],
            'satisfaction_score': r.get('satisfaction_score', 0.0),
            'gini_coefficient': r.get('gini_coefficient', 0.0),
            'fairness_score': 1.0 - r.get('gini_coefficient', 0.0),
            'num_students': r.get('num_students', 0),
            'timestamp': r.get('timestamp', ''),
            **{f'pref_{k}': v for k, v in r.get('preference_satisfaction', {}).items()}
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save CSV
    csv_path = output_dir / 'fine_grid_search_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"   ✅ Saved: {csv_path}")
    
    # Save summary
    summary_path = output_dir / 'fine_grid_search_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Fine-Grained Grid Search - Summary\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Total Simulations: {len(results)}\n")
        f.write(f"Granularity: {granularity} steps per parameter\n")
        f.write(f"Top N Solutions Analyzed: {top_n}\n\n")
        
        if results:
            # Best results by different criteria
            best_satisfaction = max(results, key=lambda x: x.get('satisfaction_score', 0.0))
            best_fairness = max(results, key=lambda x: 1.0 - x.get('gini_coefficient', 1.0))
            best_combined = max(results, key=lambda x: x.get('satisfaction_score', 0.0) * (1.0 - x.get('gini_coefficient', 1.0)))
            
            f.write(f"🎯 BEST RESULTS\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"📊 Best Combined (Satisfaction × Fairness):\n")
            f.write(f"   Cost: {best_combined.get('cost_combo', [])}\n")
            f.write(f"   Satisfaction: {best_combined.get('satisfaction_score', 0.0):.3f}\n")
            f.write(f"   Fairness: {1.0 - best_combined.get('gini_coefficient', 1.0):.3f}\n")
            f.write(f"   Combined: {best_combined.get('satisfaction_score', 0.0) * (1.0 - best_combined.get('gini_coefficient', 1.0)):.4f}\n")
            f.write(f"   Gini: {best_combined.get('gini_coefficient', 1.0):.3f}\n\n")
            
            f.write(f"😊 Best Satisfaction:\n")
            f.write(f"   Cost: {best_satisfaction.get('cost_combo', [])}\n")
            f.write(f"   Satisfaction: {best_satisfaction.get('satisfaction_score', 0.0):.3f}\n")
            f.write(f"   Fairness: {1.0 - best_satisfaction.get('gini_coefficient', 1.0):.3f}\n")
            f.write(f"   Gini: {best_satisfaction.get('gini_coefficient', 1.0):.3f}\n\n")
            
            f.write(f"⚖️  Best Fairness:\n")
            f.write(f"   Cost: {best_fairness.get('cost_combo', [])}\n")
            f.write(f"   Satisfaction: {best_fairness.get('satisfaction_score', 0.0):.3f}\n")
            f.write(f"   Fairness: {1.0 - best_fairness.get('gini_coefficient', 1.0):.3f}\n")
            f.write(f"   Gini: {best_fairness.get('gini_coefficient', 1.0):.3f}\n\n")
            
            f.write(f"\n{'='*60}\n")
            f.write(f"ALL RESULTS (Top 20 by Satisfaction)\n")
            f.write(f"{'='*60}\n\n")
            
            sorted_results = sorted(results, key=lambda x: x.get('satisfaction_score', 0.0), reverse=True)[:20]
            for i, sol in enumerate(sorted_results, 1):
                f.write(f"{i}. Cost: {sol.get('cost_combo', [])}\n")
                f.write(f"   Satisfaction: {sol.get('satisfaction_score', 0.0):.3f}\n")
                f.write(f"   Fairness: {1.0 - sol.get('gini_coefficient', 1.0):.3f}\n")
                f.write(f"   Gini: {sol.get('gini_coefficient', 1.0):.3f}\n\n")
    
    print(f"   ✅ Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Fine-grained grid search with tier support')
    parser.add_argument('--students', type=str, default='data/input/students.csv',
                       help='Students CSV file')
    parser.add_argument('--capacities', type=str, default='data/input/capacities.csv',
                       help='Capacities CSV file')
    parser.add_argument('--output', type=str, default='test_grid_search/fine_with_tiers',
                       help='Output directory')
    parser.add_argument('--focus-results', type=str, default=None,
                       help='CSV file with previous results to focus search around')
    parser.add_argument('--granularity', type=int, default=3,
                       help='Number of steps for focused search (default: 3, higher=more combinations)')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of top solutions to analyze for focused search (default: 20)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file
    log_path = output_dir / 'fine_grid_search.log'
    import sys
    log_file = open(log_path, 'w')
    
    # Create a class to write to both console and file
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    # Redirect stdout and stderr to both console and file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeOutput(original_stdout, log_file)
    sys.stderr = TeeOutput(original_stderr, log_file)
    
    try:
        print(f"📝 Logging to: {log_path}\n")
        
        # Create grid search instance
        grid_search = FineGridSearchWithTiers(output_dir)
        
        # Generate combinations
        combinations = grid_search.generate_grid_combinations(
            args.focus_results,
            granularity=args.granularity,
            top_n=args.top_n
        )
        
        print(f"\n⏱️  Estimated runtime: {len(combinations) * 0.02 / 60:.1f} minutes")
        
        # Run the actual grid search using multiprocessing
        print("\n🚀 Starting focused grid search...")
        
        # Prepare arguments for multiprocessing
        mp_args = [(i, combo, args.students, args.capacities) 
                   for i, combo in enumerate(combinations)]
        
        # Run with multiprocessing
        successful_simulations = 0
        failed_simulations = 0
        results_list = []
        
        print(f"   Processing {len(mp_args):,} allocations across 14 cores...")
        
        from tqdm import tqdm
        with mp.Pool(processes=14) as pool:
            results_iter = pool.imap(run_single_allocation, mp_args)
            
            # Track progress
            for result in tqdm(results_iter, total=len(mp_args), desc="Focused Search", unit=" combos", ncols=100):
                if result.get('success', False):
                    successful_simulations += 1
                    # Store result (already has all metrics from worker)
                    results_list.append(result)
                else:
                    failed_simulations += 1
        
        print(f"\n✅ Focused search complete!")
        print(f"   Successful: {successful_simulations:,}")
        print(f"   Failed: {failed_simulations:,}")
        
        if successful_simulations + failed_simulations > 0:
            print(f"   Success rate: {100*successful_simulations/(successful_simulations+failed_simulations):.1f}%")
        
        # Find best result
        if results_list:
            best = max(results_list, key=lambda x: x.get('satisfaction_score', 0.0))
            print(f"\n🎯 Best result:")
            print(f"   Cost: {best.get('cost_combo', [])}")
            print(f"   Satisfaction: {best.get('satisfaction_score', 0.0):.3f}")
        
        # Save results to file
        if results_list:
            save_fine_search_results(results_list, output_dir, args.granularity, args.top_n)
        
        print(f"\n🎉 Complete! Results saved to: {output_dir}")
    
    finally:
        # Restore stdout/stderr and close log file
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == '__main__':
    main()

