#!/usr/bin/env python3
"""
Benchmark Worker Performance

Tests different numbers of parallel workers to find optimal performance
for your system. Uses a small subset of combinations for speed.

Usage:
    ./benchmark_workers.py
    
    # Custom data files
    ./benchmark_workers.py --students data/input/students.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import time
from collections import defaultdict
import multiprocessing as mp
from typing import List, Tuple, Dict, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import allocation modules
import sys
sys.path.append('.')
from allocator.data_repository import DataRepository
from allocator.allocation_model_ilp import AllocationConfig as LegacyAllocationConfig
from allocator.preference_model import PreferenceModelConfig, PreferenceModel
from allocator.allocation_model_ilp import AllocationModelILP


def run_single_allocation(args):
    """Run a single allocation for benchmarking."""
    i, cost_combo, students_path, capacities_path = args
    
    try:
        # Load data
        repo = DataRepository(students_path, capacities_path)
        repo.load()
        
        # Create preference model configuration
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
        
        # Create and solve
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
        status = diagnostics.get('status', '')
        is_successful = status not in ['Infeasible', 'Undefined', 'Unbounded']
        
        return {
            'success': is_successful and len(allocation) > 0,
            'simulation_id': i,
            'cost_combo': cost_combo
        }
    except Exception as e:
        return {
            'success': False,
            'simulation_id': i,
            'cost_combo': cost_combo,
            'error': str(e)[:50]
        }


def benchmark_workers(num_workers: int, test_combinations: List[Tuple], 
                      students_path: str, capacities_path: str) -> Dict[str, Any]:
    """Benchmark performance with a specific number of workers."""
    print(f"  🧪 Testing {num_workers} workers...", end="", flush=True)
    
    start_time = time.time()
    
    # Prepare arguments
    args = [(i, combo, students_path, capacities_path) 
            for i, combo in enumerate(test_combinations)]
    
    # Run with specified number of workers
    with mp.Pool(processes=num_workers) as pool:
        results = list(pool.imap(run_single_allocation, args))
    
    elapsed = time.time() - start_time
    
    # Count successes
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    
    print(f" ✓ ({elapsed:.1f}s, {successful}/{len(results)} success)")
    
    return {
        'num_workers': num_workers,
        'total_time': elapsed,
        'time_per_combo': elapsed / len(test_combinations),
        'combinations_per_second': len(test_combinations) / elapsed,
        'successful': successful,
        'failed': failed,
        'success_rate': successful / len(test_combinations) * 100
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark worker performance')
    parser.add_argument('--students', type=str, default='data/input/students.csv',
                       help='Students CSV file')
    parser.add_argument('--capacities', type=str, default='data/input/capacities.csv',
                       help='Capacities CSV file')
    parser.add_argument('--num-tests', type=int, default=500,
                       help='Number of combinations to test')
    parser.add_argument('--output', type=str, default='benchmark_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔧 Generating test combinations...")
    
    import itertools
    
    # Generate exactly the number of combinations requested
    target_count = args.num_tests
    
    # Start with minimal ranges and expand until we get enough
    rank1_range = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    rank2_range = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    rank3_range = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    rank4_range = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    rank5_range = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    tier2_range = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    tier3_range = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    
    # Generate all combinations
    all_combinations = list(itertools.product(
        rank1_range, rank2_range, rank3_range, rank4_range, rank5_range,
        tier2_range, tier3_range
    ))
    
    # Filter to monotonic
    combinations = [
        combo for combo in all_combinations
        if combo[0] <= combo[1] <= combo[2] <= combo[3] <= combo[4]
    ]
    
    # Take exactly target_count combinations
    np.random.seed(42)  # Reproducible
    if len(combinations) <= target_count:
        test_combinations = combinations
    else:
        test_indices = np.random.choice(len(combinations), size=target_count, replace=False)
        test_combinations = [combinations[i] for i in sorted(test_indices)]
    
    print(f"   Generated {len(test_combinations):,} test combinations")
    print()
    
    # Test different worker counts
    max_workers = min(mp.cpu_count(), 20)  # Test up to 20 or CPU count
    worker_counts = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20][:max_workers//2 + 1]
    worker_counts = [w for w in worker_counts if w <= max_workers]
    
    print(f"📊 Benchmarking {len(worker_counts)} worker configurations...")
    print(f"   Testing: {worker_counts}")
    print()
    
    results = []
    for num_workers in worker_counts:
        result = benchmark_workers(num_workers, test_combinations,
                                   args.students, args.capacities)
        results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    csv_path = output_dir / 'benchmark_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved: {csv_path}")
    
    # Find optimal
    # Prefer more cores if time improvement is significant
    df['speedup'] = df.iloc[0]['total_time'] / df['total_time']  # vs single-core
    df['efficiency'] = df['speedup'] / df['num_workers']  # speedup per worker
    
    # Optimal: balance between speed and efficiency
    # Find where additional workers give diminishing returns
    df['marginal_gain'] = df['speedup'].diff()
    
    # Recommend: highest speedup with efficiency > 0.5
    candidates = df[(df['efficiency'] > 0.5) & (df['marginal_gain'] > 0.05)]
    if len(candidates) > 0:
        optimal = candidates['num_workers'].max()
    else:
        optimal = df.loc[df['speedup'].idxmax(), 'num_workers']
    
    print(f"\n🎯 Recommendation: Use {optimal} workers")
    print(f"   This achieves {df.loc[df['num_workers']==optimal, 'speedup'].values[0]:.1f}x speedup")
    print(f"   At {df.loc[df['num_workers']==optimal, 'efficiency'].values[0]:.1%} efficiency")
    
    # Create visualization
    create_plots(df, output_dir, optimal)
    
    print(f"\n✅ Benchmark complete! Results in: {output_dir}")


def create_plots(df, output_dir, optimal):
    """Create visualization of benchmark results."""
    print("📈 Creating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Worker Performance Benchmark', fontsize=14, fontweight='bold')
    
    # 1. Total Time
    ax = axes[0, 0]
    ax.plot(df['num_workers'], df['total_time'], 'o-', linewidth=2, markersize=8)
    ax.axvline(optimal, color='red', linestyle='--', alpha=0.7, label=f'Recommended: {optimal}')
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Total Time (seconds)')
    ax.set_title('Total Execution Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Speedup
    ax = axes[0, 1]
    single_core_time = df.loc[df['num_workers'] == 1, 'total_time'].values[0]
    ax.plot(df['num_workers'], df['total_time'], 'o-', linewidth=2, markersize=8, label='Actual')
    ax.axhline(single_core_time, color='gray', linestyle='--', alpha=0.5, label='Single-core baseline')
    ax.axvline(optimal, color='red', linestyle='--', alpha=0.7, label=f'Recommended: {optimal}')
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Time vs Baseline')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Throughput
    ax = axes[1, 0]
    ax.bar(df['num_workers'], df['combinations_per_second'], alpha=0.7, color='steelblue')
    ax.axvline(optimal, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Combinations/Second')
    ax.set_title('Throughput')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Efficiency
    ax = axes[1, 1]
    ax.bar(df['num_workers'], df['efficiency'] * 100, alpha=0.7, color='green')
    ax.axvline(optimal, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.axhline(50, color='orange', linestyle='--', alpha=0.5, label='50% threshold')
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Parallel Efficiency')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    
    plot_path = output_dir / 'benchmark_plots.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {plot_path}")


if __name__ == '__main__':
    main()

