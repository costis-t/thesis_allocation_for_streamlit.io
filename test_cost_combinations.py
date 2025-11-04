#!/usr/bin/env python3
"""
Test different cost combinations for allocation

This script allows systematic testing of different cost configurations
to understand their impact on allocation outcomes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from collections import Counter
from typing import Dict, Any, List, Tuple
import time
from datetime import datetime

# Import allocation modules
from allocator.data_repository import DataRepository
from allocator.allocation_model_ilp import AllocationConfig as LegacyAllocationConfig
from allocator.allocation_model_flow import AllocationModelFlow
from allocator.preference_model import PreferenceModelConfig
from allocator.config import CapacityConfig, SolverConfig

def calculate_gini_coefficient(allocation_df: pd.DataFrame) -> float:
    """Calculate Gini coefficient for fairness analysis."""
    if allocation_df.empty:
        return 0.0
    
    # Extract costs from allocation (assuming cost is in a 'cost' column)
    if 'cost' not in allocation_df.columns:
        return 0.0
    
    costs = allocation_df['cost'].values
    if len(costs) == 0:
        return 0.0
    
    # Sort costs
    sorted_costs = np.sort(costs)
    n = len(sorted_costs)
    
    if n == 0:
        return 0.0
    
    # Calculate Gini coefficient
    cumsum = np.cumsum(sorted_costs)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0

def extract_metrics_from_summary(summary: str) -> Dict[str, Any]:
    """Extract key metrics from allocation summary text."""
    metrics = {
        'total_cost': None,
        'average_cost': None,
        'preference_satisfaction': {},
        'utilization': {},
        'fairness': None
    }
    
    # Extract objective value
    if 'Objective Value:' in summary:
        try:
            obj_line = [line for line in summary.split('\n') if 'Objective Value:' in line][0]
            metrics['total_cost'] = float(obj_line.split(':')[1].strip())
        except:
            pass
    
    return metrics

def run_allocation_with_costs(rank1_cost: int, rank2_cost: int, rank3_cost: int, 
                            rank4_cost: int, rank5_cost: int, top2_bias: bool, 
                            unranked_cost: int, algorithm: str, 
                            students_path: str, capacities_path: str) -> Dict[str, Any]:
    """Run allocation with specific cost configuration."""
    try:
        # Load data repository
        repo = DataRepository(students_path, capacities_path)
        repo.load()
        
        # Create preference model configuration
        pref_cfg = PreferenceModelConfig(
            rank1_cost=rank1_cost,
            rank2_cost=rank2_cost,
            rank3_cost=rank3_cost,
            rank4_cost=rank4_cost,
            rank5_cost=rank5_cost,
            top2_bias=top2_bias,
            unranked_cost=unranked_cost
        )
        
        # Create preference model
        from allocator.preference_model import PreferenceModel
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
        
        if algorithm == "ilp":
            from allocator.allocation_model_ilp import AllocationModelILP
            model = AllocationModelILP(
                students=repo.students,
                topics=repo.topics,
                coaches=repo.coaches,
                departments=repo.departments,
                pref_model=pref_model,
                cfg=allocation_cfg
            )
        else:  # flow
            model = AllocationModelFlow(
                students=repo.students,
                topics=repo.topics,
                coaches=repo.coaches,
                departments=repo.departments,
                pref_model=pref_model,
                cfg=allocation_cfg
            )
        
        # Build and solve model
        model.build()
        rows, diagnostics = model.solve()
        
        # Convert rows to allocation dictionary
        allocation = {}
        for row in rows:
            allocation[row.student] = row.assigned_topic
        
        # Create summary from diagnostics
        summary = f"""
Status: {diagnostics.get('status', 'Unknown')}
Objective Value: {diagnostics.get('objective_value', 'N/A')}
Unassignable Students: {len(diagnostics.get('unassignable_students', []))}
Unassigned After Solve: {len(diagnostics.get('unassigned_after_solve', []))}
Topic Overflow: {diagnostics.get('topic_overflow', {})}
Coach Overflow: {diagnostics.get('coach_overflow', {})}
Department Shortfall: {diagnostics.get('department_shortfall', {})}
Tied Students: {len(diagnostics.get('tied_students', []))}
"""
        
        # Calculate preference satisfaction analysis
        from collections import Counter
        pref_counts = Counter(row.preference_rank for row in rows)
        
        # Extract metrics
        metrics = extract_metrics_from_summary(summary)
        metrics['objective_value'] = diagnostics.get('objective_value')
        metrics['status'] = diagnostics.get('status')
        
        # Add preference satisfaction metrics
        metrics['preference_satisfaction'] = {
            'tier1': pref_counts.get(0, 0),
            'tier2': pref_counts.get(1, 0), 
            'tier3': pref_counts.get(2, 0),
            'rank1': pref_counts.get(10, 0),
            'rank2': pref_counts.get(11, 0),
            'rank3': pref_counts.get(12, 0),
            'rank4': pref_counts.get(13, 0),
            'rank5': pref_counts.get(14, 0),
            'unranked': pref_counts.get(999, 0)
        }
        
        # Add objective value as total cost if available
        if diagnostics.get('objective_value') is not None:
            metrics['total_cost'] = diagnostics.get('objective_value')
        
        # Calculate additional metrics
        allocation_df = pd.DataFrame([
            {'student': student_id, 'assigned_topic': topic_id}
            for student_id, topic_id in allocation.items()
        ])
        
        # Calculate Gini coefficient
        metrics['gini_coefficient'] = calculate_gini_coefficient(allocation_df)
        metrics['num_students'] = len(allocation)
        
        return {
            'success': True,
            'allocation': allocation,
            'summary': summary,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }

def generate_cost_combinations(mode: str, **kwargs) -> List[Tuple[int, int, int, int, int]]:
    """Generate different cost combinations based on mode."""
    combinations = []
    
    if mode == "linear":
        # Linear progression: (0,1,2,3,4), (0,2,4,6,8), etc.
        step = kwargs.get('step', 1)
        max_cost = kwargs.get('max_cost', 20)
        
        for base in range(0, max_cost + 1, step):
            combinations.append((base, base + step, base + 2*step, base + 3*step, base + 4*step))
    
    elif mode == "exponential":
        # Exponential: (0,1,10,100,1000), (0,1,20,200,2000), etc.
        multiplier = kwargs.get('multiplier', 10)
        max_multiplier = kwargs.get('max_multiplier', 1000)
        
        for base in range(multiplier, max_multiplier + 1, multiplier):
            combinations.append((0, 1, base, base*10, base*100))
    
    elif mode == "range":
        # Range-based: test all combinations within specified ranges
        ranges = kwargs.get('ranges', [0, 10, 0, 10, 0, 10, 0, 10, 0, 10])
        step = kwargs.get('step', 1)
        
        # Convert ranges to tuples
        range_tuples = []
        for i in range(0, len(ranges), 2):
            range_tuples.append((ranges[i], ranges[i+1]))
        
        # Generate combinations
        for rank1 in range(range_tuples[0][0], range_tuples[0][1] + 1, step):
            for rank2 in range(range_tuples[1][0], range_tuples[1][1] + 1, step):
                for rank3 in range(range_tuples[2][0], range_tuples[2][1] + 1, step):
                    for rank4 in range(range_tuples[3][0], range_tuples[3][1] + 1, step):
                        for rank5 in range(range_tuples[4][0], range_tuples[4][1] + 1, step):
                            combinations.append((rank1, rank2, rank3, rank4, rank5))
        
        return combinations
    
    elif mode == "top2_bias":
        # Top-2 bias variations: (0,1,100,101,102), (0,1,200,201,202), etc.
        base_costs = kwargs.get('base_costs', [100, 200, 300, 400, 500])
        combinations = []
        for base in base_costs:
            combinations.append((0, 1, base, base+1, base+2))
        return combinations
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

def save_results(results: List[Dict[str, Any]], output_dir: Path) -> None:
    """Save results to files."""
    output_dir.mkdir(exist_ok=True)
    
    # Save raw results
    with open(output_dir / "raw_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary metrics
    summary_data = []
    for result in results:
        row = {
            'rank1_cost': result['rank_costs'][0],
            'rank2_cost': result['rank_costs'][1],
            'rank3_cost': result['rank_costs'][2],
            'rank4_cost': result['rank_costs'][3],
            'rank5_cost': result['rank_costs'][4],
            'top2_bias': result['top2_bias'],
            'unranked_cost': result['unranked_cost'],
            'algorithm': result['algorithm'],
            'total_cost': result['metrics'].get('total_cost'),
            'average_cost': result['metrics'].get('average_cost'),
            'gini_coefficient': result['metrics'].get('gini_coefficient'),
            'num_students': result['metrics'].get('num_students'),
            'timestamp': result['timestamp']
        }
        
        # Add preference satisfaction metrics
        for pref_type, count in result['metrics'].get('preference_satisfaction', {}).items():
            row[f'pref_{pref_type}'] = count
        
        # Add utilization metrics
        for util_type, value in result['metrics'].get('utilization', {}).items():
            row[f'util_{util_type}'] = value
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    
    # Save individual allocations as CSV
    for i, result in enumerate(results):
        allocation_df = pd.DataFrame([
            {'student': student_id, 'assigned_topic': topic_id}
            for student_id, topic_id in result['allocation'].items()
        ])
        allocation_df.to_csv(output_dir / f"allocation_{i:03d}.csv", index=False)
    
    # Create analysis script
    analysis_script = '''#!/usr/bin/env python3
"""
Analyze cost testing results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def main():
    # Load results
    df = pd.read_csv('summary_metrics.csv')
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Cost vs Satisfaction
    axes[0, 0].scatter(df['total_cost'], df['pref_rank1'], alpha=0.6)
    axes[0, 0].set_title('Total Cost vs 1st Choice Satisfaction')
    axes[0, 0].set_xlabel('Total Cost')
    axes[0, 0].set_ylabel('Students with 1st Choice')
    
    # Plot 2: Cost vs Fairness
    axes[0, 1].scatter(df['total_cost'], df['gini_coefficient'], alpha=0.6)
    axes[0, 1].set_title('Total Cost vs Gini Coefficient')
    axes[0, 1].set_xlabel('Total Cost')
    axes[0, 1].set_ylabel('Gini Coefficient')
    
    # Plot 3: Rank1 Cost vs Satisfaction
    axes[1, 0].scatter(df['rank1_cost'], df['pref_rank1'], alpha=0.6)
    axes[1, 0].set_title('Rank1 Cost vs 1st Choice Satisfaction')
    axes[1, 0].set_xlabel('Rank1 Cost')
    axes[1, 0].set_ylabel('Students with 1st Choice')
    
    # Plot 4: Rank2 Cost vs Fairness
    axes[1, 1].scatter(df['rank2_cost'], df['gini_coefficient'], alpha=0.6)
    axes[1, 1].set_title('Rank2 Cost vs Gini Coefficient')
    axes[1, 1].set_xlabel('Rank2 Cost')
    axes[1, 1].set_ylabel('Gini Coefficient')
    
    plt.tight_layout()
    plt.savefig('cost_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Analysis complete! Check cost_analysis.png")

if __name__ == "__main__":
    main()
'''
    
    with open(output_dir / "analyze_results.py", 'w') as f:
        f.write(analysis_script)

def main():
    parser = argparse.ArgumentParser(description='Test different cost combinations for allocation')
    parser.add_argument('--students', type=str, default='data/input/students.csv', help='Students CSV file')
    parser.add_argument('--capacities', type=str, default='data/input/capacities.csv', help='Capacities CSV file')
    parser.add_argument('--output', type=str, default='cost_test_results', help='Output directory')
    parser.add_argument('--algorithm', type=str, choices=['ilp', 'flow'], default='ilp', help='Algorithm to use')
    parser.add_argument('--mode', type=str, choices=['linear', 'exponential', 'range', 'top2_bias'], 
                       default='linear', help='Cost combination mode')
    parser.add_argument('--step', type=int, default=1, help='Step size for cost ranges')
    parser.add_argument('--max-cost', type=int, default=20, help='Maximum cost for linear mode')
    parser.add_argument('--multiplier', type=int, default=10, help='Multiplier for exponential mode')
    parser.add_argument('--max-multiplier', type=int, default=1000, help='Max multiplier for exponential mode')
    parser.add_argument('--base-costs', nargs='+', type=int, default=[100, 200, 300, 400, 500],
                       help='Base costs for top2_bias mode')
    parser.add_argument('--ranges', nargs=10, type=int, default=[0, 10, 0, 10, 0, 10, 0, 10, 0, 10],
                       help='Ranges for each rank (5 pairs: min1,max1,min2,max2,...)')
    parser.add_argument('--top2-bias', action='store_true', help='Enable top-2 bias')
    parser.add_argument('--unranked-cost', type=int, default=200, help='Cost for unranked topics')
    parser.add_argument('--limit', type=int, help='Limit number of combinations to test')
    
    args = parser.parse_args()
    
    # Check input files
    students_path = Path(args.students)
    capacities_path = Path(args.capacities)
    
    if not students_path.exists():
        print(f"❌ Students file not found: {students_path}")
        return
    
    if not capacities_path.exists():
        print(f"❌ Capacities file not found: {capacities_path}")
        return
    
    # Generate cost combinations
    print(f"🔧 Generating cost combinations using {args.mode} mode...")
    combinations = generate_cost_combinations(
        args.mode,
        step=args.step,
        max_cost=args.max_cost,
        multiplier=args.multiplier,
        max_multiplier=args.max_multiplier,
        base_costs=args.base_costs,
        ranges=args.ranges
    )
    
    # Limit combinations if specified
    if args.limit:
        combinations = combinations[:args.limit]
    
    print(f"📊 Testing {len(combinations)} cost combinations...")
    
    # Run tests
    results = []
    successful_tests = 0
    
    for i, combo in enumerate(combinations):
        print(f"🔄 Running combination {i+1}/{len(combinations)}: {combo}")
        
        result = run_allocation_with_costs(
            rank1_cost=combo[0],
            rank2_cost=combo[1],
            rank3_cost=combo[2],
            rank4_cost=combo[3],
            rank5_cost=combo[4],
            top2_bias=args.top2_bias,
            unranked_cost=args.unranked_cost,
            algorithm=args.algorithm,
            students_path=str(students_path),
            capacities_path=str(capacities_path)
        )
        
        if result['success']:
            result['rank_costs'] = combo
            result['top2_bias'] = args.top2_bias
            result['unranked_cost'] = args.unranked_cost
            result['algorithm'] = args.algorithm
            results.append(result)
            successful_tests += 1
            print(f"✅ Success: Total cost = {result['metrics'].get('total_cost', 'N/A')}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Save results
    output_dir = Path(args.output)
    save_results(results, output_dir)
    
    print(f"✅ Results saved to {output_dir}/")
    print(f"📊 Run 'python {output_dir}/analyze_results.py' to generate analysis plots")
    print(f"🎉 Completed! Tested {successful_tests} successful combinations.")

if __name__ == "__main__":
    main()
