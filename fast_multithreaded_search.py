#!/usr/bin/env python3
"""
Multithreaded Optimized Grid Search with Progress Bars

This script implements a fast, multithreaded grid search with real-time progress tracking
and focuses on the most promising regions of the cost space.
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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import multiprocessing as mp
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Import our existing modules
import sys
sys.path.append('.')
from test_cost_combinations import run_allocation_with_costs, calculate_gini_coefficient

@dataclass
class FastSearchResult:
    """Container for fast search results."""
    cost_combination: Tuple[int, int, int, int, int]
    satisfaction_score: float
    fairness_score: float
    total_cost: float
    preference_satisfaction: Dict[str, int]
    gini_coefficient: float
    num_students: int
    algorithm: str
    timestamp: str
    simulation_id: int
    search_region: str

class FastMultithreadedSearch:
    """Implements fast multithreaded grid search with progress tracking."""
    
    def __init__(self, output_dir: Path, max_workers: int = None):
        self.output_dir = Path(output_dir)
        self.results = []
        self.pareto_frontier = []
        self.simulation_count = 0
        self.start_time = time.time()
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.results_lock = threading.Lock()
        
    def generate_fast_combinations(self) -> List[Tuple[Tuple[int, int, int, int, int], str]]:
        """Generate optimized combinations focusing on promising regions."""
        print("🔧 Generating fast combinations focusing on promising regions...")
        
        combinations = []
        
        # Region 1: Low rank1, moderate rank2-3, high rank4-5 (based on previous findings)
        print("  📍 Region 1: Low rank1, moderate rank2-3, high rank4-5")
        rank1_range = list(range(0, 21, 2))    # 0, 2, 4, ..., 20 (11 values)
        rank2_range = list(range(40, 81, 4))   # 40, 44, 48, ..., 80 (11 values)
        rank3_range = list(range(60, 121, 6))  # 60, 66, 72, ..., 120 (11 values)
        rank4_range = list(range(80, 121, 4))  # 80, 84, 88, ..., 120 (11 values)
        rank5_range = list(range(80, 121, 4))  # 80, 84, 88, ..., 120 (11 values)
        
        region1_combinations = list(itertools.product(rank1_range, rank2_range, rank3_range, rank4_range, rank5_range))
        combinations.extend([(combo, "region1") for combo in region1_combinations])
        
        # Region 2: Very low rank1-2, high rank3-5 (top satisfaction region)
        print("  📍 Region 2: Very low rank1-2, high rank3-5")
        rank1_range = list(range(0, 11, 1))    # 0, 1, 2, ..., 10 (11 values)
        rank2_range = list(range(0, 11, 1))    # 0, 1, 2, ..., 10 (11 values)
        rank3_range = list(range(80, 201, 12)) # 80, 92, 104, ..., 200 (11 values)
        rank4_range = list(range(80, 201, 12)) # 80, 92, 104, ..., 200 (11 values)
        rank5_range = list(range(80, 201, 12)) # 80, 92, 104, ..., 200 (11 values)
        
        region2_combinations = list(itertools.product(rank1_range, rank2_range, rank3_range, rank4_range, rank5_range))
        combinations.extend([(combo, "region2") for combo in region2_combinations])
        
        # Region 3: Balanced moderate costs (balanced region)
        print("  📍 Region 3: Balanced moderate costs")
        rank1_range = list(range(10, 31, 2))   # 10, 12, 14, ..., 30 (11 values)
        rank2_range = list(range(30, 51, 2))   # 30, 32, 34, ..., 50 (11 values)
        rank3_range = list(range(50, 101, 5))  # 50, 55, 60, ..., 100 (11 values)
        rank4_range = list(range(70, 121, 5))  # 70, 75, 80, ..., 120 (11 values)
        rank5_range = list(range(70, 121, 5))  # 70, 75, 80, ..., 120 (11 values)
        
        region3_combinations = list(itertools.product(rank1_range, rank2_range, rank3_range, rank4_range, rank5_range))
        combinations.extend([(combo, "region3") for combo in region3_combinations])
        
        # Region 4: High rank1-2, moderate rank3-5 (fairness region)
        print("  📍 Region 4: High rank1-2, moderate rank3-5")
        rank1_range = list(range(20, 51, 3))   # 20, 23, 26, ..., 50 (11 values)
        rank2_range = list(range(40, 71, 3))   # 40, 43, 46, ..., 70 (11 values)
        rank3_range = list(range(50, 101, 5))  # 50, 55, 60, ..., 100 (11 values)
        rank4_range = list(range(60, 111, 5))  # 60, 65, 70, ..., 110 (11 values)
        rank5_range = list(range(60, 111, 5))  # 60, 65, 70, ..., 110 (11 values)
        
        region4_combinations = list(itertools.product(rank1_range, rank2_range, rank3_range, rank4_range, rank5_range))
        combinations.extend([(combo, "region4") for combo in region4_combinations])
        
        # Region 5: Extreme values (boundary exploration)
        print("  📍 Region 5: Extreme values")
        extreme_combinations = [
            # Very low costs
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 10),
            (0, 0, 0, 10, 10),
            (0, 0, 10, 10, 10),
            (0, 10, 10, 10, 10),
            # Very high costs
            (50, 50, 200, 200, 200),
            (100, 100, 200, 200, 200),
            # Mixed extreme
            (0, 100, 0, 200, 0),
            (100, 0, 200, 0, 200),
            (50, 0, 100, 0, 150),
        ]
        combinations.extend([(combo, "region5") for combo in extreme_combinations])
        
        print(f"📊 Generated {len(combinations):,} optimized combinations")
        print(f"   • Region 1: {len(region1_combinations):,} combinations")
        print(f"   • Region 2: {len(region2_combinations):,} combinations")
        print(f"   • Region 3: {len(region3_combinations):,} combinations")
        print(f"   • Region 4: {len(region4_combinations):,} combinations")
        print(f"   • Region 5: {len(extreme_combinations)} combinations")
        
        return combinations
    
    def _calculate_satisfaction_score(self, pref_satisfaction: Dict[str, int]) -> float:
        """Calculate a single satisfaction score from preference satisfaction data."""
        if not pref_satisfaction:
            return 0.0
        
        # Weighted satisfaction score
        weights = {
            'rank1': 5.0,    # 1st choice gets highest weight
            'rank2': 4.0,    # 2nd choice
            'rank3': 3.0,    # 3rd choice
            'rank4': 2.0,    # 4th choice
            'rank5': 1.0,    # 5th choice
            'unranked': 0.0  # Unranked gets no weight
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
        max_possible_score = total_students * weights['rank1']
        return total_weighted_satisfaction / max_possible_score if max_possible_score > 0 else 0.0
    
    def _run_single_simulation(self, args) -> Tuple[FastSearchResult, bool]:
        """Run a single simulation - designed for multithreading."""
        cost_combo, region, simulation_id, students_path, capacities_path = args
        
        try:
            # Run allocation
            result = run_allocation_with_costs(
                rank1_cost=cost_combo[0], 
                rank2_cost=cost_combo[1], 
                rank3_cost=cost_combo[2], 
                rank4_cost=cost_combo[3], 
                rank5_cost=cost_combo[4], 
                top2_bias=False, 
                unranked_cost=200, 
                algorithm="ilp",
                students_path=students_path, 
                capacities_path=capacities_path
            )
            
            if result and result.get('success', False):
                search_result = FastSearchResult(
                    cost_combination=cost_combo,
                    satisfaction_score=self._calculate_satisfaction_score(result['metrics'].get('preference_satisfaction', {})),
                    fairness_score=1.0 - result['metrics'].get('gini_coefficient', 0.5),
                    total_cost=result['metrics'].get('total_cost', 0),
                    preference_satisfaction=result['metrics'].get('preference_satisfaction', {}),
                    gini_coefficient=result['metrics'].get('gini_coefficient', 0.5),
                    num_students=result['metrics'].get('num_students', 0),
                    algorithm="ilp",
                    timestamp=result['timestamp'],
                    simulation_id=simulation_id,
                    search_region=region
                )
                return search_result, True
            else:
                return None, False
                
        except Exception as e:
            return None, False
    
    def run_fast_multithreaded_search(self, combinations: List[Tuple[Tuple[int, int, int, int, int], str]], 
                                    students_path: str, capacities_path: str) -> List[FastSearchResult]:
        """Run fast multithreaded grid search with progress tracking."""
        print(f"🚀 Starting fast multithreaded search with {len(combinations):,} combinations...")
        print(f"🔧 Using {self.max_workers} worker threads")
        
        # Prepare arguments for multithreading
        simulation_args = []
        for i, (cost_combo, region) in enumerate(combinations):
            simulation_args.append((cost_combo, region, i + 1, students_path, capacities_path))
        
        results = []
        successful_simulations = 0
        failed_simulations = 0
        region_stats = defaultdict(int)
        
        # Create progress bar
        with tqdm(total=len(combinations), desc="🔍 Running simulations", unit="sim") as pbar:
            # Use ThreadPoolExecutor for I/O bound operations
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_args = {executor.submit(self._run_single_simulation, args): args for args in simulation_args}
                
                # Process completed tasks
                for future in as_completed(future_to_args):
                    args = future_to_args[future]
                    cost_combo, region, simulation_id, _, _ = args
                    
                    try:
                        result, success = future.result()
                        
                        if success and result:
                            with self.results_lock:
                                results.append(result)
                                successful_simulations += 1
                                region_stats[region] += 1
                        else:
                            failed_simulations += 1
                            
                    except Exception as e:
                        failed_simulations += 1
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Update progress bar description with stats
                    if successful_simulations + failed_simulations > 0:
                        success_rate = (successful_simulations / (successful_simulations + failed_simulations)) * 100
                        pbar.set_description(f"🔍 Running simulations (Success: {success_rate:.1f}%)")
        
        self.results = results
        print(f"\n✅ Fast multithreaded search complete!")
        print(f"   • Total simulations: {len(combinations):,}")
        print(f"   • Successful: {successful_simulations:,}")
        print(f"   • Failed: {failed_simulations:,}")
        print(f"   • Success rate: {(successful_simulations/len(combinations)*100):.1f}%")
        print(f"   • Region breakdown: {dict(region_stats)}")
        print(f"   • Total time: {(time.time() - self.start_time)/60:.1f} minutes")
        print(f"   • Rate: {len(combinations)/(time.time() - self.start_time):.1f} simulations/sec")
        
        return results
    
    def find_pareto_frontier(self) -> List[FastSearchResult]:
        """Find Pareto-optimal solutions from fast search results."""
        print("🔍 Finding Pareto frontier from fast search results...")
        
        if not self.results:
            print("❌ No results to analyze")
            return []
        
        pareto_solutions = []
        
        # Use tqdm for progress tracking
        with tqdm(total=len(self.results), desc="🔍 Finding Pareto frontier", unit="sol") as pbar:
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
                
                pbar.update(1)
        
        self.pareto_frontier = pareto_solutions
        print(f"✅ Found {len(pareto_solutions)} Pareto-optimal solutions")
        
        return pareto_solutions
    
    def analyze_region_performance(self) -> Dict[str, Any]:
        """Analyze performance by region."""
        print("📊 Analyzing region performance...")
        
        if not self.results:
            return {}
        
        region_analysis = {}
        
        for region in ['region1', 'region2', 'region3', 'region4', 'region5']:
            region_results = [r for r in self.results if r.search_region == region]
            
            if region_results:
                satisfaction_scores = [r.satisfaction_score for r in region_results]
                fairness_scores = [r.fairness_score for r in region_results]
                
                region_analysis[region] = {
                    'count': len(region_results),
                    'satisfaction_mean': np.mean(satisfaction_scores),
                    'satisfaction_max': max(satisfaction_scores),
                    'fairness_mean': np.mean(fairness_scores),
                    'fairness_max': max(fairness_scores),
                    'pareto_count': len([r for r in self.pareto_frontier if r.search_region == region])
                }
        
        return region_analysis
    
    def create_fast_visualizations(self):
        """Create visualizations for fast search results."""
        print("📈 Creating fast visualizations...")
        
        if not self.results:
            print("❌ No results to visualize")
            return
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fast Multithreaded Grid Search: Multi-Objective Optimization Results', fontsize=16, fontweight='bold')
        
        # Extract data
        all_results = self.results
        pareto_results = self.pareto_frontier
        
        satisfaction_scores = [r.satisfaction_score for r in all_results]
        fairness_scores = [r.fairness_score for r in all_results]
        total_costs = [r.total_cost for r in all_results]
        regions = [r.search_region for r in all_results]
        pareto_satisfaction = [r.satisfaction_score for r in pareto_results]
        pareto_fairness = [r.fairness_score for r in pareto_results]
        
        # Plot 1: Pareto Frontier by Region
        region_colors = {'region1': 'red', 'region2': 'blue', 'region3': 'green', 'region4': 'orange', 'region5': 'purple'}
        for region in set(regions):
            region_results = [r for r in all_results if r.search_region == region]
            region_satisfaction = [r.satisfaction_score for r in region_results]
            region_fairness = [r.fairness_score for r in region_results]
            axes[0, 0].scatter(region_satisfaction, region_fairness, alpha=0.6, s=20, 
                             label=f'{region}', c=region_colors.get(region, 'gray'))
        
        axes[0, 0].scatter(pareto_satisfaction, pareto_fairness, alpha=0.9, s=80, c='black', 
                          label='Pareto Frontier', edgecolors='white', linewidth=2)
        axes[0, 0].set_title('Pareto Frontier by Search Region')
        axes[0, 0].set_xlabel('Satisfaction Score')
        axes[0, 0].set_ylabel('Fairness Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Region Performance Comparison
        region_stats = self.analyze_region_performance()
        if region_stats:
            regions_list = list(region_stats.keys())
            satisfaction_means = [region_stats[r]['satisfaction_mean'] for r in regions_list]
            fairness_means = [region_stats[r]['fairness_mean'] for r in regions_list]
            
            x = np.arange(len(regions_list))
            width = 0.35
            
            axes[0, 1].bar(x - width/2, satisfaction_means, width, label='Satisfaction', alpha=0.8)
            axes[0, 1].bar(x + width/2, fairness_means, width, label='Fairness', alpha=0.8)
            axes[0, 1].set_title('Average Performance by Region')
            axes[0, 1].set_xlabel('Search Region')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(regions_list)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Cost vs Satisfaction
        axes[0, 2].scatter(total_costs, satisfaction_scores, alpha=0.6, s=20)
        axes[0, 2].set_title('Total Cost vs Satisfaction')
        axes[0, 2].set_xlabel('Total Cost')
        axes[0, 2].set_ylabel('Satisfaction Score')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Cost vs Fairness
        axes[1, 0].scatter(total_costs, fairness_scores, alpha=0.6, s=20)
        axes[1, 0].set_title('Total Cost vs Fairness')
        axes[1, 0].set_xlabel('Total Cost')
        axes[1, 0].set_ylabel('Fairness Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Pareto Solutions Detail
        if pareto_results:
            pareto_costs = [sum(r.cost_combination) for r in pareto_results]
            pareto_combined_scores = [r.satisfaction_score + r.fairness_score for r in pareto_results]
            axes[1, 1].scatter(pareto_costs, pareto_combined_scores, c='red', alpha=0.8, s=80)
            axes[1, 1].set_title('Pareto Solutions: Total Cost vs Combined Score')
            axes[1, 1].set_xlabel('Total Cost')
            axes[1, 1].set_ylabel('Combined Score (Satisfaction + Fairness)')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Region Distribution
        region_counts = Counter(regions)
        axes[1, 2].pie(region_counts.values(), labels=region_counts.keys(), autopct='%1.1f%%')
        axes[1, 2].set_title('Distribution of Simulations by Region')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fast_multithreaded_search_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_fast_results(self):
        """Save fast search results."""
        print("💾 Saving fast results...")
        
        self.output_dir.mkdir(exist_ok=True)
        
        # Save all results
        all_data = []
        for result in self.results:
            all_data.append({
                'simulation_id': result.simulation_id,
                'search_region': result.search_region,
                'rank1_cost': result.cost_combination[0],
                'rank2_cost': result.cost_combination[1],
                'rank3_cost': result.cost_combination[2],
                'rank4_cost': result.cost_combination[3],
                'rank5_cost': result.cost_combination[4],
                'satisfaction_score': result.satisfaction_score,
                'fairness_score': result.fairness_score,
                'total_cost': result.total_cost,
                'gini_coefficient': result.gini_coefficient,
                'num_students': result.num_students,
                'is_pareto_optimal': result in self.pareto_frontier,
                'timestamp': result.timestamp
            })
        
        all_df = pd.DataFrame(all_data)
        all_df.to_csv(self.output_dir / 'fast_multithreaded_results.csv', index=False)
        
        # Save Pareto frontier
        pareto_data = []
        for result in self.pareto_frontier:
            pareto_data.append({
                'simulation_id': result.simulation_id,
                'search_region': result.search_region,
                'rank1_cost': result.cost_combination[0],
                'rank2_cost': result.cost_combination[1],
                'rank3_cost': result.cost_combination[2],
                'rank4_cost': result.cost_combination[3],
                'rank5_cost': result.cost_combination[4],
                'satisfaction_score': result.satisfaction_score,
                'fairness_score': result.fairness_score,
                'total_cost': result.total_cost,
                'gini_coefficient': result.gini_coefficient,
                'num_students': result.num_students,
                'preference_satisfaction': result.preference_satisfaction,
                'timestamp': result.timestamp
            })
        
        pareto_df = pd.DataFrame(pareto_data)
        pareto_df.to_csv(self.output_dir / 'fast_pareto_frontier.csv', index=False)
        
        # Save region analysis
        region_analysis = self.analyze_region_performance()
        
        with open(self.output_dir / 'fast_region_analysis.json', 'w') as f:
            json.dump(region_analysis, f, indent=2)
        
        # Save summary report
        with open(self.output_dir / 'fast_multithreaded_summary.txt', 'w') as f:
            f.write("FAST MULTITHREADED GRID SEARCH SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Search Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Simulations: {len(self.results):,}\n")
            f.write(f"Pareto-Optimal Solutions: {len(self.pareto_frontier)}\n")
            f.write(f"Search Duration: {(time.time() - self.start_time)/60:.1f} minutes\n")
            f.write(f"Worker Threads: {self.max_workers}\n")
            f.write(f"Simulation Rate: {len(self.results)/(time.time() - self.start_time):.1f} simulations/sec\n\n")
            
            f.write("REGION PERFORMANCE:\n")
            for region, stats in region_analysis.items():
                f.write(f"  {region}:\n")
                f.write(f"    Simulations: {stats['count']}\n")
                f.write(f"    Avg Satisfaction: {stats['satisfaction_mean']:.3f}\n")
                f.write(f"    Max Satisfaction: {stats['satisfaction_max']:.3f}\n")
                f.write(f"    Avg Fairness: {stats['fairness_mean']:.3f}\n")
                f.write(f"    Max Fairness: {stats['fairness_max']:.3f}\n")
                f.write(f"    Pareto Solutions: {stats['pareto_count']}\n\n")
            
            if self.pareto_frontier:
                f.write("TOP 10 PARETO-OPTIMAL SOLUTIONS:\n")
                sorted_pareto = sorted(self.pareto_frontier, key=lambda x: x.satisfaction_score + x.fairness_score, reverse=True)
                for i, solution in enumerate(sorted_pareto[:10]):
                    f.write(f"  #{i+1}: {solution.cost_combination} ({solution.search_region})\n")
                    f.write(f"    Satisfaction: {solution.satisfaction_score:.3f}\n")
                    f.write(f"    Fairness: {solution.fairness_score:.3f}\n")
                    f.write(f"    Total Cost: {solution.total_cost:.0f}\n\n")

def main():
    parser = argparse.ArgumentParser(description='Fast Multithreaded Grid Search for Multi-Objective Optimization')
    parser.add_argument('--students', type=str, default='data/input/students.csv', help='Students CSV file')
    parser.add_argument('--capacities', type=str, default='data/input/capacities.csv', help='Capacities CSV file')
    parser.add_argument('--output', type=str, default='fast_multithreaded_search', help='Output directory')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker threads (default: auto)')
    
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
    
    # Set output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Initialize fast search
        fast_search = FastMultithreadedSearch(output_dir, max_workers=args.workers)
        
        # Generate fast combinations
        combinations = fast_search.generate_fast_combinations()
        
        # Run fast multithreaded search
        results = fast_search.run_fast_multithreaded_search(combinations, str(students_path), str(capacities_path))
        
        if not results:
            print("❌ No successful simulations")
            return
        
        # Find Pareto frontier
        pareto_solutions = fast_search.find_pareto_frontier()
        
        # Analyze region performance
        region_analysis = fast_search.analyze_region_performance()
        
        # Create visualizations
        fast_search.create_fast_visualizations()
        
        # Save results
        fast_search.save_fast_results()
        
        print(f"\n✅ Fast multithreaded search complete!")
        print(f"📊 Results saved to: {output_dir}")
        print(f"📈 Visualizations: fast_multithreaded_search_results.png")
        print(f"📄 Data files: fast_multithreaded_results.csv, fast_pareto_frontier.csv")
        print(f"📋 Summary: fast_multithreaded_summary.txt")
        
        # Print key insights
        print(f"\n🔍 KEY INSIGHTS:")
        print(f"   • {len(results):,} total simulations completed")
        print(f"   • {len(pareto_solutions)} Pareto-optimal solutions found")
        
        if region_analysis:
            best_region = max(region_analysis.keys(), key=lambda r: region_analysis[r]['pareto_count'])
            print(f"   • Best performing region: {best_region} ({region_analysis[best_region]['pareto_count']} Pareto solutions)")
        
        if pareto_solutions:
            best_solution = max(pareto_solutions, key=lambda x: x.satisfaction_score + x.fairness_score)
            print(f"   • Best combined solution: {best_solution.cost_combination} ({best_solution.search_region})")
            print(f"     - Satisfaction: {best_solution.satisfaction_score:.3f}")
            print(f"     - Fairness: {best_solution.fairness_score:.3f}")
        
    except Exception as e:
        print(f"❌ Error during fast search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
