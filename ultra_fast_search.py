#!/usr/bin/env python3
"""
Ultra-Fast Focused Grid Search (10-minute version)

This script implements a highly focused grid search that runs in ~10 minutes
by targeting only the most promising regions with strategic sampling.
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
from tqdm import tqdm
import random
import multiprocessing as mp
warnings.filterwarnings('ignore')

# Import our existing modules
import sys
sys.path.append('.')
from test_cost_combinations import run_allocation_with_costs, calculate_gini_coefficient

@dataclass
class UltraFastResult:
    """Container for ultra-fast search results."""
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

class UltraFastSearch:
    """Implements ultra-fast focused grid search for 10-minute runs."""
    
    def __init__(self, output_dir: Path, max_workers: int = None):
        self.output_dir = Path(output_dir)
        self.results = []
        self.pareto_frontier = []
        self.simulation_count = 0
        self.start_time = time.time()
        self.max_workers = max_workers or min(16, (mp.cpu_count() or 1) + 4)
        self.results_lock = threading.Lock()
        
    def generate_ultra_fast_combinations(self) -> List[Tuple[Tuple[int, int, int, int, int], str]]:
        """Generate ultra-fast combinations targeting ~1000 simulations for 10-minute runs."""
        print("🔧 Generating ultra-fast combinations for 10-minute runs...")
        print("   Note: Only monotonic combinations (rank1 ≤ rank2 ≤ rank3 ≤ rank4 ≤ rank5) will be used")
        
        combinations = []
        
        # Region 1: Low rank1, moderate rank2-3, high rank4-5 (HIGH PRIORITY)
        print("  📍 Region 1: Low rank1, moderate rank2-3, high rank4-5 (HIGH PRIORITY)")
        rank1_range = list(range(0, 21, 5))    # 0, 5, 10, 15, 20 (5 values)
        rank2_range = list(range(40, 81, 10))  # 40, 50, 60, 70, 80 (5 values)
        rank3_range = list(range(60, 121, 15)) # 60, 75, 90, 105, 120 (5 values)
        rank4_range = list(range(80, 121, 10)) # 80, 90, 100, 110, 120 (5 values)
        rank5_range = list(range(80, 121, 10)) # 80, 90, 100, 110, 120 (5 values)
        
        region1_combinations = list(itertools.product(rank1_range, rank2_range, rank3_range, rank4_range, rank5_range))
        # Filter to ensure monotonic costs
        combinations.extend([(combo, "region1_high_priority") for combo in region1_combinations 
                            if combo[0] <= combo[1] <= combo[2] <= combo[3] <= combo[4]])
        
        # Region 2: Very low rank1-2, high rank3-5 (TOP SATISFACTION)
        print("  📍 Region 2: Very low rank1-2, high rank3-5 (TOP SATISFACTION)")
        rank1_range = list(range(0, 11, 2))    # 0, 2, 4, 6, 8, 10 (6 values)
        rank2_range = list(range(0, 11, 2))    # 0, 2, 4, 6, 8, 10 (6 values)
        rank3_range = list(range(80, 201, 24)) # 80, 104, 128, 152, 176, 200 (6 values)
        rank4_range = list(range(80, 201, 24)) # 80, 104, 128, 152, 176, 200 (6 values)
        rank5_range = list(range(80, 201, 24)) # 80, 104, 128, 152, 176, 200 (6 values)
        
        region2_combinations = list(itertools.product(rank1_range, rank2_range, rank3_range, rank4_range, rank5_range))
        # Filter to ensure monotonic costs
        combinations.extend([(combo, "region2_top_satisfaction") for combo in region2_combinations 
                            if combo[0] <= combo[1] <= combo[2] <= combo[3] <= combo[4]])
        
        # Region 3: Balanced moderate costs (BALANCED)
        print("  📍 Region 3: Balanced moderate costs (BALANCED)")
        rank1_range = list(range(10, 31, 5))   # 10, 15, 20, 25, 30 (5 values)
        rank2_range = list(range(30, 51, 5))   # 30, 35, 40, 45, 50 (5 values)
        rank3_range = list(range(50, 101, 12)) # 50, 62, 74, 86, 98 (5 values)
        rank4_range = list(range(70, 121, 12)) # 70, 82, 94, 106, 118 (5 values)
        rank5_range = list(range(70, 121, 12)) # 70, 82, 94, 106, 118 (5 values)
        
        region3_combinations = list(itertools.product(rank1_range, rank2_range, rank3_range, rank4_range, rank5_range))
        # Filter to ensure monotonic costs
        combinations.extend([(combo, "region3_balanced") for combo in region3_combinations 
                            if combo[0] <= combo[1] <= combo[2] <= combo[3] <= combo[4]])
        
        # Region 4: High rank1-2, moderate rank3-5 (FAIRNESS FOCUSED)
        print("  📍 Region 4: High rank1-2, moderate rank3-5 (FAIRNESS FOCUSED)")
        rank1_range = list(range(20, 51, 8))   # 20, 28, 36, 44, 50 (5 values)
        rank2_range = list(range(40, 71, 8))  # 40, 48, 56, 64, 70 (5 values)
        rank3_range = list(range(50, 101, 12)) # 50, 62, 74, 86, 98 (5 values)
        rank4_range = list(range(60, 111, 12)) # 60, 72, 84, 96, 108 (5 values)
        rank5_range = list(range(60, 111, 12)) # 60, 72, 84, 96, 108 (5 values)
        
        region4_combinations = list(itertools.product(rank1_range, rank2_range, rank3_range, rank4_range, rank5_range))
        # Filter to ensure monotonic costs
        combinations.extend([(combo, "region4_fairness") for combo in region4_combinations 
                            if combo[0] <= combo[1] <= combo[2] <= combo[3] <= combo[4]])
        
        # Region 5: Strategic extreme values (BOUNDARY EXPLORATION)
        print("  📍 Region 5: Strategic extreme values (BOUNDARY EXPLORATION)")
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
            # Top-2 bias patterns
            (0, 1, 100, 101, 102),
            (0, 1, 200, 201, 202),
            (0, 1, 50, 51, 52),
            # Linear patterns
            (0, 1, 2, 3, 4),
            (0, 2, 4, 6, 8),
            (0, 5, 10, 15, 20),
        ]
        # Filter to ensure monotonic costs
        combinations.extend([(combo, "region5_extreme") for combo in extreme_combinations 
                            if combo[0] <= combo[1] <= combo[2] <= combo[3] <= combo[4]])
        
        # Region 6: Random sampling for discovery (RANDOM EXPLORATION)
        print("  📍 Region 6: Random sampling for discovery (RANDOM EXPLORATION)")
        random.seed(42)  # For reproducibility
        random_combinations = []
        while len(random_combinations) < 100:  # Generate 100 monotonic combinations
            combo = (
                random.randint(0, 50),
                random.randint(0, 50),
                random.randint(0, 200),
                random.randint(0, 200),
                random.randint(0, 200)
            )
            # Only keep if monotonic
            if combo[0] <= combo[1] <= combo[2] <= combo[3] <= combo[4]:
                random_combinations.append(combo)
        combinations.extend([(combo, "region6_random") for combo in random_combinations])
        
        print(f"📊 Generated {len(combinations):,} ultra-fast combinations")
        print(f"   • Region 1 (High Priority): {len(region1_combinations):,} combinations")
        print(f"   • Region 2 (Top Satisfaction): {len(region2_combinations):,} combinations")
        print(f"   • Region 3 (Balanced): {len(region3_combinations):,} combinations")
        print(f"   • Region 4 (Fairness): {len(region4_combinations):,} combinations")
        print(f"   • Region 5 (Extreme): {len(extreme_combinations)} combinations")
        print(f"   • Region 6 (Random): {len(random_combinations)} combinations")
        print(f"   • Total: {len(combinations):,} combinations (target: ~10 minutes)")
        
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
    
    def _run_single_simulation(self, args) -> Tuple[UltraFastResult, bool]:
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
                search_result = UltraFastResult(
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
    
    def run_ultra_fast_search(self, combinations: List[Tuple[Tuple[int, int, int, int, int], str]], 
                            students_path: str, capacities_path: str) -> List[UltraFastResult]:
        """Run ultra-fast multithreaded search with progress tracking."""
        print(f"🚀 Starting ultra-fast search with {len(combinations):,} combinations...")
        print(f"🔧 Using {self.max_workers} worker threads")
        print(f"⏱️  Target time: ~10 minutes")
        
        # Prepare arguments for multithreading
        simulation_args = []
        for i, (cost_combo, region) in enumerate(combinations):
            simulation_args.append((cost_combo, region, i + 1, students_path, capacities_path))
        
        results = []
        successful_simulations = 0
        failed_simulations = 0
        region_stats = defaultdict(int)
        
        # Create progress bar
        with tqdm(total=len(combinations), desc="⚡ Ultra-fast search", unit="sim", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
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
                        elapsed_minutes = (time.time() - self.start_time) / 60
                        rate = (successful_simulations + failed_simulations) / (time.time() - self.start_time)
                        pbar.set_description(f"⚡ Ultra-fast search (Success: {success_rate:.1f}%, Rate: {rate:.1f}/s)")
        
        self.results = results
        elapsed_time = time.time() - self.start_time
        
        print(f"\n✅ Ultra-fast search complete!")
        print(f"   • Total simulations: {len(combinations):,}")
        print(f"   • Successful: {successful_simulations:,}")
        print(f"   • Failed: {failed_simulations:,}")
        print(f"   • Success rate: {(successful_simulations/len(combinations)*100):.1f}%")
        print(f"   • Total time: {elapsed_time/60:.1f} minutes")
        print(f"   • Rate: {len(combinations)/elapsed_time:.1f} simulations/sec")
        print(f"   • Region breakdown: {dict(region_stats)}")
        
        return results
    
    def find_pareto_frontier(self) -> List[UltraFastResult]:
        """Find Pareto-optimal solutions from ultra-fast search results."""
        print("🔍 Finding Pareto frontier from ultra-fast search results...")
        
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
        
        for region in ['region1_high_priority', 'region2_top_satisfaction', 'region3_balanced', 
                      'region4_fairness', 'region5_extreme', 'region6_random']:
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
    
    def create_ultra_fast_visualizations(self):
        """Create visualizations for ultra-fast search results."""
        print("📈 Creating ultra-fast visualizations...")
        
        if not self.results:
            print("❌ No results to visualize")
            return
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Ultra-Fast Grid Search: Multi-Objective Optimization Results (10-minute run)', fontsize=16, fontweight='bold')
        
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
        region_colors = {
            'region1_high_priority': 'red', 
            'region2_top_satisfaction': 'blue', 
            'region3_balanced': 'green', 
            'region4_fairness': 'orange', 
            'region5_extreme': 'purple',
            'region6_random': 'brown'
        }
        
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
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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
            axes[0, 1].set_xticklabels([r.replace('region', 'R') for r in regions_list], rotation=45)
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
        axes[1, 2].pie(region_counts.values(), labels=[r.replace('region', 'R') for r in region_counts.keys()], 
                     autopct='%1.1f%%')
        axes[1, 2].set_title('Distribution of Simulations by Region')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ultra_fast_search_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_ultra_fast_results(self):
        """Save ultra-fast search results."""
        print("💾 Saving ultra-fast results...")
        
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
        all_df.to_csv(self.output_dir / 'ultra_fast_results.csv', index=False)
        
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
        pareto_df.to_csv(self.output_dir / 'ultra_fast_pareto_frontier.csv', index=False)
        
        # Save region analysis
        region_analysis = self.analyze_region_performance()
        
        with open(self.output_dir / 'ultra_fast_region_analysis.json', 'w') as f:
            json.dump(region_analysis, f, indent=2)
        
        # Save summary report
        with open(self.output_dir / 'ultra_fast_summary.txt', 'w') as f:
            f.write("ULTRA-FAST GRID SEARCH SUMMARY (10-MINUTE RUN)\n")
            f.write("=" * 60 + "\n\n")
            
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
    parser = argparse.ArgumentParser(description='Ultra-Fast Grid Search for Multi-Objective Optimization (10-minute runs)')
    parser.add_argument('--students', type=str, default='data/input/students.csv', help='Students CSV file')
    parser.add_argument('--capacities', type=str, default='data/input/capacities.csv', help='Capacities CSV file')
    parser.add_argument('--output', type=str, default='ultra_fast_search', help='Output directory')
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
        # Initialize ultra-fast search
        ultra_fast_search = UltraFastSearch(output_dir, max_workers=args.workers)
        
        # Generate ultra-fast combinations
        combinations = ultra_fast_search.generate_ultra_fast_combinations()
        
        # Run ultra-fast search
        results = ultra_fast_search.run_ultra_fast_search(combinations, str(students_path), str(capacities_path))
        
        if not results:
            print("❌ No successful simulations")
            return
        
        # Find Pareto frontier
        pareto_solutions = ultra_fast_search.find_pareto_frontier()
        
        # Analyze region performance
        region_analysis = ultra_fast_search.analyze_region_performance()
        
        # Create visualizations
        ultra_fast_search.create_ultra_fast_visualizations()
        
        # Save results
        ultra_fast_search.save_ultra_fast_results()
        
        print(f"\n✅ Ultra-fast search complete!")
        print(f"📊 Results saved to: {output_dir}")
        print(f"📈 Visualizations: ultra_fast_search_results.png")
        print(f"📄 Data files: ultra_fast_results.csv, ultra_fast_pareto_frontier.csv")
        print(f"📋 Summary: ultra_fast_summary.txt")
        
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
        print(f"❌ Error during ultra-fast search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
