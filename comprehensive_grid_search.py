#!/usr/bin/env python3
"""
Comprehensive Grid Search for Multi-Objective Optimization

Phase 1: Systematic Grid Search (50,000 simulations)
Purpose: Map the entire cost space systematically with fine-grained resolution

This script implements a comprehensive grid search to explore all possible
cost combinations and identify the complete Pareto frontier.
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
warnings.filterwarnings('ignore')

# Import our existing modules
import sys
sys.path.append('.')
from test_cost_combinations import run_allocation_with_costs, calculate_gini_coefficient
from allocator.data_repository import DataRepository
from allocator.allocation_model_ilp import AllocationConfig as LegacyAllocationConfig
from allocator.preference_model import PreferenceModelConfig

@dataclass
class GridSearchResult:
    """Container for grid search results."""
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

class ComprehensiveGridSearch:
    """Implements comprehensive grid search for cost optimization."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.results = []
        self.pareto_frontier = []
        self.simulation_count = 0
        self.start_time = time.time()
        
    def generate_grid_combinations(self) -> List[Tuple[int, int, int, int, int]]:
        """Generate systematic grid of cost combinations."""
        print("🔧 Generating systematic grid combinations...")
        
        # Fine-grained grid search parameters
        rank1_range = list(range(0, 51, 2))    # 0, 2, 4, ..., 50 (26 values)
        rank2_range = list(range(0, 51, 2))    # 0, 2, 4, ..., 50 (26 values)
        rank3_range = list(range(0, 201, 10))  # 0, 10, 20, ..., 200 (21 values)
        rank4_range = list(range(0, 201, 10))  # 0, 10, 20, ..., 200 (21 values)
        rank5_range = list(range(0, 201, 30))  # 0, 10, 20, ..., 200 (21 values)
        
        # Generate all combinations
        all_combinations = list(itertools.product(rank1_range, rank2_range, rank3_range, rank4_range, rank5_range))
        
        # Filter to ensure monotonic costs: rank1 <= rank2 <= rank3 <= rank4 <= rank5
        # This ensures that worse preferences have higher (or equal) costs
        combinations = [
            combo for combo in all_combinations
            if combo[0] <= combo[1] <= combo[2] <= combo[3] <= combo[4]
        ]
        
        print(f"📊 Generated {len(all_combinations):,} total combinations")
        print(f"   Filtered to {len(combinations):,} monotonic combinations (rank1 ≤ rank2 ≤ rank3 ≤ rank4 ≤ rank5)")
        print(f"   • Rank1: {len(rank1_range)} values ({min(rank1_range)}-{max(rank1_range)})")
        print(f"   • Rank2: {len(rank2_range)} values ({min(rank2_range)}-{max(rank2_range)})")
        print(f"   • Rank3: {len(rank3_range)} values ({min(rank3_range)}-{max(rank3_range)})")
        print(f"   • Rank4: {len(rank4_range)} values ({min(rank4_range)}-{max(rank4_range)})")
        print(f"   • Rank5: {len(rank5_range)} values ({min(rank5_range)}-{max(rank5_range)})")
        
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
    
    def run_grid_search(self, combinations: List[Tuple[int, int, int, int, int]], 
                       students_path: str, capacities_path: str) -> List[GridSearchResult]:
        """Run comprehensive grid search."""
        print(f"🚀 Starting comprehensive grid search with {len(combinations):,} combinations...")
        
        results = []
        successful_simulations = 0
        failed_simulations = 0
        
        for i, cost_combo in enumerate(combinations):
            self.simulation_count += 1
            
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
                    grid_result = GridSearchResult(
                        cost_combination=cost_combo,
                        satisfaction_score=self._calculate_satisfaction_score(result['metrics'].get('preference_satisfaction', {})),
                        fairness_score=1.0 - result['metrics'].get('gini_coefficient', 0.5),
                        total_cost=result['metrics'].get('total_cost', 0),
                        preference_satisfaction=result['metrics'].get('preference_satisfaction', {}),
                        gini_coefficient=result['metrics'].get('gini_coefficient', 0.5),
                        num_students=result['metrics'].get('num_students', 0),
                        algorithm="ilp",
                        timestamp=result['timestamp'],
                        simulation_id=self.simulation_count
                    )
                    results.append(grid_result)
                    successful_simulations += 1
                else:
                    failed_simulations += 1
                
            except Exception as e:
                print(f"  Error with combination {cost_combo}: {e}")
                failed_simulations += 1
                continue
            
            # Progress reporting
            if (i + 1) % 1000 == 0:
                elapsed_time = time.time() - self.start_time
                rate = (i + 1) / elapsed_time
                eta = (len(combinations) - i - 1) / rate if rate > 0 else 0
                
                print(f"  Progress: {i + 1:,}/{len(combinations):,} ({((i + 1)/len(combinations)*100):.1f}%)")
                print(f"  Rate: {rate:.1f} simulations/sec")
                print(f"  ETA: {eta/60:.1f} minutes")
                print(f"  Success: {successful_simulations:,}, Failed: {failed_simulations:,}")
                print()
        
        self.results = results
        print(f"✅ Grid search complete!")
        print(f"   • Total simulations: {self.simulation_count:,}")
        print(f"   • Successful: {successful_simulations:,}")
        print(f"   • Failed: {failed_simulations:,}")
        print(f"   • Success rate: {(successful_simulations/self.simulation_count*100):.1f}%")
        
        return results
    
    def find_pareto_frontier(self) -> List[GridSearchResult]:
        """Find Pareto-optimal solutions from grid search results."""
        print("🔍 Finding Pareto frontier from grid search results...")
        
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
                # For maximization: result_j dominates result_i if:
                # - result_j.satisfaction >= result_i.satisfaction AND
                # - result_j.fairness >= result_i.fairness AND
                # - At least one is strictly better
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
        
        return pareto_solutions
    
    def analyze_optimization_landscape(self) -> Dict[str, Any]:
        """Analyze the complete optimization landscape."""
        print("📊 Analyzing optimization landscape...")
        
        if not self.results:
            return {}
        
        # Extract data
        satisfaction_scores = [r.satisfaction_score for r in self.results]
        fairness_scores = [r.fairness_score for r in self.results]
        total_costs = [r.total_cost for r in self.results]
        
        # Calculate statistics
        landscape_analysis = {
            'total_simulations': len(self.results),
            'pareto_solutions': len(self.pareto_frontier),
            'satisfaction_stats': {
                'min': min(satisfaction_scores),
                'max': max(satisfaction_scores),
                'mean': np.mean(satisfaction_scores),
                'std': np.std(satisfaction_scores),
                'median': np.median(satisfaction_scores)
            },
            'fairness_stats': {
                'min': min(fairness_scores),
                'max': max(fairness_scores),
                'mean': np.mean(fairness_scores),
                'std': np.std(fairness_scores),
                'median': np.median(fairness_scores)
            },
            'cost_stats': {
                'min': min(total_costs),
                'max': max(total_costs),
                'mean': np.mean(total_costs),
                'std': np.std(total_costs),
                'median': np.median(total_costs)
            },
            'correlation': {
                'satisfaction_fairness': np.corrcoef(satisfaction_scores, fairness_scores)[0, 1],
                'satisfaction_cost': np.corrcoef(satisfaction_scores, total_costs)[0, 1],
                'fairness_cost': np.corrcoef(fairness_scores, total_costs)[0, 1]
            }
        }
        
        return landscape_analysis
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations of the optimization landscape."""
        print("📈 Creating comprehensive visualizations...")
        
        if not self.results:
            print("❌ No results to visualize")
            return
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive figure
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Comprehensive Grid Search: Complete Optimization Landscape', fontsize=18, fontweight='bold')
        
        # Extract data
        all_results = self.results
        pareto_results = self.pareto_frontier
        
        satisfaction_scores = [r.satisfaction_score for r in all_results]
        fairness_scores = [r.fairness_score for r in all_results]
        total_costs = [r.total_cost for r in all_results]
        pareto_satisfaction = [r.satisfaction_score for r in pareto_results]
        pareto_fairness = [r.fairness_score for r in pareto_results]
        
        # Plot 1: Complete Pareto Frontier
        axes[0, 0].scatter(satisfaction_scores, fairness_scores, alpha=0.3, s=10, label='All Solutions', color='lightblue')
        axes[0, 0].scatter(pareto_satisfaction, pareto_fairness, alpha=0.8, s=50, c='red', label='Pareto Frontier', edgecolors='black')
        axes[0, 0].set_title('Complete Pareto Frontier')
        axes[0, 0].set_xlabel('Satisfaction Score')
        axes[0, 0].set_ylabel('Fairness Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Satisfaction Distribution
        axes[0, 1].hist(satisfaction_scores, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Satisfaction Score Distribution')
        axes[0, 1].set_xlabel('Satisfaction Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Fairness Distribution
        axes[0, 2].hist(fairness_scores, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Fairness Score Distribution')
        axes[0, 2].set_xlabel('Fairness Score')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Cost vs Satisfaction
        axes[1, 0].scatter(total_costs, satisfaction_scores, alpha=0.5, s=10)
        axes[1, 0].set_title('Total Cost vs Satisfaction')
        axes[1, 0].set_xlabel('Total Cost')
        axes[1, 0].set_ylabel('Satisfaction Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Cost vs Fairness
        axes[1, 1].scatter(total_costs, fairness_scores, alpha=0.5, s=10)
        axes[1, 1].set_title('Total Cost vs Fairness')
        axes[1, 1].set_xlabel('Total Cost')
        axes[1, 1].set_ylabel('Fairness Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Cost Distribution
        axes[1, 2].hist(total_costs, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 2].set_title('Total Cost Distribution')
        axes[1, 2].set_xlabel('Total Cost')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Plot 7: Rank1 Cost Analysis
        rank1_costs = [r.cost_combination[0] for r in all_results]
        axes[2, 0].scatter(rank1_costs, satisfaction_scores, alpha=0.5, s=10)
        axes[2, 0].set_title('Rank1 Cost vs Satisfaction')
        axes[2, 0].set_xlabel('Rank1 Cost')
        axes[2, 0].set_ylabel('Satisfaction Score')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 8: Rank2 Cost Analysis
        rank2_costs = [r.cost_combination[1] for r in all_results]
        axes[2, 1].scatter(rank2_costs, fairness_scores, alpha=0.5, s=10)
        axes[2, 1].set_title('Rank2 Cost vs Fairness')
        axes[2, 1].set_xlabel('Rank2 Cost')
        axes[2, 1].set_ylabel('Fairness Score')
        axes[2, 1].grid(True, alpha=0.3)
        
        # Plot 9: Rank3 Cost Analysis
        rank3_costs = [r.cost_combination[2] for r in all_results]
        axes[2, 2].scatter(rank3_costs, satisfaction_scores, alpha=0.5, s=10)
        axes[2, 2].set_title('Rank3 Cost vs Satisfaction')
        axes[2, 2].set_xlabel('Rank3 Cost')
        axes[2, 2].set_ylabel('Satisfaction Score')
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_grid_search_landscape.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed Pareto frontier plot
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(satisfaction_scores, fairness_scores, alpha=0.2, s=5, label='All Solutions', color='lightblue')
        ax.scatter(pareto_satisfaction, pareto_fairness, alpha=0.9, s=100, c='red', label='Pareto Frontier', edgecolors='black', linewidth=2)
        
        # Add labels for top Pareto solutions
        if pareto_results:
            # Sort Pareto solutions by combined score
            sorted_pareto = sorted(pareto_results, key=lambda x: x.satisfaction_score + x.fairness_score, reverse=True)
            top_solutions = sorted_pareto[:10]
            
            for i, solution in enumerate(top_solutions):
                ax.annotate(f'#{i+1}', 
                           (solution.satisfaction_score, solution.fairness_score),
                           xytext=(8, 8), textcoords='offset points',
                           fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax.set_title(f'Comprehensive Grid Search: Pareto Frontier\n({len(pareto_results)} Pareto-optimal solutions from {len(all_results):,} simulations)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Satisfaction Score', fontsize=14)
        ax.set_ylabel('Fairness Score', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_pareto_frontier.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_comprehensive_results(self):
        """Save comprehensive grid search results."""
        print("💾 Saving comprehensive results...")
        
        self.output_dir.mkdir(exist_ok=True)
        
        # Save all results
        all_data = []
        for result in self.results:
            all_data.append({
                'simulation_id': result.simulation_id,
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
        all_df.to_csv(self.output_dir / 'comprehensive_grid_search_results.csv', index=False)
        
        # Save Pareto frontier
        pareto_data = []
        for result in self.pareto_frontier:
            pareto_data.append({
                'simulation_id': result.simulation_id,
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
        pareto_df.to_csv(self.output_dir / 'comprehensive_pareto_frontier.csv', index=False)
        
        # Save landscape analysis
        landscape_analysis = self.analyze_optimization_landscape()
        
        with open(self.output_dir / 'comprehensive_landscape_analysis.json', 'w') as f:
            json.dump(landscape_analysis, f, indent=2)
        
        # Save summary report
        with open(self.output_dir / 'comprehensive_grid_search_summary.txt', 'w') as f:
            f.write("COMPREHENSIVE GRID SEARCH SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Search Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Simulations: {len(self.results):,}\n")
            f.write(f"Pareto-Optimal Solutions: {len(self.pareto_frontier)}\n")
            f.write(f"Search Duration: {(time.time() - self.start_time)/60:.1f} minutes\n\n")
            
            if landscape_analysis:
                f.write("OPTIMIZATION LANDSCAPE:\n")
                f.write(f"  Satisfaction Range: {landscape_analysis['satisfaction_stats']['min']:.3f} - {landscape_analysis['satisfaction_stats']['max']:.3f}\n")
                f.write(f"  Fairness Range: {landscape_analysis['fairness_stats']['min']:.3f} - {landscape_analysis['fairness_stats']['max']:.3f}\n")
                f.write(f"  Cost Range: {landscape_analysis['cost_stats']['min']:.0f} - {landscape_analysis['cost_stats']['max']:.0f}\n\n")
                
                f.write("CORRELATIONS:\n")
                f.write(f"  Satisfaction-Fairness: {landscape_analysis['correlation']['satisfaction_fairness']:.3f}\n")
                f.write(f"  Satisfaction-Cost: {landscape_analysis['correlation']['satisfaction_cost']:.3f}\n")
                f.write(f"  Fairness-Cost: {landscape_analysis['correlation']['fairness_cost']:.3f}\n\n")
            
            if self.pareto_frontier:
                f.write("TOP 10 PARETO-OPTIMAL SOLUTIONS:\n")
                sorted_pareto = sorted(self.pareto_frontier, key=lambda x: x.satisfaction_score + x.fairness_score, reverse=True)
                for i, solution in enumerate(sorted_pareto[:10]):
                    f.write(f"  #{i+1}: {solution.cost_combination}\n")
                    f.write(f"    Satisfaction: {solution.satisfaction_score:.3f}\n")
                    f.write(f"    Fairness: {solution.fairness_score:.3f}\n")
                    f.write(f"    Total Cost: {solution.total_cost:.0f}\n\n")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Grid Search for Multi-Objective Optimization')
    parser.add_argument('--students', type=str, default='data/input/students.csv', help='Students CSV file')
    parser.add_argument('--capacities', type=str, default='data/input/capacities.csv', help='Capacities CSV file')
    parser.add_argument('--output', type=str, default='comprehensive_grid_search', help='Output directory')
    parser.add_argument('--max-combinations', type=int, default=None, help='Maximum number of combinations to test (for testing)')
    
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
        # Initialize grid search
        grid_search = ComprehensiveGridSearch(output_dir)
        
        # Generate grid combinations
        combinations = grid_search.generate_grid_combinations()
        
        # Limit combinations for testing if specified
        if args.max_combinations:
            combinations = combinations[:args.max_combinations]
            print(f"🔧 Limited to {len(combinations):,} combinations for testing")
        
        # Run grid search
        results = grid_search.run_grid_search(combinations, str(students_path), str(capacities_path))
        
        if not results:
            print("❌ No successful simulations")
            return
        
        # Find Pareto frontier
        pareto_solutions = grid_search.find_pareto_frontier()
        
        # Analyze landscape
        landscape_analysis = grid_search.analyze_optimization_landscape()
        
        # Create visualizations
        grid_search.create_comprehensive_visualizations()
        
        # Save results
        grid_search.save_comprehensive_results()
        
        print(f"\n✅ Comprehensive grid search complete!")
        print(f"📊 Results saved to: {output_dir}")
        print(f"📈 Visualizations: comprehensive_grid_search_landscape.png, comprehensive_pareto_frontier.png")
        print(f"📄 Data files: comprehensive_grid_search_results.csv, comprehensive_pareto_frontier.csv")
        print(f"📋 Summary: comprehensive_grid_search_summary.txt")
        
        # Print key insights
        print(f"\n🔍 KEY INSIGHTS:")
        print(f"   • {len(results):,} total simulations completed")
        print(f"   • {len(pareto_solutions)} Pareto-optimal solutions found")
        if landscape_analysis:
            print(f"   • Satisfaction range: {landscape_analysis['satisfaction_stats']['min']:.3f} - {landscape_analysis['satisfaction_stats']['max']:.3f}")
            print(f"   • Fairness range: {landscape_analysis['fairness_stats']['min']:.3f} - {landscape_analysis['fairness_stats']['max']:.3f}")
            print(f"   • Satisfaction-Fairness correlation: {landscape_analysis['correlation']['satisfaction_fairness']:.3f}")
        
        if pareto_solutions:
            best_solution = max(pareto_solutions, key=lambda x: x.satisfaction_score + x.fairness_score)
            print(f"   • Best combined solution: {best_solution.cost_combination}")
            print(f"     - Satisfaction: {best_solution.satisfaction_score:.3f}")
            print(f"     - Fairness: {best_solution.fairness_score:.3f}")
        
    except Exception as e:
        print(f"❌ Error during grid search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
