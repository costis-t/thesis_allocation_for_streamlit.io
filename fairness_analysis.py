#!/usr/bin/env python3
"""
Analyze fairness of thesis allocations.
Compare fairness metrics (variance, Gini coefficient, etc.) across solutions.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter, defaultdict
from typing import List, Dict
import statistics


def load_allocation(path: str) -> List[Dict]:
    """Load allocation CSV."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_satisfaction_costs(rows: List[Dict]) -> List[int]:
    """Extract effective costs from allocation (proxy for satisfaction)."""
    return [int(row["effective_cost"]) for row in rows]


def compute_preference_ranks(rows: List[Dict]) -> List[int]:
    """Extract preference ranks from allocation."""
    return [int(row["preference_rank"]) for row in rows]


def gini_coefficient(values: List[int]) -> float:
    """
    Compute Gini coefficient (0=perfectly equal, 1=perfectly unequal).
    Lower is fairer.
    """
    if len(values) < 2:
        return 0.0
    sorted_vals = sorted(values)
    n = len(values)
    cumsum = sum(i * v for i, v in enumerate(sorted_vals, 1))
    return (2 * cumsum) / (n * sum(values)) - (n + 1) / n


def compute_fairness_metrics(rows: List[Dict]) -> Dict:
    """Compute fairness metrics for an allocation."""
    costs = compute_satisfaction_costs(rows)
    ranks = compute_preference_ranks(rows)
    
    # Count satisfaction tiers
    tier_counts = Counter(ranks)
    
    # Tier descriptions
    tier_names = {0: "Tier1", 1: "Tier2", 2: "Tier3", 10: "1st choice", 
                  11: "2nd choice", 12: "3rd choice", 13: "4th choice", 
                  14: "5th choice", 999: "Unranked"}
    
    return {
        "total_students": len(rows),
        "avg_cost": statistics.mean(costs),
        "stdev_cost": statistics.stdev(costs) if len(costs) > 1 else 0,
        "min_cost": min(costs),
        "max_cost": max(costs),
        "gini_cost": gini_coefficient(costs),
        
        "avg_rank": statistics.mean(ranks),
        "stdev_rank": statistics.stdev(ranks) if len(ranks) > 1 else 0,
        "gini_rank": gini_coefficient(ranks),
        
        "tier_distribution": {tier_names.get(k, str(k)): v for k, v in tier_counts.items()},
        "num_satisfied_top3": sum(tier_counts.get(r, 0) for r in [0, 1, 2, 10, 11, 12]),
        "num_unranked": tier_counts.get(999, 0),
    }


def main():
    p = argparse.ArgumentParser(description="Analyze fairness of thesis allocations")
    p.add_argument("allocations", nargs="+", help="Paths to allocation.csv files to compare")
    p.add_argument("--report", default="fairness_report.txt", help="Path to write report")
    args = p.parse_args()
    
    results = []
    for path in args.allocations:
        rows = load_allocation(path)
        metrics = compute_fairness_metrics(rows)
        metrics["file"] = path
        results.append(metrics)
    
    # Write report
    with open(args.report, "w") as f:
        f.write("=" * 90 + "\n")
        f.write("FAIRNESS ANALYSIS REPORT\n")
        f.write("=" * 90 + "\n\n")
        
        for i, res in enumerate(results):
            f.write(f"Solution {i+1}: {res['file']}\n")
            f.write("-" * 90 + "\n")
            
            f.write(f"  Total students: {res['total_students']}\n\n")
            
            f.write("  SATISFACTION FAIRNESS (lower is fairer):\n")
            f.write(f"    Gini coefficient (costs):  {res['gini_cost']:.4f} (0=equal, 1=unequal)\n")
            f.write(f"    Avg cost:                  {res['avg_cost']:.1f}\n")
            f.write(f"    Stdev cost:                {res['stdev_cost']:.1f}\n")
            f.write(f"    Cost range:                [{res['min_cost']}, {res['max_cost']}]\n")
            
            f.write("\n  PREFERENCE RANK FAIRNESS:\n")
            f.write(f"    Gini coefficient (ranks): {res['gini_rank']:.4f}\n")
            f.write(f"    Avg rank:                 {res['avg_rank']:.2f}\n")
            f.write(f"    Stdev rank:               {res['stdev_rank']:.2f}\n")
            
            f.write("\n  SATISFACTION DISTRIBUTION:\n")
            for tier_name, count in sorted(res["tier_distribution"].items()):
                pct = 100 * count / res['total_students']
                f.write(f"    {tier_name:20} : {count:3} ({pct:5.1f}%)\n")
            
            f.write(f"\n  SUMMARY:\n")
            f.write(f"    ✓ Top 3 satisfied: {res['num_satisfied_top3']}/{res['total_students']} ({100*res['num_satisfied_top3']/res['total_students']:.1f}%)\n")
            f.write(f"    ⚠ Unranked:        {res['num_unranked']}/{res['total_students']} ({100*res['num_unranked']/res['total_students']:.1f}%)\n")
            
            f.write("\n")
        
        # Comparison
        if len(results) > 1:
            f.write("=" * 90 + "\n")
            f.write("COMPARISON\n")
            f.write("=" * 90 + "\n\n")
            
            # Find best on different metrics
            best_gini_idx = min(range(len(results)), key=lambda i: results[i]['gini_cost'])
            best_avg_rank_idx = min(range(len(results)), key=lambda i: results[i]['avg_rank'])
            best_top3_idx = max(range(len(results)), key=lambda i: results[i]['num_satisfied_top3'])
            
            f.write(f"✓ Best fairness (lowest Gini):     Solution {best_gini_idx + 1}\n")
            f.write(f"✓ Best satisfaction (lowest rank): Solution {best_avg_rank_idx + 1}\n")
            f.write(f"✓ Best coverage (most top-3):      Solution {best_top3_idx + 1}\n")
    
    print(f"✓ Report written to: {args.report}")


if __name__ == "__main__":
    main()
