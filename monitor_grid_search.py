#!/usr/bin/env python3
"""
Monitor comprehensive grid search progress
"""

import time
import os
from pathlib import Path

def monitor_progress(output_dir: str):
    """Monitor the progress of the comprehensive grid search."""
    output_path = Path(output_dir)
    
    print(f"🔍 Monitoring comprehensive grid search progress...")
    print(f"📁 Output directory: {output_path}")
    
    # Check if results file exists
    results_file = output_path / "comprehensive_grid_search_results.csv"
    
    if results_file.exists():
        # Read current results
        import pandas as pd
        df = pd.read_csv(results_file)
        print(f"✅ Current progress: {len(df):,} simulations completed")
        
        # Check for Pareto solutions
        pareto_file = output_path / "comprehensive_pareto_frontier.csv"
        if pareto_file.exists():
            pareto_df = pd.read_csv(pareto_file)
            print(f"🎯 Pareto-optimal solutions found: {len(pareto_df)}")
            
            if len(pareto_df) > 0:
                print(f"📊 Best solution: {pareto_df.iloc[0]['rank1_cost']}, {pareto_df.iloc[0]['rank2_cost']}, {pareto_df.iloc[0]['rank3_cost']}, {pareto_df.iloc[0]['rank4_cost']}, {pareto_df.iloc[0]['rank5_cost']}")
                print(f"   Satisfaction: {pareto_df.iloc[0]['satisfaction_score']:.3f}")
                print(f"   Fairness: {pareto_df.iloc[0]['fairness_score']:.3f}")
        
        # Show recent results
        print(f"\n📈 Recent results (last 5):")
        for _, row in df.tail(5).iterrows():
            print(f"   {row['rank1_cost']}, {row['rank2_cost']}, {row['rank3_cost']}, {row['rank4_cost']}, {row['rank5_cost']} -> Satisfaction: {row['satisfaction_score']:.3f}, Fairness: {row['fairness_score']:.3f}")
    else:
        print("⏳ No results file found yet. Search may still be starting...")
    
    # Check for summary file
    summary_file = output_path / "comprehensive_grid_search_summary.txt"
    if summary_file.exists():
        print(f"\n📋 Summary available: {summary_file}")
        with open(summary_file, 'r') as f:
            content = f.read()
            print("Latest summary:")
            print(content[-500:])  # Last 500 characters

if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "comprehensive_grid_search_full"
    monitor_progress(output_dir)
