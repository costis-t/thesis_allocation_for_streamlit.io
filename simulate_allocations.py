#!/usr/bin/env python3
"""
Run multiple thesis allocation simulations and compare results.
Shows variation in ranked choice satisfaction, solver stability, and assignment differences.
"""
from __future__ import annotations
import argparse
import subprocess
import json
import csv
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict


def run_allocate(
    students_path: str,
    capacities_path: str,
    overrides_path: str | None,
    output_csv: str,
    summary_txt: str,
    random_seed: int,
    **extra_args
) -> Dict:
    """Run allocate.py with a specific random seed."""
    cmd = [
        "python3", "allocate.py",
        "--students", students_path,
        "--capacities", capacities_path,
        "--out", output_csv,
        "--summary", summary_txt,
        "--random-seed", str(random_seed),
    ]
    if overrides_path:
        cmd.extend(["--overrides", overrides_path])
    
    for key, val in extra_args.items():
        cmd.append(f"--{key.replace('_', '-')}")
        cmd.append(str(val))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Allocate failed: {result.stderr}")
    
    # Parse allocation results
    assignments = {}
    with open(output_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            student = row["student"]
            rank = int(row["preference_rank"])
            assignments[student] = {
                "topic": row["assigned_topic"],
                "rank": rank,
                "cost": int(row["effective_cost"])
            }
    
    return assignments


def extract_satisfaction(output_csv: str) -> Dict[int, int]:
    """Extract ranked choice satisfaction from allocation.csv."""
    satisfaction = defaultdict(int)
    with open(output_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank = int(row["preference_rank"])
            satisfaction[rank] += 1
    return dict(satisfaction)


def compare_simulations(simulations: List[Dict]) -> Dict:
    """Analyze differences across simulation runs."""
    num_runs = len(simulations)
    
    # Extract satisfaction stats
    satisfaction_stats = {
        "1st": [], "2nd": [], "3rd": [], "4th": [], "5th": [], "unranked": []
    }
    rank_map = {10: "1st", 11: "2nd", 12: "3rd", 13: "4th", 14: "5th", 999: "unranked", 0: "tier1", 1: "tier2", 2: "tier3"}
    
    for sim in simulations:
        sat = sim["satisfaction"]
        satisfaction_stats["1st"].append(sat.get(10, 0))
        satisfaction_stats["2nd"].append(sat.get(11, 0))
        satisfaction_stats["3rd"].append(sat.get(12, 0))
        satisfaction_stats["4th"].append(sat.get(13, 0))
        satisfaction_stats["5th"].append(sat.get(14, 0))
        satisfaction_stats["unranked"].append(sat.get(999, 0))
    
    # Count assignment differences across runs
    all_students = set()
    for sim in simulations:
        all_students.update(sim["assignments"].keys())
    
    assignment_differences = defaultdict(list)
    for student in all_students:
        topics = [sim["assignments"].get(student, {}).get("topic") for sim in simulations]
        if len(set(topics)) > 1:  # Not all the same
            assignment_differences[student] = topics
    
    return {
        "num_runs": num_runs,
        "satisfaction_stats": satisfaction_stats,
        "assignment_differences": dict(assignment_differences),
    }


def main():
    p = argparse.ArgumentParser(description="Run multiple thesis allocation simulations")
    p.add_argument("--students", required=True, help="Path to students.csv")
    p.add_argument("--capacities", required=True, help="Path to capacities.csv")
    p.add_argument("--overrides", required=False, help="Path to overrides.csv")
    p.add_argument("--num-runs", type=int, default=5, help="Number of simulation runs")
    p.add_argument("--output-dir", default="simulations", help="Directory for simulation outputs")
    p.add_argument("--report", default="simulation_report.txt", help="Path to write report")
    
    # Pass through allocate args
    p.add_argument("--allow-unranked", default="true")
    p.add_argument("--tier2-cost", type=int, default=1)
    p.add_argument("--tier3-cost", type=int, default=5)
    p.add_argument("--unranked-cost", type=int, default=200)
    p.add_argument("--top2-bias", default="true")
    p.add_argument("--dept-min-mode", default="soft")
    p.add_argument("--P-dept-shortfall", type=int, default=1000)
    p.add_argument("--enable-topic-overflow", default="true")
    p.add_argument("--enable-coach-overflow", default="true")
    p.add_argument("--P-topic", type=int, default=800)
    p.add_argument("--P-coach", type=int, default=600)
    p.add_argument("--time-limit-sec", type=int, default=None)
    
    args = p.parse_args()
    
    # Setup output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    
    print(f"Running {args.num_runs} allocation simulations...")
    
    simulations = []
    for run_num in range(args.num_runs):
        print(f"  Run {run_num + 1}/{args.num_runs} (seed={run_num})...", end=" ", flush=True)
        
        output_csv = str(out_dir / f"allocation_run{run_num}.csv")
        summary_txt = str(out_dir / f"summary_run{run_num}.txt")
        
        extra_args = {
            "allow_unranked": args.allow_unranked,
            "tier2_cost": args.tier2_cost,
            "tier3_cost": args.tier3_cost,
            "unranked_cost": args.unranked_cost,
            "top2_bias": args.top2_bias,
            "dept_min_mode": args.dept_min_mode,
            "P_dept_shortfall": args.P_dept_shortfall,
            "enable_topic_overflow": args.enable_topic_overflow,
            "enable_coach_overflow": args.enable_coach_overflow,
            "P_topic": args.P_topic,
            "P_coach": args.P_coach,
        }
        if args.time_limit_sec:
            extra_args["time_limit_sec"] = args.time_limit_sec
        
        assignments = run_allocate(
            args.students, args.capacities, args.overrides,
            output_csv, summary_txt, random_seed=run_num,
            **extra_args
        )
        satisfaction = extract_satisfaction(output_csv)
        simulations.append({
            "run": run_num,
            "assignments": assignments,
            "satisfaction": satisfaction,
        })
        print("✓")
    
    # Analyze
    print("\nAnalyzing results...")
    analysis = compare_simulations(simulations)
    
    # Write report
    with open(args.report, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("THESIS ALLOCATION SIMULATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Simulations: {analysis['num_runs']} runs with different random seeds\n")
        f.write(f"Configuration: {args.allow_unranked=}, {args.unranked_cost=}, {args.tier2_cost=}, {args.tier3_cost=}\n\n")
        
        # Satisfaction statistics
        f.write("RANKED CHOICE SATISFACTION (across runs):\n")
        f.write("-" * 70 + "\n")
        for rank_name in ["tier1", "tier2", "tier3", "1st", "2nd", "3rd", "4th", "5th", "unranked"]:
            if rank_name not in analysis["satisfaction_stats"]:
                continue
            values = analysis["satisfaction_stats"][rank_name]
            if not values:
                continue
            min_v, max_v, avg_v = min(values), max(values), sum(values) / len(values)
            f.write(f"  {rank_name:12} : avg={avg_v:.1f}  min={min_v}  max={max_v}  " + 
                   f"values={values}\n")
        
        # Assignment stability
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"ASSIGNMENT STABILITY:\n")
        f.write("-" * 70 + "\n")
        if analysis["assignment_differences"]:
            f.write(f"⚠ {len(analysis['assignment_differences'])} student(s) got different assignments across runs:\n")
            for student, topics in sorted(analysis["assignment_differences"].items())[:20]:
                f.write(f"  {student}: {topics}\n")
            if len(analysis["assignment_differences"]) > 20:
                f.write(f"  ... and {len(analysis['assignment_differences']) - 20} more\n")
        else:
            f.write("✓ All students got the SAME assignment in every run (deterministic)\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Individual run details saved to:\n")
        for i in range(args.num_runs):
            f.write(f"  - simulations/allocation_run{i}.csv\n")
            f.write(f"  - simulations/summary_run{i}.txt\n")
    
    print(f"\n✓ Report written to: {args.report}")
    print(f"✓ Detailed results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
