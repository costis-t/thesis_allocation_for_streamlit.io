#!/usr/bin/env python3
"""
Main entry point for thesis allocation solver.
Supports multiple algorithms and comprehensive configuration.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from allocator.data_repository import DataRepository
from allocator.preference_model import PreferenceModel
from allocator.allocation_model_ilp import AllocationModelILP, AllocationConfig as LegacyAllocationConfig
from allocator.allocation_model_flow import AllocationModelFlow
from allocator.config import AllocationConfig, PreferenceConfig, CapacityConfig, SolverConfig
from allocator.validation import InputValidator
from allocator.outputs import write_allocation_csv, write_summary_txt
from allocator.viz_sankey import build_sankey
from allocator.logging_config import setup_logging, get_logger


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Thesis Topic Allocator - Professional ILP/Flow Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python allocate.py --students data/students.csv --capacities data/capacities.csv \\
    --out output/allocation.csv --summary output/summary.txt
  
  # With config file
  python allocate.py --config config.json --students data/students.csv \\
    --capacities data/capacities.csv --out output/allocation.csv --summary output/summary.txt
  
  # With visualization
  python allocate.py --students data/students.csv --capacities data/capacities.csv \\
    --out output/allocation.csv --summary output/summary.txt --sankey output/sankey.html
  
  # Export current config
  python allocate.py --save-config my_config.json
        """
    )
    
    # Input files
    p.add_argument("--students", help="Path to students.csv")
    p.add_argument("--capacities", help="Path to capacities.csv")
    p.add_argument("--overrides", help="Path to overrides.csv (optional)")
    
    # Output files
    p.add_argument("--out", help="Path to write allocation.csv")
    p.add_argument("--summary", help="Path to write summary.txt")
    
    # Configuration
    p.add_argument("--config", help="Load configuration from JSON file")
    p.add_argument("--save-config", help="Save default configuration to JSON file and exit")
    
    # Preference model parameters
    p.add_argument("--allow-unranked", default="true", choices=["true", "false"],
                  help="Allow students to be assigned to unranked topics")
    p.add_argument("--tier2-cost", type=int, default=None, help="Cost for tier2 preferences")
    p.add_argument("--tier3-cost", type=int, default=None, help="Cost for tier3 preferences")
    p.add_argument("--unranked-cost", type=int, default=None, help="Cost for unranked topics")
    p.add_argument("--top2-bias", default="true", choices=["true", "false"],
                  help="Apply bias towards top 2 preferences")
    
    # Capacity constraints
    p.add_argument("--dept-min-mode", choices=["soft", "hard"], default=None,
                  help="Department minimum satisfaction mode")
    p.add_argument("--enable-topic-overflow", default="true", choices=["true", "false"],
                  help="Allow topics to exceed capacity")
    p.add_argument("--enable-coach-overflow", default="true", choices=["true", "false"],
                  help="Allow coaches to exceed capacity")
    p.add_argument("--P-dept-shortfall", type=int, default=None,
                  help="Penalty for department minimum shortfall")
    p.add_argument("--P-topic", type=int, default=None,
                  help="Penalty for topic overflow")
    p.add_argument("--P-coach", type=int, default=None,
                  help="Penalty for coach overflow")
    
    # Solver parameters
    p.add_argument("--algorithm", choices=["ilp", "flow", "hybrid"], default=None,
                  help="Solver algorithm")
    p.add_argument("--time-limit-sec", type=int, default=None,
                  help="Solver time limit in seconds")
    p.add_argument("--random-seed", type=int, default=None,
                  help="Random seed for reproducibility")
    p.add_argument("--epsilon-suboptimal", type=float, default=None,
                  help="Allow solutions within epsilon of optimal (e.g., 0.05)")
    
    # Visualization
    p.add_argument("--sankey", help="Path to write Sankey visualization (optional)")
    
    # Logging
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                  help="Logging level")
    p.add_argument("--log-file", help="Path to write log file (optional)")
    
    # Validation
    p.add_argument("--validate-only", action="store_true",
                  help="Only validate input data and exit")
    p.add_argument("--no-validate", action="store_true",
                  help="Skip input validation")

    return p.parse_args()


def as_bool(s: str) -> bool:
    """Convert string to boolean."""
    return str(s).strip().lower() == "true"


def merge_configs(base: AllocationConfig, cli_args) -> AllocationConfig:
    """Merge CLI arguments with base configuration."""
    pref = base.preference
    cap = base.capacity
    solver = base.solver
    
    # Update preference config
    if cli_args.allow_unranked is not None:
        pref.allow_unranked = as_bool(cli_args.allow_unranked)
    if cli_args.tier2_cost is not None:
        pref.tier2_cost = cli_args.tier2_cost
    if cli_args.tier3_cost is not None:
        pref.tier3_cost = cli_args.tier3_cost
    if cli_args.unranked_cost is not None:
        pref.unranked_cost = cli_args.unranked_cost
    if cli_args.top2_bias is not None:
        pref.top2_bias = as_bool(cli_args.top2_bias)
    
    # Update capacity config
    if cli_args.dept_min_mode is not None:
        cap.dept_min_mode = cli_args.dept_min_mode
    if cli_args.enable_topic_overflow is not None:
        cap.enable_topic_overflow = as_bool(cli_args.enable_topic_overflow)
    if cli_args.enable_coach_overflow is not None:
        cap.enable_coach_overflow = as_bool(cli_args.enable_coach_overflow)
    if cli_args.P_dept_shortfall is not None:
        cap.P_dept_shortfall = cli_args.P_dept_shortfall
    if cli_args.P_topic is not None:
        cap.P_topic = cli_args.P_topic
    if cli_args.P_coach is not None:
        cap.P_coach = cli_args.P_coach
    
    # Update solver config
    if cli_args.algorithm is not None:
        solver.algorithm = cli_args.algorithm
    if cli_args.time_limit_sec is not None:
        solver.time_limit_sec = cli_args.time_limit_sec
    if cli_args.random_seed is not None:
        solver.random_seed = cli_args.random_seed
    if cli_args.epsilon_suboptimal is not None:
        solver.epsilon_suboptimal = cli_args.epsilon_suboptimal
    
    return AllocationConfig(preference=pref, capacity=cap, solver=solver)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = getattr(__import__('logging'), args.log_level)
    logger = setup_logging(level=log_level, log_file=args.log_file)
    logger.info("Thesis Allocation Solver Started")
    
    # Handle --save-config
    if args.save_config:
        config = AllocationConfig.default()
        config.save_json(args.save_config)
        logger.info(f"Default configuration saved to: {args.save_config}")
        return 0
    
    # Validate required arguments
    required_args = ["students", "capacities", "out", "summary"]
    missing_args = [arg for arg in required_args if not getattr(args, arg)]
    if missing_args:
        logger.error(f"Missing required arguments: {', '.join(missing_args)}")
        print(f"Error: Missing required arguments: {', '.join(missing_args)}", file=sys.stderr)
        return 1
    
    try:
        # Load configuration
        config = AllocationConfig.default()
        if args.config:
            logger.info(f"Loading configuration from: {args.config}")
            config = AllocationConfig.load_json(args.config)
        
        # Merge CLI arguments
        config = merge_configs(config, args)
        
        # Validate configuration
        config.validate()
        logger.info("Configuration validated successfully")
        
        # Load data
        logger.info("Loading input data...")
        repo = DataRepository(args.students, args.capacities, args.overrides)
        repo.load()
        logger.info(f"Loaded {len(repo.students)} students, {len(repo.topics)} topics, "
                   f"{len(repo.coaches)} coaches, {len(repo.departments)} departments")
        
        # Validate input data
        if not args.no_validate:
            logger.info("Validating input data...")
            validator = InputValidator()
            is_valid, validation_results = validator.validate_all(
                repo.students, repo.topics, repo.coaches, repo.departments
            )
            
            logger.info(validator.get_summary())
            for result in validation_results:
                if result.severity == "error":
                    logger.error(str(result))
                else:
                    logger.warning(str(result))
            
            if not is_valid:
                logger.error("Input validation failed. Aborting.")
                return 1
        
        if args.validate_only:
            logger.info("Validation successful. Exiting as requested.")
            return 0
        
        # Build preference model
        logger.info("Building preference model...")
        from allocator.preference_model import PreferenceModelConfig
        pref_model = PreferenceModel(
            topics=repo.topics,
            overrides=repo.overrides,
            cfg=PreferenceModelConfig(
                allow_unranked=config.preference.allow_unranked,
                tier2_cost=config.preference.tier2_cost,
                tier3_cost=config.preference.tier3_cost,
                unranked_cost=config.preference.unranked_cost,
                top2_bias=config.preference.top2_bias,
                rank1_cost=config.preference.rank1_cost,
                rank2_cost=config.preference.rank2_cost,
                rank3_cost=config.preference.rank3_cost,
                rank4_cost=config.preference.rank4_cost,
                rank5_cost=config.preference.rank5_cost
            )
        )
        
        # Build allocation config for legacy models
        legacy_alloc_cfg = LegacyAllocationConfig(
            pref_cfg=PreferenceModelConfig(
                allow_unranked=config.preference.allow_unranked,
                tier2_cost=config.preference.tier2_cost,
                tier3_cost=config.preference.tier3_cost,
                unranked_cost=config.preference.unranked_cost,
                top2_bias=config.preference.top2_bias,
                rank1_cost=config.preference.rank1_cost,
                rank2_cost=config.preference.rank2_cost,
                rank3_cost=config.preference.rank3_cost,
                rank4_cost=config.preference.rank4_cost,
                rank5_cost=config.preference.rank5_cost
            ),
            dept_min_mode=config.capacity.dept_min_mode,
            dept_max_mode=config.capacity.dept_max_mode,
            enable_topic_overflow=config.capacity.enable_topic_overflow,
            enable_coach_overflow=config.capacity.enable_coach_overflow,
            P_dept_shortfall=config.capacity.P_dept_shortfall,
            P_dept_overflow=config.capacity.P_dept_overflow,
            P_topic=config.capacity.P_topic,
            P_coach=config.capacity.P_coach,
            time_limit_sec=config.solver.time_limit_sec,
            random_seed=config.solver.random_seed,
            epsilon_suboptimal=config.solver.epsilon_suboptimal,
            min_acceptable_preference_rank=config.preference.min_acceptable_preference_rank,
            max_acceptable_preference_rank=config.preference.max_acceptable_preference_rank,
            excluded_preference_ranks=config.preference.excluded_preference_ranks
        )
        
        # Build and solve
        logger.info(f"Building allocation model ({config.solver.algorithm})...")
        if config.solver.algorithm == "ilp":
            model = AllocationModelILP(
                students=repo.students,
                topics=repo.topics,
                coaches=repo.coaches,
                departments=repo.departments,
                pref_model=pref_model,
                cfg=legacy_alloc_cfg
            )
        elif config.solver.algorithm == "flow":
            model = AllocationModelFlow(
                students=repo.students,
                topics=repo.topics,
                coaches=repo.coaches,
                departments=repo.departments,
                pref_model=pref_model,
                cfg=legacy_alloc_cfg
            )
        elif config.solver.algorithm == "hybrid":
            # Hybrid: solve with ILP first, then verify with flow
            model = AllocationModelILP(
                students=repo.students,
                topics=repo.topics,
                coaches=repo.coaches,
                departments=repo.departments,
                pref_model=pref_model,
                cfg=legacy_alloc_cfg
            )
        
        model.build()
        logger.info("Solving...")
        rows, diagnostics = model.solve()
        
        # Hybrid: verify ILP solution with flow
        if config.solver.algorithm == "hybrid":
            logger.info("Verifying ILP solution with flow algorithm...")
            flow_model = AllocationModelFlow(
                students=repo.students,
                topics=repo.topics,
                coaches=repo.coaches,
                departments=repo.departments,
                pref_model=pref_model,
                cfg=legacy_alloc_cfg
            )
            flow_model.build()
            flow_rows, flow_diag = flow_model.solve()
            
            # Compare objectives
            ilp_obj = diagnostics.get("objective_value", float('inf'))
            flow_obj = flow_diag.get("objective_value", float('inf'))
            
            if ilp_obj <= flow_obj:
                diagnostics["algorithm"] = "hybrid (ILP better)"
                logger.info(f"ILP solution better: {ilp_obj} vs {flow_obj}")
            else:
                rows, diagnostics = flow_rows, flow_diag
                diagnostics["algorithm"] = "hybrid (flow better)"
                logger.info(f"Flow solution better: {flow_obj} vs {ilp_obj}")
        
        # Write outputs
        logger.info(f"Writing allocation to: {args.out}")
        write_allocation_csv(args.out, rows)
        
        logger.info(f"Writing summary to: {args.summary}")
        write_summary_txt(args.summary, rows, repo.topics, repo.coaches, repo.departments, diagnostics)
        
        # Visualization
        if args.sankey:
            try:
                logger.info(f"Building Sankey visualization...")
                build_sankey(rows, repo.topics, args.sankey, title="Thesis Allocation")
                logger.info(f"Sankey written to: {args.sankey}")
            except Exception as e:
                logger.warning(f"Failed to build Sankey visualization: {e}")
        
        # Final summary
        logger.info(f"Allocation complete: {len(rows)} students assigned")
        if diagnostics.get("unassignable_students"):
            logger.warning(f"{len(diagnostics['unassignable_students'])} students have no admissible topics")
        if diagnostics.get("unassigned_after_solve"):
            logger.warning(f"{len(diagnostics['unassigned_after_solve'])} students remained unassigned")
        
        logger.info("✓ Success")
        return 0
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
