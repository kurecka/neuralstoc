import argparse
import os
import sys
import yaml

from typing import Tuple

from neuralstoc.monitor import ExperimentMonitor

import logging
logger = logging.getLogger("neuralstoc")


def interpret_size_arg(cmd) -> int:
    """
    Converts a string with multiplications into an integer with optional unit multipliers.
    
    Parses expressions like "16k", "2*8M", or "4*8*256" into appropriate integer values.
    Supports k/K (kilo), M (mega), and G (giga) unit multipliers.
    
    Args:
        cmd (str): The size string to interpret
        
    Returns:
        int: The calculated size value
        
    Examples:
        >>> interpret_size_arg("16k")
        16384
        >>> interpret_size_arg("2*8M")
        16777216
        >>> interpret_size_arg("4*8*256")
        8192
    """
    parts = cmd.split("*")
    bs = 1
    for p in parts:
        if "k" in p or "K" in p:
            p = p.replace("k", "").replace("K", "")
            bs *= 1024 * int(p)
        elif "M" in p:
            p = p.replace("M", "")
            bs *= 1024 * 1024 * int(p)
        elif "G" in p:
            p = p.replace("G", "")
            bs *= 1024 * 1024 * 1024 * int(p)
        else:
            bs *= int(p)
    return bs


def load_config(config_path) -> dict:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def configure() -> Tuple[argparse.Namespace, ExperimentMonitor]:
    """
        Parse command line arguments, load configuration from a YAML file, configure logging, and set up directories.

        Priority of arguments:
        1. Command line arguments
        2. Configuration file arguments
        3. Default values

        Returns:
            argparse.Namespace: Parsed command line arguments with configuration values
            ExperimentMonitor: Monitor object for experiment tracking
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--no_config", action="store_true")
    parser.add_argument("--config_path", type=str, default="scripts/config.yaml")
    parser.add_argument("--model", default="tmlp")
    parser.add_argument("--env", default="vlds")
    parser.add_argument("--task", default="control")
    parser.add_argument("--spec", default="reach_avoid")
    parser.add_argument("--timeout", default=24 * 60, type=int)  # in minutes
    parser.add_argument("--prob", default=0.3, type=float)
    parser.add_argument("--eps", default=0.001, type=float)
    parser.add_argument("--lip_lambda", default=0.001, type=float)
    parser.add_argument("--p_lip", default=3.0, type=float)
    parser.add_argument("--v_lip", default=10.0, type=float)
    parser.add_argument("--hidden_v", default=128, type=int)
    parser.add_argument("--min_iters", default=0, type=int)
    parser.add_argument("--num_layers_v", default=2, type=int)
    parser.add_argument("--num_layers_p", default=2, type=int)
    parser.add_argument("--batch_size", default="16k")
    parser.add_argument("--learner_batch_size", default="16k")
    parser.add_argument("--ppo_iters", default=100, type=int)
    parser.add_argument("--n_step", default=1, type=int)
    parser.add_argument("--v_act", default="relu")
    parser.add_argument("--norm", default="linf")
    parser.add_argument("--ds_type", default="all")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--skip_initialize", action="store_true")
    parser.add_argument("--only_initialize", action="store_true")
    parser.add_argument("--sac_steps", default="1200000000")
    parser.add_argument("--continue_rsm", type=int, default=0)
    parser.add_argument("--rsm_path", type=str, default=None)
    parser.add_argument("--no_train", action="store_true")
    parser.add_argument("--train_p", type=int, default=3)
    parser.add_argument("--soft_constraint", action="store_true")
    parser.add_argument("--normalize_r", type=int, default=0)
    parser.add_argument("--normalize_a", type=int, default=1)
    parser.add_argument("--grid_size", default="16M")
    parser.add_argument("--std_start", default=1.0, type=float)
    parser.add_argument("--std_end", default=0.05, type=float)
    parser.add_argument("--p_lr", default=0.00005, type=float)
    parser.add_argument("--c_lr", default=0.0005, type=float)
    parser.add_argument("--v_lr", default=0.0005, type=float)
    parser.add_argument("--n_local", default=10, type=int)
    parser.add_argument("--load_from_brax", action="store_true")
    parser.add_argument("--initialize", default="ppo")
    parser.add_argument("--hidden_p", default=128, type=int)
    parser.add_argument("--buffer_size", default=3_000_000, type=int)
    parser.add_argument("--exp_name", default="rasm_lds")
    parser.add_argument("--env_dim", default=None, type=int)
    parser.add_argument("--improved_loss", action="store_true")
    parser.add_argument("--policy_rollback", action="store_true")
    parser.add_argument("--estimate_expected_via_ibp", action="store_true")
    parser.add_argument("--init_with_static", action="store_true")
    parser.add_argument("--policy_path", default=None)
    parser.add_argument("--rollback_threshold", default=0.99, type=float)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--logger_level", default="ERROR")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--initial_epochs", default=50, type=int)

    args = parser.parse_args()

    # Parse config file
    if not args.no_config:
        if args.config_path is None:
            logger.error("Error: --config_path must be specified when using --load_config")
            sys.exit(1)
        
        config = load_config(args.config_path)
        for key, value in config.items():
            if hasattr(args, key) and "--"+key not in sys.argv:
                setattr(args, key, value)

    # Interpret arguments in the human-readable format
    args.batch_size = interpret_size_arg(args.batch_size)
    args.learner_batch_size = interpret_size_arg(args.learner_batch_size)
    args.grid_size = interpret_size_arg(args.grid_size)
    args.sac_steps = interpret_size_arg(args.sac_steps)

    # Configure logger
    try:
        logger.setLevel(args.logger_level)
    except AttributeError:
        raise ValueError(f"Invalid logger level: {args.logger_level}. Allowed values are: DEBUG, INFO, WARNING, ERROR, CRITICAL.")

    # Check for valid norm
    assert args.norm.lower() in ["l1", "linf"], "L1 and Linf norms are allowed"

    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    monitor = ExperimentMonitor(args.exp_name)

    return args, monitor
