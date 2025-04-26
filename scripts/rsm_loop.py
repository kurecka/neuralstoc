import argparse
import os
import sys
import yaml

from brax import envs

from brax.io import model as brax_model

from neuralstoc.environments.vrl_environments import (
    vLDSEnv,
    vHumanoidBalance2,
    vInvertedPendulum,
    vCollisionAvoidanceEnv,
    vTripleIntegrator,
    vLDSS,
    v2CollisionAvoidanceEnv,
    v2LDSS,
    v2LDSEnv,
    v2InvertedPendulum
)

from neuralstoc.environments.brl_environments import (
    bHumanoidBalance2,
    bLDSEnv,
    bInvertedPendulum,
    bCollisionAvoidanceEnv,
    bTripleIntegrator,
    bLDSS,
    b2CollisionAvoidanceEnv,
    b2LDSS,
    b2LDSEnv,
    b2InvertedPendulum
)

from neuralstoc.rsm.loop import RSMLoop
from neuralstoc.rsm.learner import RSMLearner
from neuralstoc.rsm.verifier import RSMVerifier

def interpret_size_arg(cmd):
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

def load_config(config_path):
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



if __name__ == "__main__":
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

    args = parser.parse_args()
    
    if not args.no_config:
        if args.config_path is None:
            print("Error: --config_path must be specified when using --load_config")
            sys.exit(1)
        
        config = load_config(args.config_path)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    if args.env.startswith("vcavoid"):
        env = vCollisionAvoidanceEnv()
        envs.register_environment(args.env, bCollisionAvoidanceEnv)
        env.name = args.env
    elif args.env.startswith("v2cavoid"):
        env = v2CollisionAvoidanceEnv()
        envs.register_environment(args.env, b2CollisionAvoidanceEnv)
        env.name = args.env
    elif args.env.startswith("v2ldss"):
        env = v2LDSS(num_dims=args.env_dim)
        envs.register_environment(args.env, b2LDSS)
        env.name = args.env
    elif args.env.startswith("v2lds"):
        env = v2LDSEnv()
        envs.register_environment(args.env, b2LDSEnv)
        env.name = args.env
    elif args.env.startswith("v2pend"):
        env = v2InvertedPendulum()
        envs.register_environment(args.env, bInvertedPendulum)
        env.name = args.env
    elif args.env.startswith("vldss"):
        env = vLDSS(num_dims=args.env_dim)
        envs.register_environment(args.env, bLDSS)
        env.name = args.env
    elif args.env.startswith("vlds"):
        env = vLDSEnv()
        envs.register_environment(args.env, bLDSEnv)
        env.name = args.env
    elif args.env.startswith("vpend"):
        env = vInvertedPendulum()
        envs.register_environment(args.env, bInvertedPendulum)
        env.name = args.env
    elif args.env.startswith("vhuman2"):
        env = vHumanoidBalance2()
        envs.register_environment(args.env, bHumanoidBalance2)
        env.name = args.env
    elif args.env.startswith("vtri"):
        env = vTripleIntegrator()
        envs.register_environment(args.env, bTripleIntegrator)
        env.name = args.env
    else:
        raise ValueError(f"Unknown environment '{args.env}'")

    assert args.norm.lower() in ["l1", "linf"], "L1 and Linf norms are allowed"
    os.makedirs("checkpoints", exist_ok=True)
    learner = RSMLearner(
        [args.hidden_v for i in range(args.num_layers_v)],
        [args.hidden_p for i in range(args.num_layers_p)],
        env,
        p_lr=args.p_lr,
        c_lr=args.c_lr,
        v_lr=args.v_lr,
        p_lip=args.p_lip,
        v_lip=args.v_lip,
        lip_lambda=args.lip_lambda,
        eps=args.eps,
        prob=args.prob,
        v_activation=args.v_act,
        norm=args.norm.lower(),
        n_step=args.n_step,
        model=args.model,
        n_local=args.n_local,
        ppo_num_iters=args.ppo_iters,
        ppo_lip_start=0.05 / 10,
        ppo_lip_end=0.05,
        ppo_save_every=None,
        batch_size=interpret_size_arg(args.learner_batch_size),
        ppo_std_start=args.std_start,
        ppo_std_end=args.std_end,
        ppo_normalize_r=args.normalize_r > 0,
        ppo_normalize_a=args.normalize_a > 0,
        use_brax=args.load_from_brax,
        improved_loss=args.improved_loss,
        env_dim=args.env_dim if args.env.startswith("vldss") or args.env.startswith("v2ldss") else None,
        estimate_expected_via_ibp=args.estimate_expected_via_ibp,
        init_with_static=args.init_with_static,
        spec=args.spec,
        task=args.task,
        policy_type=args.initialize,
    )
    if args.load_from_brax or args.initialize == "sac":
        policy_path = args.policy_path if args.policy_path is not None else f"checkpoints/{args.env}_{args.initialize}"
    else:
        policy_path = args.policy_path if args.policy_path is not None else f"checkpoints/{args.env}_{args.initialize}.jax"
    if args.skip_initialize and args.continue_rsm <= 0:
        if args.load_from_brax:
            params = brax_model.load_params(policy_path)
            learner.load_from_brax(params)
        else:
            learner.load(policy_path, force_load_all=False)

    if not args.skip_initialize:
        learner.pretrain_policy(args.initialize, filename=policy_path)

    verifier = RSMVerifier(
        learner,
        env,
        batch_size=interpret_size_arg(args.batch_size),
        prob=args.prob,
        target_grid_size=interpret_size_arg(args.grid_size),
        dataset_type=args.ds_type,
        norm=args.norm.lower(),
        n_local=args.n_local,
        buffer_size=args.buffer_size,
        spec=args.spec,
    )

    if args.continue_rsm > 0:
        learner.load(args.rsm_path)
        verifier.grid_size *= args.continue_rsm

    loop = RSMLoop(
        learner,
        verifier,
        env,
        plot=args.plot,
        train_p=args.train_p,
        min_iters=args.min_iters,
        soft_constraint=args.soft_constraint,
        exp_name=args.exp_name,
        policy_rollback=args.policy_rollback,
        rollback_threshold=args.rollback_threshold,
        no_train=args.no_train,
        skip_first=args.continue_rsm > 0,
    )
    txt_return, res_dict = learner.evaluate_rl()

    # loop.plot_l(f"{loop.exp_name}/plots/{args.env}_start_{args.exp_name}.png")
    with open("initialize_results.txt", "a") as f:
        f.write(f"{args.env}: {txt_return}\n")

    if args.only_initialize:
        import sys

        sys.exit(0)

    if args.smoke_test:
        import sys
        if res_dict['num_end_in_target'] <= 0:
            sys.exit(1)
        else:
            sys.exit(0)


    sat = loop.run(args.timeout * 60)
    loop.plot_l(f"{loop.exp_name}/plots/{args.env}_end_{args.exp_name}.png")

    os.makedirs("study_results", exist_ok=True)
    env_name = args.env.split("_")
    if len(env_name) > 2:
        env_name = env_name[0] + "_" + env_name[1]
    else:
        env_name = args.env
    cmd_line = " ".join(sys.argv)
    with open(f"study_results/info_{args.exp_name}.log", "a") as f:
        f.write(f"python3 {cmd_line}\n")
        f.write("    args=" + str(vars(args)) + "\n")
        f.write("    return =" + txt_return + "\n")
        f.write("    info=" + str(loop.info) + "\n")
        f.write("    sat=" + str(sat) + "\n")
        f.write("\n\n")
    with open(f"global_summary.txt", "a") as f:
        f.write(f"{cmd_line}\n")
        f.write("    args=" + str(vars(args)) + "\n")
        f.write("    return =" + txt_return + "\n")
        f.write("    info=" + str(loop.info) + "\n")
        f.write("    sat=" + str(sat) + "\n")
        f.write("\n\n")