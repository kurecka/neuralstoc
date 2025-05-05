import argparse
import os
import sys
import yaml
import numpy as np

from brax.io import model as brax_model

from neuralstoc.configure import configure
from neuralstoc.environments import get_env
from neuralstoc.rsm.loop import RSMLoop
from neuralstoc.rsm.learner import RSMLearner
from neuralstoc.rsm.verifier import RSMVerifier
from neuralstoc.rsm.descent_verifier import DescentVerifier
from neuralstoc.rsm.lipschitz import get_lipschitz_k


import logging
logger = logging.getLogger("neuralstoc")
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    args, monitor = configure()
    
    env = get_env(args)

    logger.info("Initializing learner")
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
        batch_size=args.learner_batch_size,
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
        logger.info("Loading policy")
        if args.load_from_brax:
            params = brax_model.load_params(policy_path)
            learner.load_from_brax(params)
        else:
            learner.load(policy_path, force_load_all=False)

    if not args.skip_initialize:
        logger.info("Pretraining policy")
        learner.pretrain_policy(args.initialize, filename=policy_path)

    logger.info("Initializing verifier")
    verifier = RSMVerifier(
        learner,
        env,
        batch_size=args.batch_size,
        prob=args.prob,
        target_grid_size=args.grid_size,
        dataset_type=args.ds_type,
        norm=args.norm.lower(),
        n_local=args.n_local,
        buffer_size=args.buffer_size,
        spec=args.spec,
    )

    # TODO: Make this systematic
    if args.env == 'v2lds':
        subspace_size = np.array([3.3, 3.3])
    elif args.env == 'v2cavoid':
        subspace_size = np.array([1.8, 1.8])
    else:
        raise ValueError(f"Unknown environment {args.env}")

    descent_verifier = DescentVerifier(
        env,
        subspace_size=subspace_size,
        policy_apply=learner.p_state.apply_fn,
        policy_ibp=learner.p_ibp.apply,
        value_apply=learner.v_state.apply_fn,
        value_ibp=learner.v_ibp.apply,
        get_lipschitz=lambda: get_lipschitz_k(env, verifier, learner, log=monitor.log)[0],
        target_grid_size=args.grid_size,
        spec=args.spec,
        norm=args.norm.lower(),
    )

    if args.continue_rsm > 0:
        logger.info(f"Continue RSM. path = {args.rsm_path}")
        learner.load(args.rsm_path)
        verifier.grid_size *= args.continue_rsm

    logger.info("Initializing the loop")
    loop = RSMLoop(
        learner,
        verifier,
        descent_verifier,
        env,
        monitor,
        plot=args.plot,
        train_p=args.train_p,
        min_iters=args.min_iters,
        soft_constraint=args.soft_constraint,
        exp_name=args.exp_name,
        policy_rollback=args.policy_rollback,
        rollback_threshold=args.rollback_threshold,
        no_train=args.no_train,
        skip_first=args.continue_rsm > 0,
        epochs=args.epochs,
        initial_epochs=args.initial_epochs,
    )
    logger.info("Evaluating the policy")
    txt_return, res_dict = learner.evaluate_rl()

    logger.info("Plotting")
    monitor.plot_l(env, verifier, learner, f"plots/{args.env}_start_{args.exp_name}.png")
    with open("initialize_results.txt", "a") as f:
        f.write(f"{args.env}: {txt_return}\n")

    if args.only_initialize:
        import sys

        sys.exit(0)

    if args.smoke_test:
        import sys
        if res_dict['num_end_in_target'] <= 0:
            sys.exit(0)
        else:
            sys.exit(1)

    logger.info("Running the loop")
    sat = loop.run(args.timeout * 60)

    logger.info("Plotting")
    monitor.plot_l(env, verifier, learner, f"plots/{args.env}_end_{args.exp_name}.png")

    logger.info("Writing results")
    monitor.write_results(args, txt_return, sat)
