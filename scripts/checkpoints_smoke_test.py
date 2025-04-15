#!/usr/bin/env python3
"""
Smoke test to verify that checkpoint loading works properly.
"""

import os
import subprocess
import sys
import time
from datetime import datetime

os.makedirs("logs", exist_ok=True)

TESTS = [
    {
        "name": "lds_old_reach_avoid_loop_0",
        "cmd": "python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.95 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2lds --exp_name lds_old --no_config --load_from_brax --plot --learner_batch_size 32k --batch_size 32k --model mlp --norm l1 --ds_type all --v_lr 0.0005 --p_lr 0.000005 --lip_lambda 0.001 --train_p 3 --eps 0.3 --p_lip 4 --v_lip 15 --min_iters 0 --grid_size 32M --buffer_size 6000000 --continue_rsm 1 --skip_initialize --rsm_path scripts/checkpoints/lds_old_reach_avoid_loop_0.jax --smoke_test"
    },
    {
        "name": "lds_new_reach_avoid_loop_0",
        "cmd": "python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.995 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2lds --exp_name lds_new --load_from_brax --rollback_threshold 0.995 --min_iters 0 --train_p 3 --n_local 10 --continue_rsm 1 --skip_initialize --rsm_path scripts/checkpoints/lds_new_reach_avoid_loop_0.jax --smoke_test"
    },
    {
        "name": "pend_old_reach_avoid_loop_0",
        "cmd": "python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.9 --initialize ppo --num_layers_p 2 --num_layers_v 2 --hidden_p 128 --hidden_v 256 --env v2pend_200 --exp_name pend_old --no_config --plot --learner_batch_size 16k --batch_size 32k --model mlp --norm linf --ds_type all --v_lr 0.0005 --p_lr 0.00005 --lip_lambda 0.001 --train_p 3 --eps 0.01 --p_lip 4 --v_lip 15 --min_iters 0 --grid_size 32M --buffer_size 6000000 --env_dim 4 --ppo_iters 200 --continue_rsm 1 --skip_initialize --rsm_path scripts/checkpoints/pend_old_reach_avoid_loop_0.jax --smoke_test"
    },
    {
        "name": "pend_new_reach_avoid_loop_0",
        "cmd": "python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.99 --initialize ppo --num_layers_p 2 --num_layers_v 2 --hidden_p 128 --hidden_v 256 --env v2pend_100 --exp_name pend_new --rollback_threshold 0.99 --min_iters 3 --train_p 3 --learner_batch_size 32k --n_local 10 --continue_rsm 1 --skip_initialize --rsm_path scripts/checkpoints/pend_new_reach_avoid_loop_0.jax --smoke_test"
    },
    {
        "name": "cavoid_old_reach_avoid_loop_0",
        "cmd": "python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.9 --initialize ppo --num_layers_p 2 --num_layers_v 2 --hidden_p 128 --hidden_v 256 --env v2cavoid_100 --exp_name cavoid_old --no_config --plot --learner_batch_size 16k --batch_size 32k --model mlp --norm linf --ds_type all --v_lr 0.0005 --p_lr 0.00005 --lip_lambda 0.001 --train_p 3 --eps 0.05 --p_lip 4 --v_lip 15 --min_iters 0 --grid_size 32M --buffer_size 6000000 --env_dim 4 --ppo_iters 100 --continue_rsm 1 --skip_initialize --rsm_path scripts/checkpoints/cavoid_old_reach_avoid_loop_0.jax --smoke_test"
    },
    {
        "name": "cavoid_new_reach_avoid_loop_0",
        "cmd": "python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.97 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2cavoid --exp_name cavoid_new --load_from_brax --rollback_threshold 0.97 --min_iters 0 --train_p 0 --n_local 10 --continue_rsm 1 --skip_initialize --rsm_path scripts/checkpoints/cavoid_new_reach_avoid_loop_0.jax --smoke_test"
    },
    {
        "name": "tri_new_reach_avoid_loop_0",
        "cmd": "python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.95 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env vtri --exp_name tri_new --load_from_brax --rollback_threshold 0.95 --min_iters 0 --train_p 0 --learner_batch_size 8k --n_local 4 --continue_rsm 1 --skip_initialize --rsm_path scripts/checkpoints/tri_new_reach_avoid_loop_0.jax --smoke_test"
    },
    {
        "name": "human2_new_reach_avoid_loop_0",
        "cmd": "python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.8 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env vhuman2 --exp_name human2_new --load_from_brax --rollback_threshold 0.8 --min_iters 0 --train_p 0 --n_local 3 --continue_rsm 1 --skip_initialize --rsm_path scripts/checkpoints/human2_new_reach_avoid_loop_0.jax --smoke_test"
    },
    {
        "name": "slds_new_stability_loop_0",
        "cmd": "python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2lds --exp_name slds_new --load_from_brax --rollback_threshold 0.99 --min_iters 0 --train_p 3 --n_local 10 --continue_rsm 1 --skip_initialize --rsm_path scripts/checkpoints/lds_new_stability_loop_0.jax --smoke_test"
    },
    {
        "name": "spend_old_stability_loop_0",
        "cmd": "python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize ppo --num_layers_p 2 --num_layers_v 2 --hidden_p 128 --hidden_v 256 --env vpend_200 --exp_name spend_old --no_config --plot --learner_batch_size 16k --batch_size 32k --model mlp --norm linf --ds_type all --v_lr 0.0001 --p_lr 0.000001 --lip_lambda 0.001 --train_p 3 --eps 0.01 --p_lip 4 --v_lip 15 --min_iters 0 --grid_size 32M --buffer_size 6000000 --ppo_iters 200 --continue_rsm 1 --skip_initialize --rsm_path scripts/checkpoints/pend_old_stability_loop_0.jax --smoke_test"
    },
    {
        "name": "spend_new_stability_loop_0",
        "cmd": "python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env vpend --exp_name spend_new --load_from_brax --rollback_threshold 0.9 --v_lip 0 --min_iters 0 --train_p 0 --n_local 10 --init_with_static --continue_rsm 1 --skip_initialize --rsm_path scripts/checkpoints/pend_new_stability_loop_0.jax --smoke_test"
    },
    {
        "name": "scavoid_new_stability_loop_0",
        "cmd": "python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2cavoid --exp_name scavoid_new --load_from_brax --rollback_threshold 0.99 --min_iters 0 --train_p 3 --n_local 3 --continue_rsm 1 --skip_initialize --rsm_path scripts/checkpoints/cavoid_new_stability_loop_0.jax --smoke_test"
    },
    {
        "name": "stri_new_stability_loop_0",
        "cmd": "python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env vtri --exp_name stri_new --load_from_brax --rollback_threshold 0.99 --min_iters 0 --train_p 0 --learner_batch_size 8k --n_local 4 --continue_rsm 1 --skip_initialize --rsm_path scripts/checkpoints/tri_new_stability_loop_0.jax --smoke_test"
    },
    {
        "name": "slds4d_new_stability_loop_0",
        "cmd": "python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2ldss4 --exp_name slds4d_new --load_from_brax --rollback_threshold 0.99 --min_iters 0 --train_p 0 --n_local 3 --env_dim 4 --continue_rsm 1 --skip_initialize --rsm_path scripts/checkpoints/ldss4_new_stability_loop_0.jax --smoke_test"
    },
    {
        "name": "lds_old_reach_avoid_loop_5",
        "cmd": "python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.95 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2lds --exp_name lds_old --no_config --load_from_brax --plot --learner_batch_size 32k --batch_size 32k --model mlp --norm l1 --ds_type all --v_lr 0.0005 --p_lr 0.000005 --lip_lambda 0.001 --train_p 3 --eps 0.3 --p_lip 4 --v_lip 15 --min_iters 0 --grid_size 32M --buffer_size 6000000 --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/lds_old_reach_avoid_loop_5.jax --smoke_test"
    },
    {
        "name": "lds_new_reach_avoid_loop_15",
        "cmd": "python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.995 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2lds --exp_name lds_new --load_from_brax --rollback_threshold 0.995 --min_iters 0 --train_p 3 --n_local 10 --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/lds_new_reach_avoid_loop_15.jax --smoke_test"
    },
    {
        "name": "pend_old_reach_avoid_loop_14",
        "cmd": "python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.9 --initialize ppo --num_layers_p 2 --num_layers_v 2 --hidden_p 128 --hidden_v 256 --env v2pend_200 --exp_name pend_old --no_config --plot --learner_batch_size 16k --batch_size 32k --model mlp --norm linf --ds_type all --v_lr 0.0005 --p_lr 0.00005 --lip_lambda 0.001 --train_p 3 --eps 0.01 --p_lip 4 --v_lip 15 --min_iters 0 --grid_size 32M --buffer_size 6000000 --env_dim 4 --ppo_iters 200 --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/pend_old_reach_avoid_loop_14.jax --smoke_test"
    },
    {
        "name": "pend_new_reach_avoid_loop_7",
        "cmd": "python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.99 --initialize ppo --num_layers_p 2 --num_layers_v 2 --hidden_p 128 --hidden_v 256 --env v2pend_100 --exp_name pend_new --rollback_threshold 0.99 --min_iters 3 --train_p 3 --learner_batch_size 32k --n_local 10 --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/pend_new_reach_avoid_loop_7.jax --smoke_test"
    },
    {
        "name": "cavoid_old_reach_avoid_loop_12",
        "cmd": "python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.9 --initialize ppo --num_layers_p 2 --num_layers_v 2 --hidden_p 128 --hidden_v 256 --env v2cavoid_100 --exp_name cavoid_old --no_config --plot --learner_batch_size 16k --batch_size 32k --model mlp --norm linf --ds_type all --v_lr 0.0005 --p_lr 0.00005 --lip_lambda 0.001 --train_p 3 --eps 0.05 --p_lip 4 --v_lip 15 --min_iters 0 --grid_size 32M --buffer_size 6000000 --env_dim 4 --ppo_iters 100 --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/cavoid_old_reach_avoid_loop_12.jax --smoke_test"
    },
    {
        "name": "cavoid_new_reach_avoid_loop_4",
        "cmd": "python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.97 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2cavoid --exp_name cavoid_new --load_from_brax --rollback_threshold 0.97 --min_iters 0 --train_p 0 --n_local 10 --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/cavoid_new_reach_avoid_loop_4.jax --smoke_test"
    },
    {
        "name": "tri_new_reach_avoid_loop_9",
        "cmd": "python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.95 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env vtri --exp_name tri_new --load_from_brax --rollback_threshold 0.95 --min_iters 0 --train_p 0 --learner_batch_size 8k --n_local 4 --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/tri_new_reach_avoid_loop_9.jax --smoke_test"
    },
    {
        "name": "human2_new_reach_avoid_loop_31",
        "cmd": "python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.8 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env vhuman2 --exp_name human2_new --load_from_brax --rollback_threshold 0.8 --min_iters 0 --train_p 0 --n_local 3 --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/human2_new_reach_avoid_loop_31.jax --smoke_test"
    },
    {
        "name": "slds_new_stability_loop_1",
        "cmd": "python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2lds --exp_name slds_new --load_from_brax --rollback_threshold 0.99 --min_iters 0 --train_p 3 --n_local 10 --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/lds_new_stability_loop_1.jax --smoke_test"
    },
    {
        "name": "spend_old_stability_loop_8",
        "cmd": "python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize ppo --num_layers_p 2 --num_layers_v 2 --hidden_p 128 --hidden_v 256 --env vpend_200 --exp_name spend_old --no_config --plot --learner_batch_size 16k --batch_size 32k --model mlp --norm linf --ds_type all --v_lr 0.0001 --p_lr 0.000001 --lip_lambda 0.001 --train_p 3 --eps 0.01 --p_lip 4 --v_lip 15 --min_iters 0 --grid_size 32M --buffer_size 6000000 --ppo_iters 200 --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/pend_old_stability_loop_8.jax --smoke_test"
    },
    {
        "name": "spend_new_stability_loop_4",
        "cmd": "python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env vpend --exp_name spend_new --load_from_brax --rollback_threshold 0.9 --v_lip 0 --min_iters 0 --train_p 0 --n_local 10 --init_with_static --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/pend_new_stability_loop_4.jax --smoke_test"
    },
    {
        "name": "scavoid_new_stability_loop_16",
        "cmd": "python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2cavoid --exp_name scavoid_new --load_from_brax --rollback_threshold 0.99 --min_iters 0 --train_p 3 --n_local 3 --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/cavoid_new_stability_loop_16.jax --smoke_test"
    },
    {
        "name": "stri_new_stability_loop_1",
        "cmd": "python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env vtri --exp_name stri_new --load_from_brax --rollback_threshold 0.99 --min_iters 0 --train_p 0 --learner_batch_size 8k --n_local 4 --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/tri_new_stability_loop_1.jax --smoke_test"
    },
    {
        "name": "slds4d_new_stability_loop_14",
        "cmd": "python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2ldss4 --exp_name slds4d_new --load_from_brax --rollback_threshold 0.99 --min_iters 0 --train_p 0 --n_local 3 --env_dim 4 --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/ldss4_new_stability_loop_14.jax --smoke_test"
    }
]

def run_test(test):
    """Run a test command and log its output"""
    test_name = test["name"]
    cmd = test["cmd"]
    log_file = f"logs/{test_name}_log.txt"
    
    print(f"Running test: {test_name}")
    print(f"Command: {cmd}")
    print(f"Logging to: {log_file}")
    
    with open(log_file, "w") as f:
        f.write(f"=== Running command: {cmd} ===\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    start_time = time.time()
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8'
    )
    
    with open(log_file, "a") as f:
        for line in process.stdout:
            f.write(line)
            f.flush()
    
    exit_code = process.wait()
    duration = time.time() - start_time
    
    with open(log_file, "a") as f:
        f.write(f"\n=== Test completed with exit code: {exit_code} ===\n")
        f.write(f"Duration: {duration:.2f} seconds\n")
    
    return exit_code == 0

def main():
    """Run all tests and report results"""
    print(f"Starting smoke tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Running {len(TESTS)} tests\n")
    
    failed_tests = []
    
    for test in TESTS:
        success = run_test(test)
        if success:
            print(f"✓ Test passed: {test['name']}\n")
        else:
            print(f"✗ Test failed: {test['name']}\n")
            failed_tests.append(test["name"])
    
    print(f"Tests completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{len(TESTS) - len(failed_tests)}/{len(TESTS)} tests passed")
    
    if failed_tests:
        print(f"Failed tests: {', '.join(failed_tests)}")
        return 1
    else:
        print("Smoke test passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 