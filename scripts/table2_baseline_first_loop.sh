#!/bin/bash

## Continue from First Successful Iteration (Table 2 Baseline - stability)
## Unfortunately, the checkpoints for 2D Linear System (v2lds) are corrupted and cannot be used.
python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize ppo --num_layers_p 2 --num_layers_v 2 --hidden_p 128 --hidden_v 256 --env vpend_200 --exp_name spend_old --no_config --plot --learner_batch_size 16k --batch_size 32k --model mlp --norm linf --ds_type all --v_lr 0.0001 --p_lr 0.000001 --lip_lambda 0.001 --train_p 3 --eps 0.01 --p_lip 4 --v_lip 15 --min_iters 0 --grid_size 32M --buffer_size 6000000 --ppo_iters 200 --continue_rsm 1 --skip_initialize --rsm_path scripts/checkpoints/pend_old_stability_loop_0.jax 