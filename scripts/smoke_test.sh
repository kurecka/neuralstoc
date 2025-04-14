#!/bin/bash


python3 scripts/rsm_loop.py --exp_name spend_old --initialize ppo --plot --learner_batch_size 2 --batch_size 2 --env vpend --model mlp --norm linf --ds_type all --v_lr 0.0001 --p_lr 0.000001 --lip_lambda 0.001 --train_p 0 --eps 0.01 --p_lip 4 --v_lip 15 --prob 1 --min_iters 0 --hidden_p 128 --hidden_v 128 --grid_size 4 --buffer_size 2 --spec stability --ppo_iters 1 --timeout 1
