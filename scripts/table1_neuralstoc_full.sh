#!/bin/bash

## Full Experiment Execution — Reproduce Results from Scratch (Table 1 NeuralStoc - reach avoid)
python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.995 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2lds --exp_name lds_new --load_from_brax --rollback_threshold 0.995 --min_iters 0 --train_p 3 --n_local 10

python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.99 --initialize ppo --num_layers_p 2 --num_layers_v 2 --hidden_p 128 --hidden_v 256 --env v2pend --exp_name pend_new --rollback_threshold 0.99 --min_iters 3 --train_p 3 --learner_batch_size 32k --n_local 10

python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.97 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2cavoid --exp_name cavoid_new --load_from_brax --rollback_threshold 0.97 --min_iters 0 --train_p 0 --n_local 10

python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.95 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env vtri --exp_name tri_new --load_from_brax --rollback_threshold 0.95 --min_iters 0 --train_p 0 --n_local 4

python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.8 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env vhuman2 --exp_name human2_new --load_from_brax --rollback_threshold 0.8 --min_iters 0 --train_p 0 --n_local 3 