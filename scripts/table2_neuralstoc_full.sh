#!/bin/bash

## Full Experiment Execution — Reproduce Results from Scratch (Table 2 NeuralStoc - stability)
python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2lds --exp_name slds_new --load_from_brax --rollback_threshold 0.99 --min_iters 0 --train_p 3 --n_local 10

python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env vpend --exp_name spend_new --load_from_brax --rollback_threshold 0.9 --v_lip 0 --min_iters 0 --train_p 0 --n_local 10 --init_with_static

python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2cavoid --exp_name scavoid_new --load_from_brax --rollback_threshold 0.99 --min_iters 0 --train_p 3 --n_local 3

python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env vtri --exp_name stri_new --load_from_brax --rollback_threshold 0.99 --min_iters 0 --train_p 0 --learner_batch_size 8k --n_local 4

python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2ldss4 --exp_name slds4d_new --load_from_brax --rollback_threshold 0.99 --min_iters 0 --train_p 0 --n_local 3 --env_dim 4 