#!/bin/bash

## Verification of Final Results (Table 2 NeuralStoc - stability)
python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2lds --exp_name slds_new --load_from_brax --rollback_threshold 0.99 --min_iters 0 --train_p 3 --n_local 10 --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/lds_new_stability_loop_1.jax

python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env vpend --exp_name spend_new --load_from_brax --rollback_threshold 0.9 --v_lip 0 --min_iters 0 --train_p 0 --n_local 10 --init_with_static --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/pend_new_stability_loop_4.jax

python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2cavoid --exp_name scavoid_new --load_from_brax --rollback_threshold 0.99 --min_iters 0 --train_p 3 --n_local 3 --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/cavoid_new_stability_loop_16.jax

python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env vtri --exp_name stri_new --load_from_brax --rollback_threshold 0.99 --min_iters 0 --train_p 0 --learner_batch_size 8k --n_local 4 --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/tri_new_stability_loop_1.jax

python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 --env v2ldss4 --exp_name slds4d_new --load_from_brax --rollback_threshold 0.99 --min_iters 0 --train_p 0 --n_local 3 --env_dim 4 --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/ldss4_new_stability_loop_14.jax 