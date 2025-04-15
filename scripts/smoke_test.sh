#!/bin/bash


python3 scripts/rsm_loop.py --env vpend --initialize ppo --task control --spec stability --prob 1 --num_layers_v 2 --num_layers_p 2 --hidden_p 64 --hidden_v 64 --exp_name spend --plot --learner_batch_size 2 --batch_size 2 --n_local 2 --grid_size 64 --buffer_size 4 --ppo_iters 1 --timeout 1 --no_config --improved_loss --policy_rollback --estimate_expected_via_ibp
