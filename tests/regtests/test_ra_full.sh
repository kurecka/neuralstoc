#!/bin/bash

# Test whethre the RSM loop can train a policy with SAC and synthesize a certificate.
# Should take under 5 minutes to run.

python3 scripts/rsm_loop.py --spec reach_avoid --task control --prob 0.995 \
    --initialize sac --sac_steps 500M --load_from_brax \
    --num_layers_p 2 --num_layers_v 2 --hidden_p 256 --hidden_v 256 \
    --env v2lds --exp_name lds_new \
    --rollback_threshold 0.995 \
    --min_iters 0 --train_p 3 --n_local 1 \
    --fft_threshold 100k --verifier_chunk_size [3.3,3.3]
