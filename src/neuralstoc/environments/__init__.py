from brax import envs

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

def get_env(args):
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

    return env
