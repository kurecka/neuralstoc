from types import SimpleNamespace
from copy import deepcopy

import jax
import jax.numpy as jnp
from functools import partial


from neuralstoc.rsm.ibp import IBPMLP
from neuralstoc.utils import (
    jax_save,
    jax_load,
    lipschitz_l1_jax,
    martingale_loss,
    triangular,
    MLP,
    create_train_state,
    lipschitz_linf_jax,
    get_pmass_grid,
    compute_expected_l,
    jv_contains, TMLP, set_tnet_params,
)
import numpy as np

from neuralstoc.rl.sac import SAC
from neuralstoc.rl.ppo import vPPO
from brax.training.acme import running_statistics
from brax.training.acme import specs


class RSMLearner:
    """
    RSMLearner implements the learner component of the learner-verifier framework.
    
    The learner module is responsible for training neural networks that represent:
    1. A neural supermartingale certificate (V) that proves a specification is satisfied
    2. A neural controller (π) that controls the system to satisfy the specification
    
    The module implements different training strategies depending on the specification:
    - Reach-avoid (RASM)
    - Reachability (RSM)
    - Safety (SBFs)
    - Stability (sRSM)
    
    The learner can either work in control synthesis mode, where it trains both the
    certificate and controller, or in verification mode, where it only trains the
    certificate for a given fixed controller.
    
    The learning process uses a loss function that encodes the defining conditions of
    the respective supermartingale, and is guided by counterexamples from the verifier.

    """
    
    def __init__(
            self,
            l_hidden=[256, 256],
            p_hidden=[256, 256],
            env='v2lds',
            lip_lambda=0.2,
            p_lip=4.0,
            v_lip=15.0,
            eps=1,
            prob=0.3,
            v_activation="relu",
            norm="linf",
            batch_size=256,
            p_lr=0.00005,
            c_lr=0.0005,
            c_ema=0.9,
            v_lr=0.0005,
            v_ema=0.9,
            n_step=1,
            model="tmlp",
            n_local=100,
            ppo_num_iters=10,
            ppo_std_start=0.3,
            ppo_std_end=0.03,
            ppo_lip_start=0.0,
            ppo_lip_end=0.1,
            ppo_normalize_r=False,
            ppo_normalize_a=True,
            ppo_save_every=None,
            use_brax=True,
            opt="adamw",
            improved_loss=True,
            estimate_expected_via_ibp=True,
            small_delta=0.1,
            env_dim=None,
            policy_type="sac",
            init_with_static=False,
            spec='reach_avoid',
            task='control',
    ) -> None:
        """
        Initialize the RSMLearner module.
        
        Args:
            l_hidden: List of hidden layer sizes for the supermartingale certificate network
            p_hidden: List of hidden layer sizes for the controller network
            env: The stochastic environment
            lip_lambda: If old loss used, regularization factor multiplied with the Lipschitz loss. If improved loss used, it is an estimatoin of the looseness of the global Lipschitz constant compared to the local Lipschitz constant.
            p_lip: Maximum allowed Lipschitz constant for the controller (old loss)
            v_lip: Maximum allowed Lipschitz constant for the certificate (old loss)
            eps: Desired expected decrease value for the certificate (old loss)
            prob: Target probability threshold for the specification
            v_activation: Activation function for the certificate network ('relu' or 'tanh')
            norm: Norm for Lipschitz calculations ('l1' or 'linf')
            batch_size: Batch size for training
            p_lr: Learning rate for the controller network
            c_lr: Learning rate for the critic network (used in PPO pre-training)
            c_ema: Exponential moving average factor for the critic (used in PPO pre-training)
            v_lr: Learning rate for the certificate network
            v_ema: Exponential moving average factor for the certificate (used in PPO pre-training)
            n_step: Number of environment steps per training iteration
            model: Network architecture type ('mlp' or 'tmlp' for local Lipschitz calculations) - tmlp considers the torch variant of the model at the same time for local Lipschitz calculations
            n_local: Grid size for computing local Lipschitz constants
            ppo_num_iters: Number of PPO iterations for controller pre-training
            ppo_std_start: Initial standard deviation for PPO's Gaussian policy
            ppo_std_end: Final standard deviation for PPO's Gaussian policy
            ppo_lip_start: Initial Lipschitz regularization factor for PPO
            ppo_lip_end: Final Lipschitz regularization factor for PPO
            ppo_normalize_r: Whether to normalize rewards in PPO
            ppo_normalize_a: Whether to normalize advantages in PPO
            ppo_save_every: Save frequency during PPO training (None = no saving)
            use_brax: Whether to use Brax for the environment
            opt: Optimizer type ('adam', 'adamw')
            improved_loss: Whether to use the optimized loss function
            estimate_expected_via_ibp: Whether to use IBP for expected value estimation
            small_delta: Small delta constant in stability specifications
            env_dim: Environment dimension (optional - for LDS 4D)
            policy_type: Type of policy for initialization ('ppo' or 'sac')
            init_with_static: Whether to initialize with the old loss
            spec: Specification type ('reach_avoid', 'safety', 'reachability', or 'stability')
            task: Task type ('control' or 'verification')
        """
        self.env = env
        self.n_step = n_step
        self.eps = jnp.float32(eps)
        self.prob = jnp.float32(prob)
        self.small_delta = small_delta
        self.ppo_num_iters = ppo_num_iters
        self.ppo_std_start = ppo_std_start
        self.ppo_std_end = ppo_std_end
        self.policy_type = policy_type
        self.ppo_lip_start = ppo_lip_start
        self.ppo_lip_end = ppo_lip_end
        self.ppo_normalize_r = ppo_normalize_r
        self.ppo_normalize_a = ppo_normalize_a
        self.ppo_save_every = ppo_save_every
        self.obs_normalization = None
        self.model = model
        self.init_with_static = init_with_static
        self.opt = opt
        self.p_lr = p_lr
        self.c_lr = c_lr
        self.v_lr = v_lr
        self.grid_size = None
        self.batch_size = batch_size
        self.K_p = None
        self.K_l = None
        self.improved_loss = improved_loss
        assert spec in ["reach_avoid", "safety", "reachability", "stability"]
        if spec == "reach_avoid" or spec == "safety":
            assert prob >= 0.0
            assert prob < 1.0
        elif spec == "stability":
            assert prob == 1.0
        else:
            assert prob >= 0.0
            assert prob <= 1.0
        self.spec = spec
        assert task in ["control", "verification"]
        self.task = task

        assert norm in ["l1", "linf"]
        self.norm = norm
        self.n_local = n_local
        self.estimate_expected_via_ibp = estimate_expected_via_ibp
        self.use_brax = use_brax
        self.norm_fn = lipschitz_l1_jax if norm == "l1" else lipschitz_linf_jax
        action_dim = self.env.action_space.shape[0] if not use_brax else self.env.action_space.shape[0] * 2
        self.action_dim = action_dim
        obs_dim = self.env.observation_dim
        pmass_n = (
            10 if self.env.observation_dim == 2 else 6
        )  # number of sums for the expectation computation
        self._cached_pmass_grid = get_pmass_grid(self.env, pmass_n)

        self.p_ibp = IBPMLP(
            p_hidden + [action_dim], activation="relu", softplus_output=False
        )
        if model == "tmlp":
            self.v_ibp = IBPMLP(
                l_hidden + [1], activation=v_activation, softplus_output=False
            )
            v_net = MLP(l_hidden + [1], activation=v_activation, softplus_output=False)
        else:
            self.v_ibp = IBPMLP(
                l_hidden + [1], activation=v_activation, softplus_output=True
            )
            v_net = MLP(l_hidden + [1], activation=v_activation, softplus_output=True)
        self.c_net = MLP(l_hidden + [1], activation="relu", softplus_output=False)
        self.p_net = MLP(p_hidden + [action_dim], activation="relu")

        if model == "tmlp":
            self.v_tnet = TMLP(obs_dim, l_hidden + [1], activation=v_activation, softplus_output=False)
            self.p_tnet = TMLP(obs_dim, p_hidden + [action_dim], activation='relu', softplus_output=False)

            self.v_tnet.eval()
            self.p_tnet.eval()


        self.v_state = create_train_state(
            v_net, jax.random.PRNGKey(1), obs_dim, v_lr, ema=v_ema, opt=opt
        )
        self.c_state = create_train_state(
            self.c_net, jax.random.PRNGKey(3), obs_dim, c_lr, ema=c_ema, opt=opt
        )
        self.p_state = create_train_state(
            self.p_net,
            jax.random.PRNGKey(2),
            obs_dim,
            p_lr,
            use_brax=use_brax,
            out_dim=self.env.action_space.shape[0],
            obs_normalization=self.obs_normalization,
            opt=opt
        )
        self.p_lip = jnp.float32(p_lip)
        self.v_lip = jnp.float32(v_lip)
        self.lip_lambda = jnp.float32(lip_lambda)
        self.lip_lambda_l = jnp.float32(lip_lambda)
        self.lip_lambda_p = jnp.float32(lip_lambda)

        self.rng = jax.random.PRNGKey(777)
        self._debug_init = []
        self._debug_unsafe = []
        self.ppo = vPPO(
                self.p_state,
                self.c_state,
                self.env,
                self.p_lip,
                norm=self.norm,
                normalize_r=self.ppo_normalize_r,
                normalize_a=self.ppo_normalize_a,
        )
        self.sac = SAC(env.name, env_dim, p_hidden)
        self.p_init_params = None

    def pretrain_policy(
            self,
            initialize,
            verbose=True,
            filename=None
    ):
        """
        Pre-trains the policy network using reinforcement learning.
        
        Uses either PPO or SAC algorithm to initialize the controller before
        starting the learner-verifier loop.
        
        Args:
            initialize: Algorithm to use for pre-training ('ppo' or 'sac')
            verbose: Whether to print training progress information
            filename: Path to save the pre-trained model
        """
        if initialize == "ppo":

            self.ppo.run(
                self.ppo_num_iters,
                self.ppo_std_start,
                self.ppo_std_end,
                self.ppo_lip_start,
                self.ppo_lip_end,
                self.ppo_save_every,
                verbose
            )

            # Copy from PPO
            self.p_state = self.ppo.p_state
            self.c_state = self.ppo.c_state
            self.save(filename)
            print("[SAVED]")
        elif initialize == "sac":
            params = self.sac.train(filename)
            self.load_from_brax(params)
        else:
            raise ValueError("Unknown initialization method")

    def evaluate_rl(self):
        """
        Evaluates the current policy by running episodes in the environment.
        
        Runs multiple trajectories and computes statistics on:
        - Accumulated rewards
        - Percentage of trajectories ending in target region
        
        Returns:
            text: String summary of the evaluation results
            res_dict: Dictionary containing detailed evaluation metrics
        """
        n = 512
        rng = jax.random.PRNGKey(2)
        rng, r = jax.random.split(rng)
        r = jax.random.split(r, n)
        state, obs = self.env.v_reset(r)
        total_reward = jnp.zeros(n)
        done = jnp.zeros(n, dtype=jnp.bool_)
        while not np.any(done):
            action_mean = self.p_state.apply_fn(self.p_state.params, obs)
            rng, r = jax.random.split(rng)
            r = jax.random.split(r, n)
            state, obs, reward, next_done = self.env.v_step(state, action_mean, r)
            total_reward += reward * (1.0 - done)
            done = next_done

        contains = None
        for target_space in self.env.target_spaces:
            c = jv_contains(target_space, obs)
            if contains is not None:
                contains = jnp.logical_or(contains, c)
            else:
                contains = c

        num_end_in_target = jnp.sum(contains.astype(jnp.int64))
        num_traj = contains.shape[0]

        text = f"Rollouts (n={n}): {np.mean(total_reward):0.1f} +- {np.std(total_reward):0.1f} [{np.min(total_reward):0.1f}, {np.max(total_reward):0.1f}] ({100 * num_end_in_target / num_traj:0.2f}% end in target)"
        print(text)
        res_dict = {
            "mean_r": np.mean(total_reward),
            "std_r": np.std(total_reward),
            "min_r": np.min(total_reward),
            "max_r": np.max(total_reward),
            "num_end_in_target": num_end_in_target,
            "num_traj": num_traj,
        }
        return text, res_dict

    @partial(jax.jit, static_argnums=(0, 2))
    def sample_init(self, rng, n):
        """
        Generates n random samples from the initial state space.
        
        The function samples states from:
        - Random points within initial regions
        - Points on the boundaries of initial regions
        - Corner points of initial regions
        
        Args:
            rng: Random number generator key
            n: Number of samples to generate per initial space
            
        Returns:
            Array of sampled initial states
        """
        rngs = jax.random.split(rng, len(self.env.init_spaces))
        per_space_n = n // len(self.env.init_spaces)

        batch = []
        for i in range(len(self.env.init_spaces)):
            x = jax.random.uniform(
                rngs[i],
                (per_space_n, self.env.observation_dim),
                minval=self.env.init_spaces[i].low,
                maxval=self.env.init_spaces[i].high,
            )
            batch.append(x)
            # projecting x onto the boundary of the space
            for j in range(self.env.observation_dim):
                x_j = x.at[:, j].set(self.env.init_spaces[i].low[j])
                batch.append(x_j)
                x_j = x.at[:, j].set(self.env.init_spaces[i].high[j])
                batch.append(x_j)

            # adding the corner points of the space
            low_values = self.env.init_spaces[i].low
            high_values = self.env.init_spaces[i].high
            # corner points
            corner_points = []
            for j in range(2 ** self.env.observation_dim):
                corner_points.append(
                    jnp.array([
                        low_values[k] if (j & (1 << k)) == 0 else high_values[k]
                        for k in range(self.env.observation_dim)
                    ])
                )
            batch.append(jnp.array(corner_points))
        return jnp.concatenate(batch, axis=0)

    @partial(jax.jit, static_argnums=(0, 2))
    def sample_unsafe(self, rng, n):
        """
        Generates n random samples from the unsafe state space.
        
        The function samples states from:
        - Random points within unsafe regions
        - Points on the boundaries of unsafe regions
        - Corner points of unsafe regions
        
        Args:
            rng: Random number generator key
            n: Number of samples to generate per unsafe space
            
        Returns:
            Array of sampled unsafe states
        """
        rngs = jax.random.split(rng, len(self.env.unsafe_spaces))
        per_space_n = n // len(self.env.unsafe_spaces)

        batch = []
        for i in range(len(self.env.unsafe_spaces)):
            x = jax.random.uniform(
                rngs[i],
                (per_space_n, self.env.observation_dim),
                minval=self.env.unsafe_spaces[i].low,
                maxval=self.env.unsafe_spaces[i].high,
            )
            batch.append(x)
            # projecting x onto the boundary of the space
            for j in range(self.env.observation_dim):
                x_j = x.at[:, j].set(self.env.unsafe_spaces[i].low[j])
                batch.append(x_j)
                x_j = x.at[:, j].set(self.env.unsafe_spaces[i].high[j])
                batch.append(x_j)

            # adding the corner points of the space
            low_values = self.env.unsafe_spaces[i].low
            high_values = self.env.unsafe_spaces[i].high
            # corner points
            corner_points = []
            for j in range(2 ** self.env.observation_dim):
                corner_points.append(
                    jnp.array([
                        low_values[k] if (j & (1 << k)) == 0 else high_values[k]
                        for k in range(self.env.observation_dim)
                    ])
                )
            batch.append(jnp.array(corner_points))
        return jnp.concatenate(batch, axis=0)

    @partial(jax.jit, static_argnums=(0, 2))
    def sample_unsafe_complement(self, rng, n):
        """
        Generates states from the complement of the unsafe space.
        
        Samples points from the entire observation space and creates a mask
        indicating which points are outside all unsafe regions.
        
        Args:
            rng: Random number generator key
            n: Number of samples to generate (actual number will be 2*n)
            
        Returns:
            x: Array of sampled states
            mask: Boolean mask indicating which points are outside unsafe regions
        """
        x = jax.random.uniform(
            rng,
            (2 * n, self.env.observation_dim),
            minval=self.env.observation_space.low,
            maxval=self.env.observation_space.high,
        )

        mask = jnp.zeros(2 * n, dtype=jnp.bool_)
        for unsafe_space in self.env.unsafe_spaces:
            b_low = jnp.expand_dims(unsafe_space.low, axis=0)
            b_high = jnp.expand_dims(unsafe_space.high, axis=0)
            contains = jnp.logical_and(
                jnp.all(x >= b_low, axis=1), jnp.all(x <= b_high, axis=1)
            )
            mask = jnp.logical_or(
                mask,
                contains,
            )

        mask = jnp.logical_not(mask)

        return x, mask


    @partial(jax.jit, static_argnums=(0, 2))
    def sample_target(self, rng, n, limit_ratio=0.8):
        """
        Generates n random samples from the target state space.
        
        The function samples states from:
        - Random points within target regions (optionally scaled by limit_ratio)
        - Points on the boundaries of target regions
        - Corner points of target regions
        
        Args:
            rng: Random number generator key
            n: Number of samples to generate per target space
            limit_ratio: Scale factor to control how much of the target space to sample from
                         (values < 1.0 focus sampling toward the center of the region)
            
        Returns:
            Array of sampled target states
        """
        rngs = jax.random.split(rng, len(self.env.target_spaces))
        per_space_n = n // len(self.env.target_spaces)

        batch = []
        for i in range(len(self.env.target_spaces)):
            x = jax.random.uniform(
                rngs[i],
                (per_space_n, self.env.observation_dim),
                minval=self.env.target_spaces[i].low * limit_ratio,
                maxval=self.env.target_spaces[i].high * limit_ratio,
            )
            batch.append(x)
            # projecting x onto the boundary of the space
            for j in range(self.env.observation_dim):
                x_j = x.at[:, j].set(self.env.target_spaces[i].low[j])
                batch.append(x_j)
                x_j = x.at[:, j].set(self.env.target_spaces[i].high[j])
                batch.append(x_j)

            # adding the corner points of the space
            low_values = self.env.target_spaces[i].low
            high_values = self.env.target_spaces[i].high
            # corner points
            corner_points = []
            for j in range(2 ** self.env.observation_dim):
                corner_points.append(
                    jnp.array([
                        low_values[k] if (j & (1 << k)) == 0 else high_values[k]
                        for k in range(self.env.observation_dim)
                    ])
                )
            batch.append(jnp.array(corner_points))
        return jnp.concatenate(batch, axis=0)

    def sample_obs(self, rng, n):
        """
        Generates n random samples from the observation state space.
        
        The function samples states from:
        - Random points within the observation space
        - Points on the boundaries of the observation space
        - Corner points of the observation space
        
        Args:
            rng: Random number generator key
            n: Number of samples to generate
            
        Returns:
            Array of sampled observation states
        """
        batch = []
        x = jax.random.uniform(
            rng,
            (n, self.env.observation_dim),
            minval=self.env.observation_space.low,
            maxval=self.env.observation_space.high,
        )
        batch.append(x)
        # projecting x onto the boundary of the space
        for j in range(self.env.observation_dim):
            x_j = x.at[:, j].set(self.env.observation_space.low[j])
            batch.append(x_j)
            x_j = x.at[:, j].set(self.env.observation_space.high[j])
            batch.append(x_j)

        # adding the corner points of the space
        low_values = self.env.observation_space.low
        high_values = self.env.observation_space.high
        # corner points
        corner_points = []
        for j in range(2 ** self.env.observation_dim):
            corner_points.append(
                jnp.array([
                    low_values[k] if (j & (1 << k)) == 0 else high_values[k]
                    for k in range(self.env.observation_dim)
                ])
            )
        batch.append(jnp.array(corner_points))
        return jnp.concatenate(batch, axis=0)

    @partial(jax.jit, static_argnums=(0, 1))
    def train_step(self, omega, v_state, p_state, state, rng, current_delta):
        """
        Performs a single training step with sampled states.
        
        Args:
            omega: Number of samples to generate for each state category (init, unsafe, target)
            v_state: Current state of the certificate (value) network
            p_state: Current state of the controller (policy) network
            state: States to train on
            rng: Random number generator key
            current_delta: Current delta value for Lipschitz error term
            
        Returns:
            Updated networks and loss metrics
        """
        rngs = jax.random.split(rng, 7)
        obs_samples = self.sample_obs(rngs[0], omega)
        init_samples = self.sample_init(rngs[1], omega)
        unsafe_samples = self.sample_unsafe(rngs[2], omega)
        if self.improved_loss:
            target_samples = self.sample_target(rngs[3], omega, limit_ratio=0.8)
        else:
            target_samples = self.sample_target(rngs[3], omega, limit_ratio=1.0)
        unsafe_complement_samples, unsafe_complement_mask = self.sample_unsafe_complement(rngs[5], omega)

        # Adds a bit of randomization to the grid
        s_random = jax.random.uniform(rngs[4], state.shape, minval=-0.5, maxval=0.5)
        state = state + current_delta * s_random

        def loss_fn(l_params, p_params, state):
            """
            Computes the loss function based on the specification type.
            
            Args:
                l_params: Parameters of the certificate (value) network
                p_params: Parameters of the controller (policy) network
                state: States to compute loss on
                
            Returns:
                Total loss value combining all components based on specification
            """
            loss = 0
            l = v_state.apply_fn(l_params, state)
            a = p_state.apply_fn(p_params, state)

            if self.estimate_expected_via_ibp:
                pmass, batched_grid_lb, batched_grid_ub = self._cached_pmass_grid
                exp_l_next = compute_expected_l(
                    self.env,
                    self.v_ibp.apply,
                    l_params,
                    state,
                    a,
                    pmass,
                    batched_grid_lb,
                    batched_grid_ub,
                )
            else:
                s_next = self.env.v_next(state, a)
                s_next = jnp.expand_dims(
                    s_next, axis=1
                )  # broadcast dim 1 with random noise
                noise = triangular(
                    rngs[6], (s_next.shape[0], 16, self.env.observation_dim)
                )
                noise = noise * self.env.noise
                s_next_random = s_next + noise
                l_next_fn = jax.vmap(v_state.apply_fn, in_axes=(None, 0))
                l_next = l_next_fn(l_params, s_next_random)
                exp_l_next = jnp.mean(l_next, axis=1)

            exp_l_next = exp_l_next.flatten()
            l = l.flatten()
            violations = (exp_l_next >= l).astype(jnp.float32)
            violations = jnp.mean(violations)
            if self.improved_loss and not self.init_with_static:
                steps = (self.env.observation_space.high - self.env.observation_space.low) / self.grid_size
                if self.norm == "l1":
                    delta = 0.5 * jnp.sum(steps)
                    K_p = lipschitz_l1_jax(p_params, obs_normalization=self.obs_normalization)
                    K_l = lipschitz_l1_jax(l_params)
                    K_f = self.env.lipschitz_constant
                    lipschitz_k = K_l * K_f * (1 + K_p) + K_l
                else:
                    delta = 0.5 * jnp.max(steps)
                    K_p = lipschitz_linf_jax(p_params, obs_normalization=self.obs_normalization)
                    K_l = jnp.maximum(lipschitz_linf_jax(l_params), self.v_lip)
                    K_p = K_p * self.lip_lambda_p
                    K_l = K_l * self.lip_lambda_l

                    K_f = self.env.lipschitz_constant_linf
                    lipschitz_k = K_l * K_f * jnp.maximum(1, K_p) + K_l

                if self.spec in ["reach_avoid", "safety", "reachability"]:
                    l_at_init = v_state.apply_fn(l_params, init_samples)
                    l_at_unsafe = v_state.apply_fn(l_params, unsafe_samples)
                    l_at_target = v_state.apply_fn(l_params, target_samples)
                    l_at_obs = v_state.apply_fn(l_params, obs_samples)

                    loss_ra = 0

                    min_at_target = jnp.min(l_at_target)
                    min_at_init = jnp.min(l_at_init)
                    min_at_unsafe = jnp.min(l_at_unsafe)
                    min_at_obs = jnp.min(l_at_obs)
                    max_at_init = jnp.max(l_at_init)
                    l_min = jnp.min(l)

                    min_ = jnp.minimum(jnp.minimum(jnp.minimum(jnp.minimum(min_at_init,
                                                                           min_at_unsafe),
                                                               min_at_obs),
                                                   min_at_target),
                                       l_min) - 1e-6
                    sc_ = (max_at_init - min_ + 1e-6)
                    l_at_init = (l_at_init - min_) / sc_
                    l_at_unsafe = (l_at_unsafe - min_) / sc_
                    l_at_obs = (l_at_obs - min_) / sc_

                    l = (l - min_) / sc_
                    lipschitz_k = lipschitz_k / sc_
                    exp_l_next = (exp_l_next - min_) / sc_

                    min_at_target = (min_at_target - min_) / sc_
                    loss_ra += jnp.mean(jnp.maximum(min_at_target - l_at_init, 0))
                    loss_ra += jnp.mean(jnp.maximum(min_at_target - l_at_unsafe, 0))
                    loss_ra += jnp.mean(jnp.maximum(min_at_target - l_at_obs, 0))

                    if self.spec != 'reachability':
                        all_at_unsafe = l_at_unsafe - (1 / (1 - self.prob)) - 1e-6
                        loss_ra += jnp.mean(-jnp.minimum(all_at_unsafe, 0))

                    loss += loss_ra
                else:
                    if self.norm == "l1":
                        lip_f = self.env.lipschitz_constant
                        lip_norm = lip_f * (K_p + 1) * current_delta
                        t = p_state.apply_fn(p_params, unsafe_complement_samples)
                        next_det_grids = self.env.v_next(unsafe_complement_samples, t)
                        next_dis = jnp.sum(jnp.abs(next_det_grids - unsafe_complement_samples), axis=1)
                        next_max_norm = jnp.max(jnp.where(unsafe_complement_mask, next_dis, 0))
                        t_unsafe = p_state.apply_fn(p_params, unsafe_samples)
                        next_det_unsafe = self.env.v_next(unsafe_samples, t_unsafe)
                        next_dis_unsafe = jnp.sum(jnp.abs(next_det_unsafe - unsafe_samples), axis=1)
                        next_max_norm = jnp.maximum(next_max_norm, jnp.max(next_dis_unsafe))
                        noise_norm = jnp.sum(jnp.abs(self.env.noise))
                        current_big_delta = next_max_norm + noise_norm + lip_norm
                    elif self.norm == "linf":
                        lip_f = self.env.lipschitz_constant_linf
                        lip_norm = lip_f * jnp.maximum(K_p, 1) * current_delta
                        t = p_state.apply_fn(p_params, unsafe_complement_samples)
                        next_det_grids = self.env.v_next(unsafe_complement_samples, t)
                        next_dis = jnp.max(jnp.abs(next_det_grids - unsafe_complement_samples), axis=1)
                        next_max_norm = jnp.max(jnp.where(unsafe_complement_mask, next_dis, 0))
                        t_unsafe = p_state.apply_fn(p_params, unsafe_samples)
                        next_det_unsafe = self.env.v_next(unsafe_samples, t_unsafe)
                        next_dis_unsafe = jnp.max(jnp.abs(next_det_unsafe - unsafe_samples), axis=1)
                        next_max_norm = jnp.maximum(next_max_norm, jnp.max(next_dis_unsafe))
                        noise_norm = jnp.max(jnp.abs(self.env.noise))
                        current_big_delta = next_max_norm + noise_norm + lip_norm
                    else:
                        raise ValueError("Unknown norm")

                    l_at_init = v_state.apply_fn(l_params, init_samples)
                    l_at_unsafe = v_state.apply_fn(l_params, unsafe_samples)
                    l_at_target = v_state.apply_fn(l_params, target_samples)
                    l_at_obs = v_state.apply_fn(l_params, obs_samples)
                    l_at_unsafe_comp = v_state.apply_fn(l_params, unsafe_complement_samples)
                    l_at_unsafe_comp = jnp.reshape(l_at_unsafe_comp, -1)
                    l_at_unsafe_comp = jnp.where(unsafe_complement_mask, l_at_unsafe_comp, jnp.float64(1e18))
                    min_at_unsafe_comp = jnp.max(l_at_unsafe_comp[jnp.argsort(l_at_unsafe_comp)[:32]])

                    loss_st = 0

                    min_at_target = jnp.min(l_at_target)
                    min_at_init = jnp.min(l_at_init)
                    min_at_unsafe = jnp.min(l_at_unsafe)
                    min_at_obs = jnp.min(l_at_obs)
                    max_at_target = jnp.max(l_at_target)
                    l_min = jnp.min(l)

                    min_ = jnp.minimum(jnp.minimum(jnp.minimum(jnp.minimum(min_at_init,
                                                                           min_at_unsafe),
                                                               min_at_obs),
                                                   min_at_target),
                                       l_min) - 1e-6
                    sc_ = (max_at_target - min_ + 1e-6)
                    l_at_init = (l_at_init - min_) / sc_
                    l_at_unsafe = (l_at_unsafe - min_) / sc_
                    l_at_obs = (l_at_obs - min_) / sc_
                    l = (l - min_) / sc_
                    exp_l_next = (exp_l_next - min_) / sc_

                    lipschitz_k = lipschitz_k / sc_

                    min_at_target = (min_at_target - min_) / sc_
                    min_at_unsafe_comp = (min_at_unsafe_comp - min_) / sc_
                    loss_st += jnp.mean(jnp.maximum(min_at_target - l_at_init, 0))
                    loss_st += jnp.mean(jnp.maximum(min_at_target - l_at_unsafe, 0))
                    loss_st += jnp.mean(jnp.maximum(min_at_target - l_at_obs, 0))
                    loss_st += jnp.mean(jnp.maximum(min_at_target - min_at_unsafe_comp, 0))

                    unsafe_lb = 1 + K_l * current_big_delta / sc_ + self.small_delta
                    all_at_unsafe = l_at_unsafe - unsafe_lb - 1e-6
                    loss_st += jnp.mean(-jnp.minimum(all_at_unsafe, 0))

                    loss += loss_st

                dec_loss = martingale_loss(l, exp_l_next, lipschitz_k * delta)
                loss += dec_loss

            else:
                dec_loss = martingale_loss(l, exp_l_next, self.eps)
                loss = dec_loss
                if self.norm == "l1":
                    K_l = lipschitz_l1_jax(l_params)
                    K_p = lipschitz_l1_jax(p_params)
                elif self.norm == "linf":
                    K_p = lipschitz_linf_jax(p_params, obs_normalization=self.obs_normalization)
                    K_l = lipschitz_linf_jax(l_params)
                else:
                    raise ValueError("Unknown norm")
                lip_loss_l = jnp.maximum(K_l - self.v_lip, 0)
                lip_loss_p = jnp.maximum(K_p - self.p_lip, 0)
                loss += self.lip_lambda * (lip_loss_l + lip_loss_p)

                l_at_init = v_state.apply_fn(l_params, init_samples)
                l_at_unsafe = v_state.apply_fn(l_params, unsafe_samples)
                l_at_target = v_state.apply_fn(l_params, target_samples)

                if self.spec in ["reach_avoid", "safety", "reachability"]:
                    # Zero at zero
                    s_zero = jnp.zeros(self.env.observation_dim)
                    l_at_zero = v_state.apply_fn(l_params, s_zero)
                    loss += jnp.sum(
                        jnp.maximum(jnp.abs(l_at_zero), 0.3)
                    )  # min to an eps of 0.3

                    max_at_init = jnp.max(l_at_init)
                    min_at_unsafe = jnp.min(l_at_unsafe)
                    if self.spec != "reachability":
                        # Maximize this term to at least 1/(1-reach prob)
                        loss += -jnp.minimum(min_at_unsafe, 1 / (1 - self.prob))

                    # Minimize the max at init to below 1
                    loss += jnp.maximum(max_at_init, 1)

                    # Global minimum should be inside target
                    min_at_target = jnp.min(l_at_target)
                    min_at_init = jnp.min(l_at_init)
                    min_at_unsafe = jnp.min(l_at_unsafe)
                    loss += jnp.maximum(min_at_target - min_at_init, 0)
                    loss += jnp.maximum(min_at_target - min_at_unsafe, 0)
                else:
                    l_at_unsafe = v_state.apply_fn(l_params, unsafe_samples)
                    l_at_target = v_state.apply_fn(l_params, target_samples)
                    loss += 0.1 / jnp.minimum(K_l, 0.1)

                    lip_f = self.env.lipschitz_constant
                    lip_p = lipschitz_l1_jax(p_params)
                    lip_norm = lip_f * (lip_p + 1) * current_delta
                    t = p_state.apply_fn(p_params, unsafe_complement_samples)
                    next_det_grids = self.env.v_next(unsafe_complement_samples, t)
                    next_dis = jnp.sum(jnp.abs(next_det_grids - unsafe_complement_samples), axis=1)
                    next_max_norm = jnp.max(jnp.where(unsafe_complement_mask, next_dis, 0))
                    t_unsafe = p_state.apply_fn(p_params, unsafe_samples)
                    next_det_unsafe = self.env.v_next(unsafe_samples, t_unsafe)
                    next_dis_unsafe = jnp.sum(jnp.abs(next_det_unsafe - unsafe_samples), axis=1)
                    next_max_norm = jnp.maximum(next_max_norm, jnp.max(next_dis_unsafe))
                    noise_norm = jnp.sum(jnp.abs(self.env.noise))
                    current_big_delta = next_max_norm + noise_norm + lip_norm

                    min_at_unsafe = jnp.min(l_at_unsafe)
                    unsafe_lb = 1 +  K_l * current_big_delta + self.small_delta
                    loss += -jnp.minimum(min_at_unsafe - unsafe_lb, 0)

                    l_at_unsafe_comp = v_state.apply_fn(l_params, unsafe_complement_samples)
                    l_at_unsafe_comp = jnp.reshape(l_at_unsafe_comp, -1)
                    l_at_unsafe_comp = jnp.where(unsafe_complement_mask, l_at_unsafe_comp, jnp.float64(1e18))
                    min_at_unsafe_comp = jnp.max(l_at_unsafe_comp[jnp.argsort(l_at_unsafe_comp)[:40]])
                    max_at_target = jnp.max(l_at_target)
                    loss += jnp.maximum(max_at_target - 1, 0)
                    min_at_target = jnp.min(l_at_target)
                    min_at_unsafe = jnp.min(l_at_unsafe)
                    loss += jnp.maximum(min_at_target - min_at_unsafe_comp, 0)
                    loss += jnp.maximum(max_at_target - min_at_unsafe, 0)
                    loss -= jnp.minimum(min_at_target, 0)

            kp_param = K_p
            return loss, (dec_loss, violations, kp_param)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0, 1))
        (loss, (dec_loss, violations, kp_param)), (l_grad, p_grad) = grad_fn(
            v_state.params, p_state.params, state
        )

        v_state = v_state.apply_gradients(grads=l_grad)
        p_state = p_state.apply_gradients(grads=p_grad)
        metrics = {"loss": loss, "dec_loss": dec_loss, "train_violations": violations, "kp_param": kp_param, "omega": omega}
        return v_state, p_state, metrics


    def train_epoch(
            self, train_ds, current_delta=0, train_v=True, train_p=True, omega=0
    ):
        """
        Trains the networks for one epoch.
        
        Args:
            train_ds: The training dataset containing counterexamples
            current_delta: Current delta value for Lipschitz error calculations
            train_v: Whether to train the certificate (value) network
            train_p: Whether to train the controller (policy) network
            omega: Number of samples to use for loss computation
            
        Returns:
            Dictionary containing loss values and other training metrics
        """
        current_delta = jnp.float32(current_delta)
        batch_metrics = []
        for state in train_ds.as_numpy_iterator():
            state = jnp.array(state)
            self.rng, rng = jax.random.split(self.rng, 2)

            new_v_state, new_p_state, metrics = self.train_step(
                int(omega), self.v_state, self.p_state, state, rng, current_delta
            )
            if train_p:
                self.p_state = new_p_state
            if train_v:
                self.v_state = new_v_state
            batch_metrics.append(metrics)

        # compute mean of metrics across each batch in epoch.
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {
            k: np.mean([metrics[k] for metrics in batch_metrics_np])
            for k in batch_metrics_np[0]
        }

        return epoch_metrics_np

    def save(self, filename):
        """
        Saves the neural networks parameters to a file.
        
        Args:
            filename: Path where the model parameters will be saved
        """
        jax_save(
            {"policy": self.p_state, "value": self.c_state, "martingale": self.v_state},
            filename,
        )
        try:
            if filename.endswith(".jax"):
                jax_save(self.obs_normalization, filename=filename.replace(".jax", "_obs_normalization.jax"))
            else:
                jax_save(self.obs_normalization, filename=filename + "_obs_normalization.jax")
        except Exception as e:
            print(e)

    def load(self, filename, force_load_all=True):
        """
        Loads the neural networks parameters from a file.
        
        Args:
            filename: Path where the model parameters are stored
            force_load_all: If True, loads all parameters. If False, loads only
                            the networks specified by the task type
                            
        Returns:
            True if the load was successful, False otherwise
        """
        try:
            params = jax_load(
                {
                    "policy": self.p_state,
                    "value": self.c_state,
                    "martingale": self.v_state,
                },
                filename,
            )
            self.p_state = params["policy"]
            self.v_state = params["martingale"]
            self.c_state = params["value"]
        except Exception as e:
            if force_load_all:
                try:
                    tmp_state = create_train_state(
                        self.p_net,
                        jax.random.PRNGKey(2),
                        self.env.observation_dim,
                        self.p_lr,
                        use_brax=self.use_brax,
                        out_dim=self.env.action_space.shape[0],
                        obs_normalization=self.obs_normalization,
                        opt='adam'
                    )
                    params = {
                        "policy": tmp_state,
                        "value": self.c_state,
                        "martingale": self.v_state
                    }
                    lrs = {'policy': self.p_lr, 'value': self.c_lr, 'martingale': self.v_lr}
                    params = jax_load(params, filename, replace_with_adamw=self.opt == 'adamw', lrs=lrs)
                    self.p_state = params["policy"]
                    self.c_state = params["value"]
                    self.v_state = params["martingale"]
                except Exception:
                    raise e
            # Legacy load
            try:
                params = {"policy": self.p_state, "value": self.c_state}
                params = jax_load(params, filename)
                self.p_state = params["policy"]
                self.c_state = params["value"]
            except Exception:
                try:
                    params = {"policy": self.p_state}
                    params = jax_load(params, filename)
                    self.p_state = params["policy"]
                except Exception:
                    tmp_state = create_train_state(
                        self.p_net,
                        jax.random.PRNGKey(2),
                        self.env.observation_dim,
                        self.p_lr,
                        use_brax=self.use_brax,
                        out_dim=self.env.action_space.shape[0],
                        obs_normalization=self.obs_normalization,
                        opt='adam'
                    )
                    params = {"policy": tmp_state}
                    lrs = {'policy': self.p_lr, 'value': self.c_lr, 'martingale': self.v_lr}
                    params = jax_load(params, filename, replace_with_adamw=self.opt == 'adamw', lrs=lrs)
                    self.p_state = params["policy"]

        self.p_init_params = deepcopy(self.p_state.params['params'])
        if self.policy_type == "sac":
            self.obs_normalization: running_statistics.RunningStatisticsState = running_statistics.init_state(
                specs.Array((self.env.observation_dim,), jnp.dtype('float32')))
            try:
                if filename.endswith(".jax"):
                    self.obs_normalization = jax_load(self.obs_normalization, filename.replace(".jax", "_obs_normalization.jax"))
                else:
                    self.obs_normalization = jax_load(self.obs_normalization, filename + "_obs_normalization.jax")
            except Exception as e:
                print(e)
            self.load_from_brax((self.obs_normalization, self.p_state.params))

    def load_from_brax(self, params):
        """
        Loads policy network parameters from Brax format.
        
        Args:
            params: Parameters in Brax format to be loaded into the policy network
        """
        self.p_state = create_train_state(
            self.p_net,
            jax.random.PRNGKey(2),
            self.env.observation_dim,
            self.p_lr,
            use_brax=self.use_brax,
            out_dim=self.env.action_space.shape[0],
            obs_normalization=params[0]
        )
        di = {}
        for (sk, _), (_, pv) in zip(self.p_state.params['params'].items(), params[1]['params'].items()):
            di[sk] = pv
        params[1]['params'] = di
        self.p_state.params['params'] = di
        self.obs_normalization = params[0]
        self.p_init_params = deepcopy(self.p_state.params)

    def update_tmodels(self):
        """
        Updates the torch model (TMLP) parameters with the current network parameters.
        
        This is used when calculating local Lipschitz constants.
        """
        set_tnet_params(self.p_state.params, self.p_tnet)
        set_tnet_params(self.v_state.params, self.v_tnet)
