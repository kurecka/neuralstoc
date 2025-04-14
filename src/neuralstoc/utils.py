from typing import Sequence, Optional, Any

import jax
import jax.numpy as jnp
import flax.linen as nn
import torch
import torch.nn as tnn
from auto_LiRPA.jacobian import JacobianOP, GradNorm
from flax.training import train_state  # Useful dataclass to keep train state
import flax
import numpy as np  # Ordinary NumPy
import optax  # Optimizers
from functools import partial
from gym import spaces
import matplotlib.pyplot as plt
import seaborn as sns
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from brax.training import distribution


def pretty_time(elapsed):
    """
    Converts a float (seconds) into a pretty string, i.e., "1023983" -> "2h 34min"
    """
    if elapsed > 60 * 60:
        h = int(elapsed // (60 * 60))
        mins = int((elapsed // 60) % 60)
        return f"{h}h {mins:02d} min"
    elif elapsed > 60:
        mins = elapsed // 60
        secs = int(elapsed) % 60
        return f"{mins:0.0f}min {secs}s"
    elif elapsed < 1:
        return f"{elapsed * 1000:0.1f}ms"
    else:
        return f"{elapsed:0.1f}s"


def pretty_number(number):
    """
    Converts a large number into SI representation (3409230384-> 3.4G)
    """
    if number >= 1.0e9:
        return f"{number / 1e9:0.3g}G"
    elif number >= 1.0e6:
        return f"{number / 1e6:0.3g}M"
    elif number >= 1.0e3:
        return f"{number / 1e3:0.3g}k"
    else:
        return number


def v_contains(box, states):
    """
    Computes a bool-array indicating whether states are inside the box.
    NumPy version (for Jax version see jv_contains)
    """
    b_low = np.expand_dims(box.low, axis=0)
    b_high = np.expand_dims(box.high, axis=0)
    contains = np.logical_and(
        np.all(states >= b_low, axis=1), np.all(states <= b_high, axis=1)
    )
    return contains


def jv_contains(box, states):
    """
    Computes a bool-array indicating whether states are inside the box.
    JAX version (for NumPy version see jv_contains)
    """
    b_low = jnp.expand_dims(box.low, axis=0)
    b_high = jnp.expand_dims(box.high, axis=0)
    contains = np.logical_and(
        jnp.all(states >= b_low, axis=1), jnp.all(states <= b_high, axis=1)
    )
    return contains


def v_intersect(box, lb, ub):
    """
    Computes a bool-array indicating whether (lb,ub) boxes overlap/intersect with the box.
    NumPy version (for Jax version see jv_intersect)
    """
    b_low = np.expand_dims(box.low, axis=0)
    b_high = np.expand_dims(box.high, axis=0)
    contain_lb = np.logical_and(lb >= b_low, lb <= b_high)
    contain_ub = np.logical_and(ub >= b_low, ub <= b_high)
    contains_any = np.all(np.logical_or(contain_lb, contain_ub), axis=1)

    return contains_any


def jv_intersect(box, lb, ub):
    """
    Computes a bool-array indicating whether (lb,ub) boxes overlap/intersect with the box.
    Jax version (for numpy version see v_intersect)
    """
    b_low = jnp.expand_dims(box.low, axis=0)
    b_high = jnp.expand_dims(box.high, axis=0)
    contain_lb = jnp.logical_and(lb >= b_low, lb <= b_high)
    contain_ub = jnp.logical_and(ub >= b_low, ub <= b_high)
    # every axis much either lb or ub contain
    contains_any = jnp.all(jnp.logical_or(contain_lb, contain_ub), axis=1)

    return contains_any


def clip_and_filter_spaces(obs_space, space_list):
    """
    Projects the list of Box spaces into the obs_space and returns a list of projected Box spaces
    """
    new_space_list = []
    for space in space_list:
        new_space = spaces.Box(
            low=np.clip(space.low, obs_space.low, obs_space.high),
            high=np.clip(space.high, obs_space.low, obs_space.high),
        )
        volume = np.prod(new_space.high - new_space.low)
        if volume > 0:
            new_space_list.append(new_space)
    return new_space_list


def make_unsafe_spaces(obs_space, unsafe_bounds):
    """
    Creates a list of Box spaces that represent the set complement of the obs_space
    minus the sets inside the unsafe bounds
    """
    unsafe_spaces = []
    dims = obs_space.shape[0]
    for i in range(dims):
        low = np.array(obs_space.low)
        high = np.array(obs_space.high)
        high[i] = -unsafe_bounds[i]
        if not np.allclose(low, high):
            unsafe_spaces.append(spaces.Box(low=low, high=high, dtype=np.float32))

        high = np.array(obs_space.high)
        low = np.array(obs_space.low)
        low[i] = unsafe_bounds[i]
        if not np.allclose(low, high):
            unsafe_spaces.append(spaces.Box(low=low, high=high, dtype=np.float32))
    return unsafe_spaces


def make_corner_spaces(obs_space, unsafe_bounds):
    """
    Creates a list of Box spaces that represent the corners of the obs_space.
    Size of the corners is such that they are greater than the unsafe bounds
    """
    unsafe_spaces = []
    dims = obs_space.shape[0]
    for i in range(dims):
        low = np.array(obs_space.low)
        high = np.array(obs_space.high)
        high[i] = low[i] + unsafe_bounds[i]
        if not np.allclose(low, high):
            unsafe_spaces.append(spaces.Box(low=low, high=high, dtype=np.float32))

        high = np.array(obs_space.high)
        low = np.array(obs_space.low)
        low[i] = high[i] - unsafe_bounds[i]
        if not np.allclose(low, high):
            unsafe_spaces.append(spaces.Box(low=low, high=high, dtype=np.float32))
    return unsafe_spaces


def enlarge_space(space, bound, limit_space=None):
    """
    Enlarges a given space by the values of bound (multi-dim array).
    If a limit_space is given, the resulting enlarged space will be projected into the limit_space
    """
    new_space = spaces.Box(low=space.low - bound, high=space.high + bound)
    if limit_space is not None:
        new_space = spaces.Box(
            low=np.clip(new_space.low, limit_space.low, limit_space.high),
            high=np.clip(new_space.high, limit_space.low, limit_space.high),
        )
    return new_space


@jax.jit
def clip_grad_norm(grad, max_norm):
    """
    Clips the gradient norm to a maximum value.
    If the norm exceeds max_norm, scales the gradient down proportionally.
    """
    norm = jnp.linalg.norm(
        jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, grad))
    )
    factor = jnp.minimum(max_norm, max_norm / (norm + 1e-6))
    return jax.tree_map((lambda x: x * factor), grad)


def contained_in_any(spaces, state):
    """
    Returns True if state is contained in at least of of the Box spaces, False otherwise
    """
    for space in spaces:
        if space.contains(state):
            return True
    return False


def triangular(rng_key, shape):
    """
    Samples from a triangular distribution with mean 0 and range (-1,+1) with shape shape
    """
    U = jax.random.uniform(rng_key, shape=shape)
    p1 = -1 + jnp.sqrt(2 * U)
    p2 = 1 - jnp.sqrt((1 - U) * 2)
    return jnp.where(U <= 0.5, p1, p2)


class TMLP(tnn.Module):
    """
    A PyTorch implementation of a multi-layer perceptron.
    Similar to MLP but implemented in PyTorch for compatibility with auto_LiRPA.
    """
    def __init__(self, input_size, hidden, activation='relu', softplus_output=False):
        super().__init__()
        # self.flatten = tnn.Flatten()
        self.seq = tnn.Sequential()
        self.seq.append(tnn.Linear(input_size, hidden[0]))
        for i in range(len(hidden) - 1):
            if activation == 'relu':
                self.seq.append(tnn.ReLU())
            else:
                self.seq.append(tnn.Tanh())
            self.seq.append(tnn.Linear(hidden[i], hidden[i + 1]))
        if softplus_output:
            self.seq.append(tnn.Softplus())

    def forward(self, x):
        # x = self.flatten(x)
        x = self.seq(x)
        return x


class MLP(nn.Module):
    """
    A standard multi-layer perceptron implementation in JAX.
    """
    features: Sequence[int]
    activation: str = "relu"
    softplus_output: bool = False
    rng: jnp.ndarray = jax.random.PRNGKey(7)

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat, kernel_init=jax.nn.initializers.glorot_uniform())(x)
            if self.activation == "relu":
                x = nn.relu(x)
            else:
                x = nn.tanh(x)
        x = nn.Dense(
            self.features[-1], kernel_init=jax.nn.initializers.glorot_uniform()
        )(x)
        if self.softplus_output:
            x = jax.nn.softplus(x)
        return x


def martingale_loss(l, l_next, eps):
    """
    Computes the martingale loss between current and next state values.
    """
    diff = l_next - l
    return jnp.mean(jnp.maximum(diff + eps, 0))
    # return jnp.sum(jnp.maximum(diff + eps, 0))


def jax_save(params, filename):
    """Saves parameters into a file"""
    bytes_v = flax.serialization.to_bytes(params)
    with open(filename, "wb") as f:
        f.write(bytes_v)



def jax_load(params, filename, replace_with_adamw=False, lrs=None):
    """Loads parameters from a file"""
    with open(filename, "rb") as f:
        bytes_v = f.read()
    params = flax.serialization.from_bytes(params, bytes_v)
    if replace_with_adamw:
        for key in params:
            tx = optax.adamw(lrs[key])
            params[key] = train_state.TrainState.create(apply_fn=params[key].apply_fn,
                                                        params=params[key].params,
                                                        tx=tx)

    return params


def lipschitz_l1_jax(params, obs_normalization=None):
    """
    Computes the L1 Lipschitz constant of a JAX model.
    Optionally applies observation normalization if provided.
    """
    lipschitz_l1 = 1
    sum_axis = 1  # flax dense is transposed
    for i, (k, v) in enumerate(params["params"].items()):
        lipschitz_l1 *= jnp.max(jnp.sum(jnp.abs(v["kernel"]), axis=sum_axis))
    if obs_normalization is not None:
        lipschitz_l1 *= jnp.abs(1/obs_normalization.std).sum()

    return lipschitz_l1


def lipschitz_linf_jax(params, obs_normalization=None):
    """
    Computes the L-infinity Lipschitz constant of a JAX model.
    Optionally applies observation normalization if provided.
    """
    lipschitz_linf = 1
    sum_axis = 0  # flax dense is transposed
    for i, (k, v) in enumerate(params["params"].items()):
        lipschitz_linf *= jnp.max(jnp.sum(jnp.abs(v["kernel"]), axis=sum_axis))
    if obs_normalization is not None:
        lipschitz_linf *= jnp.max(1 / obs_normalization.std)
    return lipschitz_linf


def compute_local_lipschitz(tnet, x0, eps, out_dim=1, obs_normalization=None):
    """
    Computes the local Lipschitz constant around a given point x0.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print('device: ', device, 'shape: ', x0.shape, 'eps: ', eps, 'out_dim: ', out_dim)
    mean_ = None
    std_ = None
    if obs_normalization is not None:
        mean_ = np.asarray(obs_normalization.mean)
        std_ = np.asarray(obs_normalization.std)
        mean_ = torch.from_numpy(mean_).to(device).float()
        std_ = torch.from_numpy(std_).to(device).float()
    x0 = torch.from_numpy(x0).to(device).float()
    tnet = tnet.to(device)

    default_bound_opts = {
        'conv_mode': 'patches',
        'sparse_intermediate_bounds': False,
        'sparse_conv_intermediate_bounds': False,
        'sparse_intermediate_bounds_with_ibp': False,
        'sparse_features_alpha': False,
        'sparse_spec_alpha': False,
        'minimum_sparsity': 0.0,
        'enable_opt_interm_bounds': True,
        'crown_batch_size': np.inf,
        'forward_refinement': True,
        'dynamic_forward': True,
        'forward_max_dim': int(1e9),
        # Do not share alpha for conv layers.
        'use_full_conv_alpha': True,
        'disabled_optimization': [],
        # Threshold for number of unstable neurons for each layer to disable
        #  use_full_conv_alpha.
        'use_full_conv_alpha_thresh': 512,
        'verbosity': 0,
        'optimize_graph': {'optimizer': None},
    }

    class LocalLipschitzWrapper(tnn.Module):
        def __init__(self, model, mask):
            super().__init__()
            self.model = model
            self.mask = mask
            self.grad_norm = GradNorm(norm=1)

        def forward(self, x):
            if obs_normalization is not None:
                mean = mean_.to(x.device)
                std = std_.to(x.device)
                x = (x - mean) / std
            # y = self.model(x.permute(1, 0))
            y = self.model(x)
            self.mask = self.mask.to(y.device)
            y_selected = y.matmul(self.mask)
            jacobian = JacobianOP.apply(y_selected, x)
            lipschitz = self.grad_norm(jacobian)
            return lipschitz

    # mask = torch.zeros(out_dim, 1, device=device)
    mask = torch.zeros(out_dim, 1)
    mask[0, 0] = 1
    x0.requires_grad_(True)
    sample = (x0[:1]).to(device)
    # # reorder x0 as pytorch batch shape
    # x0 = x0.permute(1, 0)
    # sample = sample.permute(1, 0)
    model = BoundedModule(LocalLipschitzWrapper(tnet, mask=mask), (sample), device=device, bound_opts=default_bound_opts)
    if obs_normalization is not None:
        mean_ = mean_.to(device)
        std_ = std_.to(device)
        sample_ = (sample - mean_) / std_
    else:
        sample_ = torch.tensor(sample).to(device)
    sample_.requires_grad_(True)
    y = tnet(sample_)
    ret_ori = torch.autograd.grad(y[:, 0].sum(), sample_)[0].abs().flatten(1).sum(dim=-1).view(-1)
    ret_new = model(sample, mask).view(-1)
    assert torch.allclose(ret_ori, ret_new)

    lip_ans = []
    for i in range(len(x0)):
        x = BoundedTensor(x0[i:i+1].to(device), PerturbationLpNorm(norm=np.inf, eps=eps))
        lip = []
        for j in range(mask.shape[0]):
            mask.zero_()
            mask[j, 0] = 1
            ub = model.compute_jacobian_bounds((x, mask), bound_lower=False)[1]
            lip.append(ub)
        lip = torch.max(torch.cat(lip))
        lip_ans.append(lip.item())

    return np.array(lip_ans)


def set_tnet_params(params, tnet):
    """
    Sets the parameters of a PyTorch network from JAX parameters.
    """
    for i, (k, v) in enumerate(params["params"].items()):
        tnet.seq[2 * i].weight = tnn.Parameter(torch.from_numpy(np.array(v["kernel"]).T).float())
        tnet.seq[2 * i].bias = tnn.Parameter(torch.from_numpy(np.array(v["bias"])).float())


def create_train_state(model, rng, in_dim, learning_rate, ema=0, clip_norm=None,
                       use_brax=False, out_dim=None, obs_normalization=None, opt='adamw'):
    """Creates initial `TrainState`."""
    params = model.init(rng, jnp.ones([1, in_dim]))
    if opt == 'adam':
        tx = optax.adam(learning_rate)
    elif opt == 'adamw':
        tx = optax.adamw(learning_rate)
    else:
        raise NotImplemented
    parametric_action_distribution = None
    if use_brax:
        parametric_action_distribution = distribution.NormalTanhDistribution(event_size=out_dim)

    if clip_norm is not None:
        tx = optax.chain(tx, optax.clip_by_global_norm(clip_norm))
    if ema > 0:
        tx = optax.chain(tx, optax.ema(ema))
    if not use_brax:
        apply_fn = model.apply
    else:
        def policy(params, observations, deterministic=True):
            # logits = model.apply(*params, observations)
            if obs_normalization is not None:
                observations = (observations - obs_normalization.mean) / obs_normalization.std
            logits = model.apply(params, observations)
            if deterministic:
                return parametric_action_distribution.mode(logits)
            model.rng, key_sample = jax.random.split(model.rng)
            return parametric_action_distribution.sample(logits, key_sample)
        apply_fn = policy

    return train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=tx)


def get_pmass_grid(env, n):
    """
    Compute the bounds of the sum terms and corresponding probability masses
    for the expectation computation
    """
    dims = len(env.noise_bounds[0])
    grid, steps = [], []
    for i in range(dims):
        samples, step = jnp.linspace(
            env.noise_bounds[0][i],
            env.noise_bounds[1][i],
            n,
            endpoint=False,
            retstep=True,
        )
        grid.append(samples)
        steps.append(step)
    grid_lb = jnp.meshgrid(*grid)
    grid_lb = [x.flatten() for x in grid_lb]
    grid_ub = [grid_lb[i] + steps[i] for i in range(dims)]

    if dims < env.observation_dim:
        # Fill remaining dimensions with 0
        remaining = env.observation_dim - len(env.noise_bounds)
        for i in range(remaining):
            grid_lb.append(jnp.zeros_like(grid_lb[0]))
            grid_ub.append(jnp.zeros_like(grid_lb[0]))
    batched_grid_lb = jnp.stack(grid_lb, axis=1)  # stack on input  dim
    batched_grid_ub = jnp.stack(grid_ub, axis=1)  # stack on input dim
    pmass = env.integrate_noise(grid_lb, grid_ub)
    return pmass, batched_grid_lb, batched_grid_ub


@partial(jax.jit, static_argnums=(0, 1))
def compute_expected_l(
        env, ibb_apply_fn, params, s, a, pmass, pmass_grid_lb, pmass_grid_ub
):
    """
    Compute kernel (jit compiled) that computes an upper bounds on the expected value of L(s next)
    """
    deterministic_s_next = env.v_next(s, a)
    batch_size = s.shape[0]
    ibp_size = pmass_grid_lb.shape[0]
    obs_dim = env.observation_dim

    # Broadcasting happens here, that's why we don't do directly vmap (although it's probably possible somehow)
    deterministic_s_next = deterministic_s_next.reshape((batch_size, 1, obs_dim))
    pmass_grid_lb = pmass_grid_lb.reshape((1, ibp_size, obs_dim))
    pmass_grid_ub = pmass_grid_ub.reshape((1, ibp_size, obs_dim))

    pmass_grid_lb = pmass_grid_lb + deterministic_s_next
    pmass_grid_ub = pmass_grid_ub + deterministic_s_next

    pmass_grid_lb = pmass_grid_lb.reshape((-1, obs_dim))
    pmass_grid_ub = pmass_grid_ub.reshape((-1, obs_dim))
    lb, ub = ibb_apply_fn(params, [pmass_grid_lb, pmass_grid_ub])
    ub = ub.reshape((batch_size, ibp_size))

    pmass = pmass.reshape((1, ibp_size))  # Boradcast to batch size
    exp_terms = pmass * ub
    expected_value = jnp.sum(exp_terms, axis=1)
    return expected_value


def plot_policy(env, policy, filename, rsm=None, title=None):
    """
    Plots the policy's behavior in the environment.
    """
    dims = env.observation_dim

    sns.set()
    fig, ax = plt.subplots(figsize=(6, 6))

    if env.observation_dim == 2:
        if rsm is not None:
            grid, new_steps = [], []
            for i in range(dims):
                samples = jnp.linspace(
                    env.observation_dim.low[i],
                    env.observation_dim.high[i],
                    50,
                    endpoint=False,
                    retstep=True,
                )
                grid.append(samples.flatten())
            grid = jnp.meshgrid(*grid)
            grid = jnp.stack(grid, axis=1)
            l = rsm.apply_fn(rsm.params, grid).flatten()
            l = np.array(l)
            sc = ax.scatter(
                grid[:, 0], grid[:, 1], marker="s", c=l, zorder=1, alpha=0.7
            )
            fig.colorbar(sc)

    n = 50
    rng = jax.random.PRNGKey(3)
    rng, r = jax.random.split(rng)
    r = jax.random.split(r, n)
    state, obs = env.v_reset(r)
    done = jnp.zeros(n, dtype=jnp.bool_)
    total_returns = jnp.zeros(n)
    obs_list = []
    done_list = []
    while not jnp.any(done):
        action = policy.apply_fn(policy.params, obs)
        # rng, r = jax.random.split(rng)
        # action, _ = policy(obs, r)
        rng, r = jax.random.split(rng)
        r = jax.random.split(r, n)
        state, new_obs, reward, new_done = env.v_step(state, action, r)
        total_returns += reward * (1.0 - done)
        done_list.append(done)
        obs_list.append(obs)
        obs, done = new_obs, new_done
    obs_list = jnp.stack(obs_list, 1)
    done_list = jnp.stack(done_list, 1)
    traces = [obs_list[i, jnp.logical_not(done_list[i])] for i in range(n)]

    if title is None:
        title = env.name

    title = (
            title
            + f" ({jnp.mean(total_returns):0.1f} [{jnp.min(total_returns):0.1f},{jnp.max(total_returns):0.1f}])"
    )
    ax.set_title(title)

    terminals_x, terminals_y = [], []
    for i in range(n):
        ax.plot(
            traces[i][:, 0],
            traces[i][:, 1],
            color=sns.color_palette()[0],
            zorder=2,
            alpha=0.15,
        )
        ax.scatter(
            traces[i][:, 0],
            traces[i][:, 1],
            color=sns.color_palette()[0],
            zorder=2,
            marker=".",
            alpha=0.4,
        )
        terminals_x.append(float(traces[i][-1, 0]))
        terminals_y.append(float(traces[i][-1, 1]))
    ax.scatter(terminals_x, terminals_y, color="white", marker="x", zorder=5)
    for init in env.init_spaces:
        x = [
            init.low[0],
            init.high[0],
            init.high[0],
            init.low[0],
            init.low[0],
        ]
        y = [
            init.low[1],
            init.low[1],
            init.high[1],
            init.high[1],
            init.low[1],
        ]
        ax.plot(x, y, color="cyan", alpha=0.5, zorder=7)
    if hasattr(env, "_reward_boxes"):
        for box, rs in env._reward_boxes:
            x = [
                box.low[0],
                box.high[0],
                box.high[0],
                box.low[0],
                box.low[0],
            ]
            y = [
                box.low[1],
                box.low[1],
                box.high[1],
                box.high[1],
                box.low[1],
            ]
            ax.plot(x, y, color="yellow", alpha=0.5, zorder=7)
    for unsafe in env.unsafe_spaces:
        x = [
            unsafe.low[0],
            unsafe.high[0],
            unsafe.high[0],
            unsafe.low[0],
            unsafe.low[0],
        ]
        y = [
            unsafe.low[1],
            unsafe.low[1],
            unsafe.high[1],
            unsafe.high[1],
            unsafe.low[1],
        ]
        ax.plot(x, y, color="red", alpha=0.5, zorder=7)
    for target_space in env.target_spaces:
        x = [
            target_space.low[0],
            target_space.high[0],
            target_space.high[0],
            target_space.low[0],
            target_space.low[0],
        ]
        y = [
            target_space.low[1],
            target_space.low[1],
            target_space.high[1],
            target_space.high[1],
            target_space.low[1],
        ]
        ax.plot(x, y, color="green", alpha=0.5, zorder=7)

    ax.set_xlim([env.observation_space.low[0], env.observation_space.high[0]])
    ax.set_ylim([env.observation_space.low[1], env.observation_space.high[1]])
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
