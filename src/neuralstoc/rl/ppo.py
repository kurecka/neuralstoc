import os
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


import jax
import numpy as np
import tensorflow as tf

import jax.numpy as jnp

from neuralstoc.utils import (
    pretty_time,
    jax_save,
    plot_policy,
    jv_contains,
)
from neuralstoc.rsm.lipschitz import lipschitz_l1_jax, lipschitz_linf_jax

SHUFFLE_BUFFER_SIZE = 4096 * 100


class ExperienceBuffer:
    def __init__(self):
        self.gamma = 0.99
        self.ep_states = []
        self.ep_action_sampled = []
        self.ep_action_log_prob = []
        self.ep_rewards = []
        self.ep_dones = []
        self.action_sampled = []
        self.action_log_prob = []
        self.states = []
        self.returns = []
        self.episode_total_returns = []
        self.advantage = None
        self._ds_seed = 0
        self.r_mean = None
        self.r_std = None

    def get_ds_seed(self):
        self._ds_seed += 1
        return self._ds_seed

    def close_episodes(self):
        batch_size = self.ep_states[0].shape[0]

        returns = []
        discounted_sum = jnp.zeros(batch_size)
        episode_return = jnp.zeros(batch_size)
        for i in range(len(self.ep_rewards) - 1, -1, -1):
            r = self.ep_rewards[i]
            done = self.ep_dones[i].astype(jnp.float32)
            r = (1.0 - done) * r
            discounted_sum = r + self.gamma * discounted_sum
            episode_return += r
            returns.append(discounted_sum)
        returns.reverse()

        for i in range(len(self.ep_states)):
            active = jnp.logical_not(self.ep_dones[i])
            states = self.ep_states[i][active]
            action_sampled = self.ep_action_sampled[i][active]
            action_log_prob = self.ep_action_log_prob[i][active]
            ep_returns = returns[i][active]

            self.states.append(states)
            self.action_sampled.append(action_sampled)
            self.action_log_prob.append(action_log_prob)
            self.returns.append(ep_returns)
        self.episode_total_returns.append(episode_return)

    def normalize_rewards(self):
        returns = jnp.concatenate(self.returns).flatten()
        mean, std = jnp.mean(returns), jnp.std(returns)
        if self.r_mean is None:
            self.r_mean = mean
            self.r_std = std
        else:
            alpha = 0.9
            self.r_mean = alpha * self.r_mean + (1.0 - alpha) * mean
            self.r_std = alpha * self.r_std + (1.0 - alpha) * std
        self.r_std = jnp.maximum(self.r_std, 1e-6)
        self.returns = [(r - self.r_mean) / self.r_std for r in self.returns]

    @property
    def total_returns(self):
        return np.stack(self.episode_total_returns).flatten()

    def append(self, state, sampled_action, log_prob, reward, done):
        self.ep_states.append(state)
        self.ep_action_sampled.append(sampled_action)
        self.ep_action_log_prob.append(log_prob)
        self.ep_rewards.append(reward)
        self.ep_dones.append(done)

    def clear(self):
        self.ep_states = []
        self.ep_action_sampled = []
        self.ep_action_log_prob = []
        self.ep_rewards = []
        self.ep_dones = []
        self.action_sampled = []
        self.states = []
        self.returns = []
        self.action_log_prob = []
        self.episode_total_returns = []

    def get_states(self):
        return np.array(jnp.concatenate(self.states, 0))

    def set_advantage(self, values, normalize_a=True):
        returns = np.array(jnp.concatenate(self.returns))
        advantage = returns - np.array(values)
        if normalize_a:
            mean, std = np.mean(advantage), np.std(advantage)
            advantage = (advantage - mean) / (std + 1e-6)
        self.advantage = advantage

    def get_value_train_iter(self, batch_size=1024):
        states = np.array(jnp.concatenate(self.states, axis=0))
        returns = np.array(jnp.concatenate(self.returns).reshape((-1, 1)))
        train_ds = tf.data.Dataset.from_tensor_slices((states, returns))
        train_ds = (
            train_ds.shuffle(SHUFFLE_BUFFER_SIZE, seed=self.get_ds_seed())
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return train_ds

    def get_policy_train_iter(self, batch_size=1024):
        states = np.array(jnp.concatenate(self.states, axis=0))
        action_sampled = np.array(jnp.concatenate(self.action_sampled, 0))
        action_log_prob = np.array(jnp.concatenate(self.action_log_prob, 0).flatten())
        train_ds = tf.data.Dataset.from_tensor_slices(
            (states, action_sampled, action_log_prob, self.advantage)
        )
        train_ds = (
            train_ds.shuffle(SHUFFLE_BUFFER_SIZE, seed=self.get_ds_seed())
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return train_ds


@jax.jit
def train_step_value(value_s, states, returns):
    def value_loss_fn(v_params):
        value = value_s.apply_fn(v_params, states)
        loss = jnp.mean(jnp.square(value - returns))
        return loss, value

    v_grad_fn = jax.value_and_grad(value_loss_fn, has_aux=True)
    (v_loss, values), grads = v_grad_fn(value_s.params)
    value_s = value_s.apply_gradients(grads=grads)
    metrics = {"v_loss": v_loss}
    return value_s, metrics


def gauss_log_prob(mean, std, x):
    return -1 / (2 * jnp.square(std)) * jnp.sum(jnp.square(mean - x), axis=-1)


@jax.jit
def train_step_policy(
    policy_s,
    states,
    sampled_actions,
    action_logprobs,
    advantage,
    action_std,
    max_lip,
    lip_l1,
    lip_linf,
):
    clip_ratio = 0.1

    def policy_loss_fn(p_params):
        mean = policy_s.apply_fn(p_params, states)
        log_prob = gauss_log_prob(mean, action_std, sampled_actions)
        ratio = jnp.exp(log_prob - action_logprobs)
        surr1 = ratio * advantage
        surr2 = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
        loss = -jnp.mean(jnp.minimum(surr1, surr2))
        lip_loss_l1 = jnp.maximum(lipschitz_l1_jax(p_params), max_lip)
        lip_loss_linf = jnp.maximum(lipschitz_linf_jax(p_params), max_lip)
        loss += lip_l1 * lip_loss_l1
        loss += lip_linf * lip_loss_linf

        return loss

    p_grad_fn = jax.value_and_grad(policy_loss_fn)
    p_loss, grads = p_grad_fn(policy_s.params)
    policy_s = policy_s.apply_gradients(grads=grads)
    metrics = {"p_loss": p_loss}

    return policy_s, metrics


class vPPO:
    """
    Vectorized Proximal Policy Optimization (PPO) implementation for NeuralStoc.
    
    This class implements the PPO algorithm to pre-train neural controllers before
    they are refined with certificates in the learner-verifier loop. It uses
    vectorized operations to efficiently train on multiple trajectories in parallel.
    
    Attributes:
        p_state: The policy network state (contains parameters and optimizer state)
        c_state: The critic network state (contains parameters and optimizer state)
        env: The stochastic environment for training
        max_lip: Maximum Lipschitz regularization factor
        norm: Norm for Lipschitz regularization ('l1' or 'linf')
        normalize_r: Whether to normalize rewards
        normalize_a: Whether to normalize advantages
    """
    
    def __init__(self, p_state, c_state, env, max_lip, norm, normalize_r, normalize_a):
        self.env = env
        self.norm = norm
        self.normalize_a = normalize_a
        self.normalize_r = normalize_r
        self.norm_fn = lipschitz_l1_jax if norm == "l1" else lipschitz_linf_jax

        self.action_dim = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space.shape[0]
        self.p_state = p_state
        self.c_state = c_state
        self.max_lip = jnp.float32(max_lip)

        self.gamma = 0.99
        self.lip_factor = 0
        self.action_std = 0.2
        self.i = 0
        self.rng = jax.random.PRNGKey(1)
        self.buffer = ExperienceBuffer()

        self._best_r = None
        self._best_pc_states = None

        self._num_traj = 0
        self._num_end_in_target = 0

    def clear_history(self):
        self.buffer.clear()

    def get_rng_keys(self, n):
        self.rng, rng = jax.random.split(self.rng)
        if n > 1:
            rng = jax.random.split(rng, n)
        return rng

    def sample_rollouts(self, batch_size=1024):
        rng = self.get_rng_keys(batch_size)
        state, obs = self.env.v_reset(rng)
        done = jnp.zeros(batch_size, dtype=jnp.bool_)
        while not jnp.any(done):
            action_mean = self.p_state.apply_fn(self.p_state.params, obs)
            rng = self.get_rng_keys(1)
            action = (
                jax.random.normal(rng, shape=(batch_size, self.action_dim))
                * self.action_std
                + action_mean
            )
            log_prob = gauss_log_prob(action_mean, self.action_std, action)
            rng = self.get_rng_keys(batch_size)
            state, new_obs, reward, new_done = self.env.v_step(state, action, rng)
            self.buffer.append(obs, action, log_prob, reward, done)
            obs = new_obs
            done = new_done

        contains = None
        for target_space in self.env.target_spaces:
            c = jv_contains(target_space, obs)
            if contains is not None:
                contains = jnp.logical_or(contains, c)
            else:
                contains = c
        in_target = jnp.sum(contains.astype(jnp.int64))
        self._num_traj += contains.shape[0]
        self._num_end_in_target += in_target
        self.buffer.close_episodes()

    def run(
        self, num_iters, std_start, std_end, lip_start, lip_end, save_every, verbose
    ):
        """
        Run the PPO training loop for a specified number of iterations.
        
        Args:
            num_iters: Number of training iterations to perform
            std_start: Initial standard deviation for exploration
            std_end: Final standard deviation for exploration
            lip_start: Initial Lipschitz regularization factor
            lip_end: Final Lipschitz regularization factor
            save_every: Interval for saving checkpoints (None = no saving)
            verbose: Whether to print training progress
            
        Returns:
            The trained policy network state
        """
        stds = jnp.linspace(std_start, std_end, num_iters)
        lip = jnp.linspace(lip_start, lip_end, num_iters)
        for i in range(num_iters):
            self.action_std = stds[i]
            self.lip_factor = lip[i]
            self._num_traj, self._num_end_in_target = 0, 0
            rs = self.run_iter(verbose)
            print(
                f"Trajectories ending in target: {self._num_end_in_target}/{self._num_traj} ({100*self._num_end_in_target/self._num_traj:0.2f}%)"
            )
            if save_every is not None and i % save_every == 0:
                filename = f"checkpoints/{self.env.name}_{i:d}_ppo.jax"
                self.save(filename)
                print(f"SAVED at {filename}")

            if verbose and i % 10 == 0:
                os.makedirs("plots_ppo", exist_ok=True)
                filename = f"plots_ppo/{self.env.name}_{i:03d}.png"
                plot_policy(
                    self.env,
                    self.p_state,
                    filename,
                )
        if verbose:
            os.makedirs("plots_ppo", exist_ok=True)
            filename = f"plots_ppo/{self.env.name}_{num_iters:03d}.png"
            plot_policy(
                self.env,
                self.p_state,
                filename,
            )

    def save(self, filename):
        jax_save(
            {"policy": self.p_state, "value": self.c_state},
            filename,
        )

    def run_iter(self, verbose):
        start_time = time.time()
        for i in range(3):
            self.sample_rollouts()
        rs = self.buffer.total_returns
        p_lip = self.norm_fn(self.p_state.params)
        if verbose:
            print(
                f"Iter {self.i} R={np.mean(rs):0.2f} [{np.min(rs):0.2f}, {np.max(rs):0.2f}] with lip = {p_lip:0.3f} (rollouts took {pretty_time(time.time()-start_time)})",
                flush=True,
            )
        if self.normalize_r:
            self.buffer.normalize_rewards()
        self.train_epoch(verbose)
        self.clear_history()
        self.i += 1
        return rs

    def get_mean_return(self):
        rs = [self.sample_rollout() for i in range(10)]
        return np.mean(rs)

    def train_epoch(self, verbose):
        states = self.buffer.get_states()
        values = self.c_state.apply_fn(self.c_state.params, states).flatten()

        self.buffer.set_advantage(values, normalize_a=self.normalize_a)

        policy_ds = self.buffer.get_policy_train_iter()
        policy_epochs = 30 if self.i == 0 else 10
        lip_l1 = jnp.float32(self.lip_factor if self.norm == "l1" else 0)
        lip_linf = jnp.float32(self.lip_factor if self.norm == "linf" else 0)

        for p_epoch in range(policy_epochs):
            for batch in policy_ds.as_numpy_iterator():
                action_std = jnp.float32(self.action_std)
                self.p_state, metrics = train_step_policy(
                    self.p_state,
                    batch[0],
                    batch[1],
                    batch[2],
                    batch[3],
                    action_std,
                    self.max_lip,
                    lip_l1,
                    lip_linf,
                )
        value_ds = self.buffer.get_value_train_iter()
        value_epochs = 10 if self.i == 0 else 5
        for v_epoch in range(value_epochs):
            for batch in value_ds.as_numpy_iterator():
                self.c_state, metrics = train_step_value(
                    self.c_state, batch[0], batch[1]
                )