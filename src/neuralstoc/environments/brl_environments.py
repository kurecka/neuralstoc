
from gym import spaces
import numpy as np
from scipy.stats import triang
import jax.numpy as jnp
from functools import partial
import jax

from neuralstoc.utils import triangular, make_unsafe_spaces
from brax.envs.base import Env, State



def angle_normalize(x):
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi



class bLDSEnv(Env):
    name = f"blds"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.has_render = False
        self.episode_length = 200

        safe = np.array([0.2, 0.2], np.float32)
        self.target_spaces = [spaces.Box(low=-safe, high=safe, dtype=np.float32)]
        self.init_spaces = [
            spaces.Box(
                low=np.array([-0.25, -0.1]),
                high=np.array([-0.2, 0.1]),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([0.2, -0.1]),
                high=np.array([0.25, 0.1]),
                dtype=np.float32,
            ),
        ]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-0.7 * np.ones(2, dtype=np.float32),
            high=0.7 * np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )
        self.noise = np.array([0.002, 0.001])
        self.unsafe_spaces = [
            spaces.Box(
                low=self.observation_space.low,
                high=np.array([self.observation_space.low[0] + 0.1, -0.4]),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([self.observation_space.high[0] - 0.1, 0.4]),
                high=self.observation_space.high,
                dtype=np.float32,
            ),
        ]

        self._jax_rng = jax.random.PRNGKey(777)
        self.v_next = jax.vmap(self.next, in_axes=(0, 0), out_axes=0)
        self.v_step = jax.jit(jax.vmap(self.step))
        self.v_reset = jax.jit(jax.vmap(self.reset))

    @property
    def noise_bounds(self):
        return -self.noise, self.noise

    @property
    def observation_size(self):
        return self.observation_space.shape[0]

    @property
    def action_size(self):
        return self.action_space.shape[0]

    @property
    def backend(self):
        return None

    @partial(jax.jit, static_argnums=(0,))
    def next(self, state, action):
        action = jnp.clip(action, -1, 1)

        tau = 0.98
        new_y = state[1] * tau + action[0] * 0.1
        new_x = state[0] * 1.0 + new_y * 0.02
        new_y = jnp.clip(
            new_y, self.observation_space.low[1], self.observation_space.high[1]
        )
        new_x = jnp.clip(
            new_x, self.observation_space.low[0], self.observation_space.high[0]
        )
        return jnp.array([new_x, new_y])

    def add_noise(self, state):
        self._jax_rng, rng = jax.random.split(self._jax_rng, 2)
        noise = triangular(rng, (self.observation_space.shape[0],))
        noise = noise * self.noise
        return state + noise

    @partial(jax.jit, static_argnums=(0,))
    def step(self, _state, action) -> State:  # pytype: disable=signature-mismatch

        state = _state.pipeline_state
        step = state[0]
        state = state[1:3]
        next_state = self.next(state, action)

        rng0, rng1 = jax.random.split(_state.info['rng'], 2)
        _state.info['rng'] = rng1

        noise = triangular(rng0, (self.observation_space.shape[0],))
        noise = noise * self.noise
        next_state = next_state + noise
        next_state = jnp.clip(
            next_state, self.observation_space.low, self.observation_space.high
        )
        reward = jnp.zeros((), dtype=jnp.float32)
        for unsafe in self.unsafe_spaces:
            contain = jnp.all(
                jnp.logical_and(next_state >= unsafe.low, next_state <= unsafe.high)
            )
            reward += -jnp.float32(contain)
        for target in self.target_spaces:
            contain = jnp.all(
                jnp.logical_and(next_state >= target.low, next_state <= target.high)
            )
            reward += jnp.float32(contain)

        reward -= 2 * jnp.mean(jnp.abs(next_state / self.observation_space.high))

        next_packed = jnp.array([step + 1, next_state[0], next_state[1]])
        done = jnp.array(step >= self.episode_length, dtype=jnp.float32)
        truncation = jnp.where(
            done,
            jnp.ones_like(done, dtype=jnp.float32),
            _state.info['truncation']
        )
        info = {
            'rng': _state.info['rng'],
            'truncation': truncation,
            'steps': step + 1,
            'first_pipeline_state': _state.info['first_pipeline_state'],
            'first_obs': _state.info['first_obs'],
        }
        _state = _state.replace(
            pipeline_state=next_packed,
            obs=next_state,
            reward=reward,
            done=done,
            info=info
        )
        return _state

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng) -> State:
        rng0, rng1 = jax.random.split(rng, 2)
        obs = jax.random.uniform(
            rng0,
            shape=(self.observation_space.shape[0],),
            minval=self.observation_space.low,
            maxval=self.observation_space.high,
        )
        state = jnp.array([0, obs[0], obs[1]])
        metrics = {}

        info = {
            'rng': rng1,
            'truncation': jnp.zeros((), dtype=jnp.float32),
            'steps': jnp.zeros((), dtype=jnp.int32),
            'first_pipeline_state': state,
            'first_obs': obs,
        }
        return State(
            pipeline_state=state,
            obs=obs,
            reward=jnp.zeros((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.float32),
            metrics=metrics,
            info=info
        )

    @property
    def lipschitz_constant(self):
        A = np.max(np.sum(np.array([[1, 0.02 * 0.98, 0.02 * 0.1], [0, 0.98, 0.1]]), axis=0))
        return A

    @property
    def lipschitz_constant_linf(self):
        A = np.max(np.sum(np.array([[1, 0.02 * 0.98, 0.02 * 0.1], [0, 0.98, 0.1]]), axis=1))
        return A

    @property
    def delta(self):
        return 0.1 + self.noise[0]

    def integrate_noise(self, a: list, b: list):
        dims = 2
        pmass = np.ones(a[0].shape[0])
        for i in range(dims):
            loc = self.noise_bounds[0][i]
            scale = self.noise_bounds[1][i] - self.noise_bounds[0][i]
            marginal_pmass = triang.cdf(b[i], c=0.5, loc=loc, scale=scale) - triang.cdf(
                a[i], c=0.5, loc=loc, scale=scale
            )
            pmass *= marginal_pmass
        return pmass


class bInvertedPendulum(Env):
    name = "bpend"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.episode_length = 200

        init = np.array([0.7, 0.7], np.float32)
        self.init_spaces = [spaces.Box(low=-init, high=init, dtype=np.float32)]
        init = np.array([0.75, 0.75], np.float32)
        self.init_spaces_train = [spaces.Box(low=-init, high=init, dtype=np.float32)]

        high = np.array([3, 3], dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.noise = np.array([0.005, 0.002])

        safe = np.array([0.2, 0.2], np.float32)
        self.target_spaces = [spaces.Box(low=-safe, high=safe, dtype=np.float32)]
        safe = np.array([0.1, 0.1], np.float32)
        self.target_space_train = spaces.Box(low=-safe, high=safe, dtype=np.float32)

        observation_space = np.array([3, 3], np.float32)
        self.observation_space = spaces.Box(
            low=-observation_space, high=observation_space, dtype=np.float32
        )

        self.unsafe_spaces = [
            spaces.Box(
                low=self.observation_space.low,
                high=np.array([self.observation_space.low[0] + 0.1, 0]),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([self.observation_space.high[0] - 0.1, 0]),
                high=self.observation_space.high,
                dtype=np.float32,
            ),
        ]

        self._jax_rng = jax.random.PRNGKey(777)
        self.v_next = jax.vmap(self.next, in_axes=(0, 0), out_axes=0)
        self.v_step = jax.jit(jax.vmap(self.step))
        self.v_reset = jax.jit(jax.vmap(self.reset))

    @partial(jax.jit, static_argnums=(0,))
    def next(self, state, action):
        th, thdot = state  # th := theta
        max_speed = 5
        dt = 0.05
        g = 10
        m = 0.15
        l = 0.5
        b = 0.1

        u = 2 * jnp.clip(action, -1, 1)[0]
        newthdot = (1 - b) * thdot + (
                -3 * g * 0.5 / (2 * l) * jnp.sin(th + jnp.pi) + 3.0 / (m * l ** 2) * u
        ) * dt
        newthdot = jnp.clip(newthdot, -max_speed, max_speed)
        newth = th + newthdot * dt

        newth = jnp.clip(
            newth, self.observation_space.low[0], self.observation_space.high[0]
        )
        newthdot = jnp.clip(
            newthdot, self.observation_space.low[1], self.observation_space.high[1]
        )
        return jnp.array([newth, newthdot])

    def add_noise(self, state):
        self._jax_rng, rng = jax.random.split(self._jax_rng, 2)
        noise = triangular(rng, (self.observation_space.shape[0],))
        noise = noise * self.noise
        return state + noise


    @partial(jax.jit, static_argnums=(0,))
    def step(self, _state, action) -> State:  # pytype: disable=signature-mismatch

        state = _state.pipeline_state
        step = state[0]
        state = state[1:3]
        next_state = self.next(state, action)

        rng0, rng1 = jax.random.split(_state.info['rng'], 2)
        _state.info['rng'] = rng1

        noise = triangular(rng0, (self.observation_space.shape[0],))
        noise = noise * self.noise
        next_state = next_state + noise
        next_state = jnp.clip(
            next_state, self.observation_space.low, self.observation_space.high
        )
        reward = jnp.zeros((), dtype=jnp.float32)
        for unsafe in self.unsafe_spaces:
            contain = jnp.all(
                jnp.logical_and(next_state >= unsafe.low, next_state <= unsafe.high)
            )
            reward += -jnp.float32(contain)
        contain = jnp.all(
            jnp.logical_and(
                next_state >= self.target_space_train.low,
                next_state <= self.target_space_train.high,
            )
        )
        reward += jnp.float32(contain)
        th, thdot = next_state
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2
        reward -= costs

        next_packed = jnp.array([step + 1, next_state[0], next_state[1]])
        done = jnp.array(step >= self.episode_length, dtype=jnp.float32)
        truncation = jnp.where(
            done,
            jnp.ones_like(done, dtype=jnp.float32),
            _state.info['truncation']
        )
        info = {
            'rng': _state.info['rng'],
            'truncation': truncation,
            'steps': step + 1,
            'first_pipeline_state': _state.info['first_pipeline_state'],
            'first_obs': _state.info['first_obs'],
        }
        _state = _state.replace(
            pipeline_state=next_packed,
            obs=next_state,
            reward=reward,
            done=done,
            info=info
        )
        return _state

    @property
    def observation_size(self):
        return self.observation_space.shape[0]

    @property
    def action_size(self):
        return self.action_space.shape[0]

    @property
    def noise_bounds(self):
        return -self.noise, self.noise

    @property
    def lipschitz_constant(self):
        return 1.78

    @property
    def lipschitz_constant_linf(self):
        return 1.78

    @property
    def backend(self):
        return None

    @property
    def delta(self):
        return 0.1 + self.noise[0]

    def integrate_noise(self, a: list, b: list):
        dims = 2
        pmass = np.ones(a[0].shape[0])
        for i in range(dims):
            loc = self.noise_bounds[0][i]
            scale = self.noise_bounds[1][i] - self.noise_bounds[0][i]
            marginal_pmass = triang.cdf(b[i], c=0.5, loc=loc, scale=scale) - triang.cdf(
                a[i], c=0.5, loc=loc, scale=scale
            )
            pmass *= marginal_pmass
        return pmass

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng) -> State:
        rng0, rng1 = jax.random.split(rng, 2)

        obs = jax.random.uniform(
            rng0,
            shape=(self.observation_space.shape[0],),
            minval=self.observation_space.low,
            maxval=self.observation_space.high,
        )
        state = jnp.array([0, obs[0], obs[1]])
        metrics = {}

        info = {
            'rng': rng1,
            'truncation': jnp.zeros((), dtype=jnp.float32),
            'steps': jnp.zeros((), dtype=jnp.int32),
            'first_pipeline_state': state,
            'first_obs': obs,
        }
        return State(
            pipeline_state=state,
            obs=obs,
            reward=jnp.zeros((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.float32),
            metrics=metrics,
            info=info
        )


class bCollisionAvoidanceEnv(Env):
    name = "bcavoid"
    is_paralyzed = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steps = None
        self.state = None
        self.has_render = False
        self.episode_length = 200

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )
        self.noise = np.array([0.05, 0.05])  # was 0.02 before
        safe = np.array([0.2, 0.2], np.float32)  # was 0.1 before
        self.target_spaces = [spaces.Box(low=-safe, high=safe, dtype=np.float32)]

        self.init_spaces_train = make_unsafe_spaces(
            self.observation_space, np.array([0.9, 0.9], np.float32)
        )
        self.init_spaces = [
            spaces.Box(
                low=np.array([-1, -0.6]),
                high=np.array([-0.9, 0.6]),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([0.9, -0.6]),
                high=np.array([1.0, 0.6]),
                dtype=np.float32,
            ),
        ]

        self.unsafe_spaces = []
        self.unsafe_spaces.append(
            spaces.Box(
                low=np.array([-0.3, 0.7]), high=np.array([0.3, 1.0]), dtype=np.float32
            )
        )
        self.unsafe_spaces.append(
            spaces.Box(
                low=np.array([-0.3, -1.0]), high=np.array([0.3, -0.7]), dtype=np.float32
            )
        )
        # self.noise = np.array([0.001, 0.001])
        self._jax_rng = jax.random.PRNGKey(777)
        self.v_next = jax.vmap(self.next, in_axes=(0, 0), out_axes=0)
        self.v_step = jax.jit(jax.vmap(self.step))
        self.v_reset = jax.jit(jax.vmap(self.reset))

    @property
    def noise_bounds(self):
        return -self.noise, self.noise

    @partial(jax.jit, static_argnums=(0,))
    def next(self, state, action):
        action = jnp.clip(action, -1, 1)

        obstacle1 = jnp.array((0, 1))
        force1 = jnp.array((0, 1))
        dist1 = jnp.linalg.norm(obstacle1 - state)
        dist1 = jnp.clip(dist1 / 0.3, 0, 1)
        action = action * dist1 + (1 - dist1) * force1

        obstacle2 = jnp.array((0, -1))
        force2 = jnp.array((0, -1))
        dist2 = jnp.linalg.norm(obstacle2 - state)
        dist2 = jnp.clip(dist2 / 0.3, 0, 1)
        action = action * dist2 + (1 - dist2) * force2

        state = state + action * 0.2
        state = jnp.clip(state, self.observation_space.low, self.observation_space.high)

        return state

    def add_noise(self, state):
        self._jax_rng, rng = jax.random.split(self._jax_rng, 2)
        noise = triangular(rng, (self.observation_space.shape[0],))
        noise = noise * self.noise
        return state + noise

    @partial(jax.jit, static_argnums=(0,))
    def step(self, _state, action) -> State:  # pytype: disable=signature-mismatch

        state = _state.pipeline_state
        step = state[0]
        state = state[1:3]
        next_state = self.next(state, action)

        rng0, rng1 = jax.random.split(_state.info['rng'], 2)
        _state.info['rng'] = rng1

        noise = triangular(rng0, (self.observation_space.shape[0],))
        noise = noise * self.noise
        next_state = next_state + noise
        next_state = jnp.clip(
            next_state, self.observation_space.low, self.observation_space.high
        )
        reward = jnp.zeros((), dtype=jnp.float32)
        for unsafe in self.unsafe_spaces:
            contain = jnp.all(
                # jnp.logical_and(state >= unsafe.low, state <= unsafe.high)
                jnp.logical_and(next_state >= unsafe.low, next_state <= unsafe.high)
            )
            reward += -jnp.float32(contain)
        for target in self.target_spaces:
            contain = jnp.all(
                # jnp.logical_and(state >= target.low, state <= target.high)
                jnp.logical_and(next_state >= target.low, next_state <= target.high)
            )
            reward += jnp.float32(contain)

        reward -= 2 * jnp.mean(jnp.abs(next_state / self.observation_space.high))

        next_packed = jnp.array([step + 1, next_state[0], next_state[1]])
        done = jnp.array(step >= self.episode_length, dtype=jnp.float32)
        truncation = jnp.where(
            done,
            jnp.ones_like(done, dtype=jnp.float32),
            _state.info['truncation']
        )
        info = {
            'rng': _state.info['rng'],
            'truncation': truncation,
            'steps': step + 1,
            'first_pipeline_state': _state.info['first_pipeline_state'],
            'first_obs': _state.info['first_obs'],
        }
        _state = _state.replace(
            pipeline_state=next_packed,
            obs=next_state,
            reward=reward,
            done=done,
            info=info
        )
        return _state

    @property
    def observation_size(self):
        return self.observation_space.shape[0]

    @property
    def action_size(self):
        return self.action_space.shape[0]

    @property
    def lipschitz_constant(self):
        return 1.2

    @property
    def backend(self):
        return None

    def integrate_noise(self, a: list, b: list):
        dims = 2
        pmass = np.ones(a[0].shape[0])
        for i in range(dims):
            loc = self.noise_bounds[0][i]
            scale = self.noise_bounds[1][i] - self.noise_bounds[0][i]
            marginal_pmass = triang.cdf(b[i], c=0.5, loc=loc, scale=scale) - triang.cdf(
                a[i], c=0.5, loc=loc, scale=scale
            )
            pmass *= marginal_pmass
        return pmass

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng) -> State:
        rng0, rng1 = jax.random.split(rng, 2)
        obs = jax.random.uniform(
            rng0,
            shape=(self.observation_space.shape[0],),
            minval=self.observation_space.low,
            maxval=self.observation_space.high,
        )
        state = jnp.array([0, obs[0], obs[1]])
        metrics = {}

        info = {
            'rng': rng1,
            'truncation': jnp.zeros((), dtype=jnp.float32),
            'steps': jnp.zeros((), dtype=jnp.int32),
            'first_pipeline_state': state,
            'first_obs': obs,
        }
        return State(
            pipeline_state=state,
            obs=obs,
            reward=jnp.zeros((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.float32),
            metrics=metrics,
            info=info
        )


def create_2_link_mat(state, action):
    l = 0.1
    m = 0.05
    g = 9.81
    delta = 0.01
    I = 0.1
    U = 2.0
    b = 0.3
    phi, phi_dot = jnp.split(state, 2)
    a00 = I + m * jnp.square(l) + jnp.square(l) * m
    a11 = I + m * jnp.square(l)
    a01 = m * l * l
    a10 = a01
    M = jnp.array(
        [[a00, a01 * jnp.cos(phi[0] - phi[1])], [a10 * jnp.cos(phi[1] - phi[0]), a11]]
    )
    C = jnp.array(
        [
            [0, -a01 * phi_dot[1] * jnp.sin(phi[1] - phi[0])],
            [-a10 * phi_dot[0] * jnp.sin(phi[0] - phi[1]), 0],
        ]
    )
    b0 = (m * l + l * m) * g
    b1 = (m * l) * g
    tau = jnp.array([-b0 * jnp.sin(phi[0]), -b1 * jnp.sin(phi[1])])

    M_inv = jnp.linalg.inv(M)
    phi_dot_new = (1 - b) * phi_dot + delta * (
        jnp.dot(M_inv, jnp.clip(jnp.dot(-C, phi) - tau, -1.2, 1.2) + U * action)
    )
    phi_new = phi + delta * phi_dot_new
    return jnp.concatenate([phi_new, phi_dot_new])


class bHumanoidBalance2(Env):

    name = "bhuman2"

    def __init__(self):
        self.n = 2
        self._fig_id = 0
        self.episode_length = 200

        high = np.array([0.4, 0.4, 0.35, 0.35], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        init = np.array([0.2, 0.2, 0.2, 0.2], np.float32)
        self.init_spaces = [spaces.Box(low=-init, high=init, dtype=np.float32)]

        self.unsafe_spaces = make_unsafe_spaces(
            self.observation_space, np.array([0.35, 0.35, 0.3, 0.3], np.float32)
        )

        self.noise = np.array([0.001, 0.001, 0.0, 0.0])

        safe = np.array([0.1, 0.1, 0.1, 0.1], np.float32)
        self.target_spaces = [spaces.Box(low=-safe, high=safe, dtype=np.float32)]

        self._jax_rng = jax.random.PRNGKey(777)
        self.v_next = jax.vmap(self.next, in_axes=(0, 0), out_axes=0)
        self.v_step = jax.jit(jax.vmap(self.step))
        self.v_reset = jax.jit(jax.vmap(self.reset))

    @partial(jax.jit, static_argnums=(0,))
    def next(self, state, action):
        action = jnp.clip(action, -1, 1)
        new_state = create_2_link_mat(state, action)
        new_state = jnp.clip(
            new_state, self.observation_space.low, self.observation_space.high
        )
        return new_state

    def add_noise(self, state):
        self._jax_rng, rng = jax.random.split(self._jax_rng, 2)
        noise = triangular(rng, (self.observation_space.shape[0],))
        noise = noise * self.noise
        return state + noise

    @partial(jax.jit, static_argnums=(0,))
    def step(self, _state, action) -> State:  # pytype: disable=signature-mismatch
        state = _state.pipeline_state
        step = state[0]
        state = state[1:5]
        next_state = self.next(state, action)

        rng0, rng1 = jax.random.split(_state.info['rng'], 2)
        _state.info['rng'] = rng1

        noise = triangular(rng0, (self.observation_space.shape[0],))
        noise = noise * self.noise
        next_state = next_state + noise
        next_state = jnp.clip(
            next_state, self.observation_space.low, self.observation_space.high
        )

        reward = jnp.zeros((), dtype=jnp.float32)
        for unsafe in self.unsafe_spaces:
            contain = jnp.all(
                jnp.logical_and(state >= unsafe.low, state <= unsafe.high)
            )
            reward += -10 * jnp.float32(contain)
        for target in self.target_spaces:
            contain = jnp.all(
                jnp.logical_and(state >= target.low, state <= target.high)
            )
            center = 0.5 * (target.low + target.high)
            dist = jnp.sum(jnp.abs(center - next_state))
            dist = jnp.clip(dist, 0, 2)
            reward += 2 * (2.0 - dist)
            reward += jnp.float32(contain)

        next_step = jnp.minimum(step + 1, self.episode_length)

        next_packed = jnp.array([next_step, next_state[0], next_state[1], next_state[2], next_state[3]])
        done = jnp.array(step >= self.episode_length, dtype=jnp.float32)
        truncation = jnp.where(
            done,
            jnp.ones_like(done, dtype=jnp.float32),
            _state.info['truncation']
        )
        info = {
            'rng': _state.info['rng'],
            'truncation': truncation,
            'steps': step + 1,
            'first_pipeline_state': _state.info['first_pipeline_state'],
            'first_obs': _state.info['first_obs'],
        }
        _state = _state.replace(
            pipeline_state=next_packed,
            obs=next_state,
            reward=reward,
            done=done,
            info=info
        )
        return _state

    @property
    def noise_bounds(self):
        return -self.noise[0:2], self.noise[0:2]

    @property
    def observation_size(self):
        return self.observation_space.shape[0]

    @property
    def action_size(self):
        return self.action_space.shape[0]

    @property
    def backend(self):
        return None

    @property
    def lipschitz_constant(self):
        return 1.06

    @property
    def lipschitz_constant_linf(self):
        return 1.06

    @property
    def delta(self):
        return 0.1 + self.noise[0]

    def integrate_noise(self, a: list, b: list):
        dims = 2
        pmass = np.ones(a[0].shape[0])
        for i in range(dims):
            loc = self.noise_bounds[0][i]
            scale = self.noise_bounds[1][i] - self.noise_bounds[0][i]
            marginal_pmass = triang.cdf(b[i], c=0.5, loc=loc, scale=scale) - triang.cdf(
                a[i], c=0.5, loc=loc, scale=scale
            )
            pmass *= marginal_pmass
        return pmass

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng) -> State:
        rng0, rng1 = jax.random.split(rng, 2)
        obs = jax.random.uniform(
            rng0,
            shape=(self.observation_space.shape[0],),
            minval=self.observation_space.low,
            maxval=self.observation_space.high,
        )
        state = jnp.array([0, obs[0], obs[1], obs[2], obs[3]])
        metrics = {}

        info = {
            'rng': rng1,
            'truncation': jnp.zeros((), dtype=jnp.float32),
            'steps': jnp.zeros((), dtype=jnp.int32),
            'first_pipeline_state': state,
            'first_obs': obs,
        }
        return State(
            pipeline_state=state,
            obs=obs,
            reward=jnp.zeros((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.float32),
            metrics=metrics,
            info=info
        )


class bLDSS(Env):

    name = "bldss"

    def __init__(self, num_dims=2):
        assert num_dims % 2 == 0
        self.has_render = False
        self.n = 2
        self._fig_id = 0
        self.num_dims = num_dims
        self.episode_length = 200

        safe = np.array([0.2] * num_dims, np.float32)
        self.target_spaces = [spaces.Box(low=-safe, high=safe, dtype=np.float32)]

        self.init_spaces = [
            spaces.Box(
                low=np.array([-0.25] * (num_dims // 2) + [-0.1] * (num_dims // 2)),
                high=np.array([-0.2] * (num_dims // 2) + [0.1] * (num_dims // 2)),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([0.2] * (num_dims // 2) + [-0.1] * (num_dims // 2)),
                high=np.array([0.25] * (num_dims // 2) + [0.1] * (num_dims // 2)),
                dtype=np.float32,
            ),
        ]

        self.action_space = spaces.Box(low=-1, high=1, shape=(num_dims // 2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-0.7 * np.ones(num_dims, dtype=np.float32),
            high=0.7 * np.ones(num_dims, dtype=np.float32),
            dtype=np.float32,
        )
        self.noise = np.array([0.001] * (num_dims // 2) + [0.0] * (num_dims // 2))
        self.unsafe_spaces = [
            spaces.Box(
                low=self.observation_space.low,
                high=np.array([self.observation_space.low[i_] + 0.1 for i_ in range(num_dims // 2)]
                              + [-0.4] * (num_dims // 2)),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([self.observation_space.high[i_] - 0.1 for i_ in range(num_dims // 2)]
                             + [0.4] * (num_dims // 2)),
                high=self.observation_space.high,
                dtype=np.float32,
            ),
        ]

        self._jax_rng = jax.random.PRNGKey(777)
        self.v_next = jax.vmap(self.next, in_axes=(0, 0), out_axes=0)
        self.v_step = jax.jit(jax.vmap(self.step))
        self.v_reset = jax.jit(jax.vmap(self.reset))

    @partial(jax.jit, static_argnums=(0,))
    def next(self, state, action):
        action = jnp.clip(action, -1, 1)
        delta = 0.1
        x, vx = jnp.split(state, 2)

        new_vx = vx * 0.98 + action * delta
        new_x = x + new_vx * 0.02

        new_state = jnp.concatenate([new_x, new_vx])

        new_state = jnp.clip(
            new_state, self.observation_space.low, self.observation_space.high
        )
        return new_state

    def add_noise(self, state):
        self._jax_rng, rng = jax.random.split(self._jax_rng, 2)
        noise = triangular(rng, (self.observation_space.shape[0],))
        noise = noise * self.noise
        return state + noise

    @partial(jax.jit, static_argnums=(0,))
    def step(self, _state, action) -> State:
        state = _state.pipeline_state
        step = state[0]
        state = state[1:self.num_dims + 1]
        next_state = self.next(state, action)

        rng0, rng1 = jax.random.split(_state.info['rng'], 2)
        _state.info['rng'] = rng1

        noise = triangular(rng0, (self.observation_space.shape[0],))
        noise = noise * self.noise
        next_state = next_state + noise
        next_state = jnp.clip(
            next_state, self.observation_space.low, self.observation_space.high
        )

        reward = jnp.zeros((), dtype=jnp.float32)
        for unsafe in self.unsafe_spaces:
            contain = jnp.all(
                jnp.logical_and(state >= unsafe.low, state <= unsafe.high)
            )
            reward += -10 * jnp.float32(contain)
            # next_step += 200 * jnp.int32(contain)
        for target in self.target_spaces:
            contain = jnp.all(
                jnp.logical_and(state >= target.low, state <= target.high)
            )
            center = 0.5 * (target.low + target.high)
            dist = jnp.sum(jnp.abs(center - next_state))
            dist = jnp.clip(dist, 0, 2)
            reward += 2 * (2.0 - dist)
            reward += jnp.float32(contain)

        next_step = jnp.minimum(step + 1, self.episode_length)

        next_packed = jnp.array([next_step] + [next_state[i] for i in range(self.num_dims)])
        done = jnp.array(step >= self.episode_length, dtype=jnp.float32)
        truncation = jnp.where(
            done,
            jnp.ones_like(done, dtype=jnp.float32),
            _state.info['truncation']
        )
        info = {
            'rng': _state.info['rng'],
            'truncation': truncation,
            'steps': step + 1,
            'first_pipeline_state': _state.info['first_pipeline_state'],
            'first_obs': _state.info['first_obs'],
        }
        _state = _state.replace(
            pipeline_state=next_packed,
            obs=next_state,
            reward=reward,
            done=done,
            info=info
        )
        return _state

    @property
    def noise_bounds(self):
        return -self.noise[:self.num_dims // 2], self.noise[:self.num_dims // 2]

    @property
    def observation_size(self):
        return self.observation_space.shape[0]

    @property
    def action_size(self):
        return self.action_space.shape[0]

    @property
    def backend(self):
        return None

    @property
    def lipschitz_constant(self):
        A = np.max(np.sum(np.array([[1, 0.0196, 0.002]] * (self.num_dims // 2) +
                                   [[0, 0.98, 0.1]] * (self.num_dims // 2)), axis=0))
        return A

    @property
    def lipschitz_constant_linf(self):
        A = np.max(np.sum(np.array([[1, 0.0196, 0.002]] * (self.num_dims // 2) +
                                   [[0, 0.98, 0.1]] * (self.num_dims // 2)), axis=1))
        return A

    @property
    def delta(self):
        return 0.1 + self.noise[0]

    def integrate_noise(self, a: list, b: list):
        pmass = np.ones(a[0].shape[0])
        for i in range(self.num_dims // 2):
            loc = self.noise_bounds[0][i]
            scale = self.noise_bounds[1][i] - self.noise_bounds[0][i]
            marginal_pmass = triang.cdf(b[i], c=0.5, loc=loc, scale=scale) - triang.cdf(
                a[i], c=0.5, loc=loc, scale=scale
            )
            pmass *= marginal_pmass
        return pmass

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng) -> State:
        rng0, rng1 = jax.random.split(rng, 2)
        obs = jax.random.uniform(
            rng0,
            shape=(self.observation_space.shape[0],),
            minval=self.observation_space.low,
            maxval=self.observation_space.high,
        )
        state = jnp.array([0] + [obs[i] for i in range(self.num_dims)])
        metrics = {}

        info = {
            'rng': rng1,
            'truncation': jnp.zeros((), dtype=jnp.float32),
            'steps': jnp.zeros((), dtype=jnp.int32),
            'first_pipeline_state': state,
            'first_obs': obs,
        }
        return State(
            pipeline_state=state,
            obs=obs,
            reward=jnp.zeros((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.float32),
            metrics=metrics,
            info=info
        )


class bTripleIntegrator(Env):

    name = "btri"

    def __init__(self):
        self.has_render = False
        self._fig_id = 0
        self.episode_length = 200

        safe = np.array([0.2] * 3, np.float32)
        self.target_spaces = [spaces.Box(low=-safe, high=safe, dtype=np.float32)]

        self.init_spaces = [
            spaces.Box(
                low=np.array([-0.25] * 2 + [-0.1]),
                high=np.array([-0.2] * 2 + [0.1]),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([0.2] * 2 + [-0.1]),
                high=np.array([0.25] * 2 + [0.1]),
                dtype=np.float32,
            ),
        ]

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1 * np.ones(3, dtype=np.float32),
            high=1 * np.ones(3, dtype=np.float32),
            dtype=np.float32,
        )
        self.noise = np.array([0.01] * 2 + [0.005])
        self.unsafe_spaces = [
            spaces.Box(
                low=self.observation_space.low,
                high=np.array([self.observation_space.low[i_] + 0.1 for i_ in range(2)]
                              + [0]),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([self.observation_space.high[i_] - 0.1 for i_ in range(2)]
                             + [0]),
                high=self.observation_space.high,
                dtype=np.float32,
            ),
        ]

        self._jax_rng = jax.random.PRNGKey(777)
        self.v_next = jax.vmap(self.next, in_axes=(0, 0), out_axes=0)
        self.v_step = jax.jit(jax.vmap(self.step))
        self.v_reset = jax.jit(jax.vmap(self.reset))


    @partial(jax.jit, static_argnums=(0,))
    def next(self, state, action):
        action = jnp.clip(action, -1, 1)

        A = jnp.array([[1, 0.045, 0],
                        [0, 1, 0.045],
                        [0, 0, 0.9]])
        B = jnp.array([0.35, 0.45, 0.5])

        new_state = jnp.dot(A, state.reshape(-1)) + B * action

        new_state = jnp.clip(
            new_state, self.observation_space.low, self.observation_space.high
        )
        return new_state

    def add_noise(self, state):
        self._jax_rng, rng = jax.random.split(self._jax_rng, 2)
        noise = triangular(rng, (self.observation_space.shape[0],))
        noise = noise * self.noise
        return state + noise

    @partial(jax.jit, static_argnums=(0,))
    def step(self, _state, action) -> State:
        state = _state.pipeline_state
        step = state[0]
        state = state[1:4]
        next_state = self.next(state, action)

        rng0, rng1 = jax.random.split(_state.info['rng'], 2)
        _state.info['rng'] = rng1

        noise = triangular(rng0, (self.observation_space.shape[0],))
        noise = noise * self.noise
        next_state = next_state + noise
        next_state = jnp.clip(
            next_state, self.observation_space.low, self.observation_space.high
        )

        reward = jnp.zeros((), dtype=jnp.float32)
        for unsafe in self.unsafe_spaces:
            contain = jnp.all(
                jnp.logical_and(state >= unsafe.low, state <= unsafe.high)
            )
            reward += -10 * jnp.float32(contain)
        for target in self.target_spaces:
            contain = jnp.all(
                jnp.logical_and(state >= target.low, state <= target.high)
            )
            center = 0.5 * (target.low + target.high)
            dist = jnp.sum(jnp.abs(center - next_state))
            dist = jnp.clip(dist, 0, 2)
            reward += 2 * (2.0 - dist)
            reward += jnp.float32(contain)

        next_step = jnp.minimum(step + 1, self.episode_length)

        next_packed = jnp.array([next_step, next_state[0], next_state[1], next_state[2]])
        done = jnp.array(step >= self.episode_length, dtype=jnp.float32)
        truncation = jnp.where(
            done,
            jnp.ones_like(done, dtype=jnp.float32),
            _state.info['truncation']
        )
        info = {
            'rng': _state.info['rng'],
            'truncation': truncation,
            'steps': step + 1,
            'first_pipeline_state': _state.info['first_pipeline_state'],
            'first_obs': _state.info['first_obs'],
        }
        _state = _state.replace(
            pipeline_state=next_packed,
            obs=next_state,
            reward=reward,
            done=done,
            info=info
        )
        return _state

    @property
    def noise_bounds(self):
        return -self.noise, self.noise

    @property
    def observation_size(self):
        return self.observation_space.shape[0]

    @property
    def action_size(self):
        return self.action_space.shape[0]

    @property
    def backend(self):
        return None

    @property
    def lipschitz_constant(self):
        A = np.max(np.sum(np.array([[1, 0.045, 0, 0.35],
                                    [0, 1, 0.045, 0.45],
                                    [0, 0, 0.9, 0.5]]), axis=0))
        return A

    @property
    def lipschitz_constant_linf(self):
        A = np.max(np.sum(np.array([[1, 0.045, 0, 0.35],
                                    [0, 1, 0.045, 0.45],
                                    [0, 0, 0.9, 0.5]]), axis=1))
        return A

    @property
    def delta(self):
        return 0.1 + self.noise[0]

    def integrate_noise(self, a: list, b: list):
        pmass = np.ones(a[0].shape[0])
        for i in range(3):
            loc = self.noise_bounds[0][i]
            scale = self.noise_bounds[1][i] - self.noise_bounds[0][i]
            marginal_pmass = triang.cdf(b[i], c=0.5, loc=loc, scale=scale) - triang.cdf(
                a[i], c=0.5, loc=loc, scale=scale
            )
            pmass *= marginal_pmass
        return pmass

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng) -> State:
        rng0, rng1 = jax.random.split(rng, 2)
        obs = jax.random.uniform(
            rng0,
            shape=(self.observation_space.shape[0],),
            minval=self.observation_space.low,
            maxval=self.observation_space.high,
        )
        state = jnp.array([0] + [obs[i] for i in range(3)])
        metrics = {}

        info = {
            'rng': rng1,
            'truncation': jnp.zeros((), dtype=jnp.float32),
            'steps': jnp.zeros((), dtype=jnp.int32),
            'first_pipeline_state': state,
            'first_obs': obs,
        }
        return State(
            pipeline_state=state,
            obs=obs,
            reward=jnp.zeros((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.float32),
            metrics=metrics,
            info=info
        )


class b2CollisionAvoidanceEnv(Env):
    name = "b2cavoid"
    is_paralyzed = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steps = None
        self.state = None
        self.has_render = False
        self.episode_length = 200

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )
        self.noise = np.array([0.02, 0.02])
        safe = np.array([0.2, 0.2], np.float32)  # was 0.1 before
        self.target_spaces = [spaces.Box(low=-safe, high=safe, dtype=np.float32)]

        self.init_spaces_train = make_unsafe_spaces(
            self.observation_space, np.array([0.9, 0.9], np.float32)
        )
        self.init_spaces = [
            spaces.Box(
                low=np.array([-1, -0.6]),
                high=np.array([-0.9, 0.6]),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([0.9, -0.6]),
                high=np.array([1.0, 0.6]),
                dtype=np.float32,
            ),
        ]

        self.unsafe_spaces = []
        self.unsafe_spaces.append(
            spaces.Box(
                low=np.array([-0.3, 0.7]), high=np.array([0.3, 1.0]), dtype=np.float32
            )
        )
        self.unsafe_spaces.append(
            spaces.Box(
                low=np.array([-0.3, -1.0]), high=np.array([0.3, -0.7]), dtype=np.float32
            )
        )
        self._jax_rng = jax.random.PRNGKey(777)
        self.v_next = jax.vmap(self.next, in_axes=(0, 0), out_axes=0)
        self.v_step = jax.jit(jax.vmap(self.step))
        self.v_reset = jax.jit(jax.vmap(self.reset))

    @property
    def noise_bounds(self):
        return -self.noise, self.noise

    @partial(jax.jit, static_argnums=(0,))
    def next(self, state, action):
        action = jnp.clip(action, -1, 1)

        obstacle1 = jnp.array((0, 1))
        force1 = jnp.array((0, 1))
        dist1 = jnp.linalg.norm(obstacle1 - state)
        dist1 = jnp.clip(dist1 / 0.3, 0, 1)
        action = action * dist1 + (1 - dist1) * force1

        obstacle2 = jnp.array((0, -1))
        force2 = jnp.array((0, -1))
        dist2 = jnp.linalg.norm(obstacle2 - state)
        dist2 = jnp.clip(dist2 / 0.3, 0, 1)
        action = action * dist2 + (1 - dist2) * force2

        state = state + action * 0.2
        state = jnp.clip(state, self.observation_space.low, self.observation_space.high)

        return state

    def add_noise(self, state):
        self._jax_rng, rng = jax.random.split(self._jax_rng, 2)
        noise = triangular(rng, (self.observation_space.shape[0],))
        noise = noise * self.noise
        return state + noise

    @partial(jax.jit, static_argnums=(0,))
    def step(self, _state, action) -> State:  # pytype: disable=signature-mismatch

        state = _state.pipeline_state
        step = state[0]
        state = state[1:3]
        next_state = self.next(state, action)

        rng0, rng1 = jax.random.split(_state.info['rng'], 2)
        _state.info['rng'] = rng1

        noise = triangular(rng0, (self.observation_space.shape[0],))
        noise = noise * self.noise
        next_state = next_state + noise
        next_state = jnp.clip(
            next_state, self.observation_space.low, self.observation_space.high
        )
        reward = jnp.zeros((), dtype=jnp.float32)
        for unsafe in self.unsafe_spaces:
            contain = jnp.all(
                jnp.logical_and(next_state >= unsafe.low, next_state <= unsafe.high)
            )
            reward += -jnp.float32(contain)
        for target in self.target_spaces:
            contain = jnp.all(
                jnp.logical_and(next_state >= target.low, next_state <= target.high)
            )
            reward += jnp.float32(contain)

        reward -= 2 * jnp.mean(jnp.abs(next_state / self.observation_space.high))

        next_packed = jnp.array([step + 1, next_state[0], next_state[1]])
        done = jnp.array(step >= self.episode_length, dtype=jnp.float32)
        truncation = jnp.where(
            done,
            jnp.ones_like(done, dtype=jnp.float32),
            _state.info['truncation']
        )
        info = {
            'rng': _state.info['rng'],
            'truncation': truncation,
            'steps': step + 1,
            'first_pipeline_state': _state.info['first_pipeline_state'],
            'first_obs': _state.info['first_obs'],
        }
        _state = _state.replace(
            pipeline_state=next_packed,
            obs=next_state,
            reward=reward,
            done=done,
            info=info
        )
        return _state

    @property
    def observation_size(self):
        return self.observation_space.shape[0]

    @property
    def action_size(self):
        return self.action_space.shape[0]

    @property
    def lipschitz_constant(self):
        return 1.2

    @property
    def backend(self):
        return None

    def integrate_noise(self, a: list, b: list):
        dims = 2
        pmass = np.ones(a[0].shape[0])
        for i in range(dims):
            loc = self.noise_bounds[0][i]
            scale = self.noise_bounds[1][i] - self.noise_bounds[0][i]
            marginal_pmass = triang.cdf(b[i], c=0.5, loc=loc, scale=scale) - triang.cdf(
                a[i], c=0.5, loc=loc, scale=scale
            )
            pmass *= marginal_pmass
        return pmass

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng) -> State:
        rng0, rng1 = jax.random.split(rng, 2)
        obs = jax.random.uniform(
            rng0,
            shape=(self.observation_space.shape[0],),
            minval=self.observation_space.low,
            maxval=self.observation_space.high,
        )
        state = jnp.array([0, obs[0], obs[1]])
        metrics = {}

        info = {
            'rng': rng1,
            'truncation': jnp.zeros((), dtype=jnp.float32),
            'steps': jnp.zeros((), dtype=jnp.int32),
            'first_pipeline_state': state,
            'first_obs': obs,
        }
        return State(
            pipeline_state=state,
            obs=obs,
            reward=jnp.zeros((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.float32),
            metrics=metrics,
            info=info
        )



class b2InvertedPendulum(Env):
    name = "b2pend"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.episode_length = 200

        init = np.array([0.3, 0.3], np.float32)
        self.init_spaces = [spaces.Box(low=-init, high=init, dtype=np.float32)]
        init = np.array([-1, 1], np.float32)
        self.init_spaces_train = [spaces.Box(low=-init, high=init, dtype=np.float32)]

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.noise = np.array([0.02, 0.01])

        safe = np.array([0.2, 0.2], np.float32)
        self.target_spaces = [spaces.Box(low=-safe, high=safe, dtype=np.float32)]
        safe = np.array([0.1, 0.1], np.float32)
        self.target_space_train = spaces.Box(low=-safe, high=safe, dtype=np.float32)

        observation_space = np.array([0.7, 0.7], np.float32)
        self.observation_space = spaces.Box(
            low=-observation_space, high=observation_space, dtype=np.float32
        )

        self.unsafe_spaces = [
            spaces.Box(
                low=self.observation_space.low,
                high=np.array([self.observation_space.low[0] + 0.1, 0.0]),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([self.observation_space.high[0] - 0.1, 0.0]),
                high=self.observation_space.high,
                dtype=np.float32,
            ),
        ]

        self._jax_rng = jax.random.PRNGKey(777)
        self.v_next = jax.vmap(self.next, in_axes=(0, 0), out_axes=0)
        self.v_step = jax.jit(jax.vmap(self.step))
        self.v_reset = jax.jit(jax.vmap(self.reset))

    @partial(jax.jit, static_argnums=(0,))
    def next(self, state, action):
        th, thdot = state  # th := theta
        max_speed = 5
        dt = 0.05
        g = 10
        m = 0.15
        l = 0.5
        b = 0.1

        u = 2 * jnp.clip(action, -1, 1)[0]
        newthdot = (1 - b) * thdot + (
                -3 * g * 0.5 / (2 * l) * jnp.sin(th + jnp.pi) + 3.0 / (m * l ** 2) * u
        ) * dt
        newthdot = jnp.clip(newthdot, -max_speed, max_speed)
        newth = th + newthdot * dt

        newth = jnp.clip(
            newth, self.observation_space.low[0], self.observation_space.high[0]
        )
        newthdot = jnp.clip(
            newthdot, self.observation_space.low[1], self.observation_space.high[1]
        )
        return jnp.array([newth, newthdot])

    def add_noise(self, state):
        self._jax_rng, rng = jax.random.split(self._jax_rng, 2)
        noise = triangular(rng, (self.observation_space.shape[0],))
        noise = noise * self.noise
        return state + noise

    @partial(jax.jit, static_argnums=(0,))
    def step(self, _state, action) -> State:
        state = _state.pipeline_state
        step = state[0]
        state = state[1:3]
        next_state = self.next(state, action)
        
        rng0, rng1 = jax.random.split(_state.info['rng'], 2)
        _state.info['rng'] = rng1
        
        noise = triangular(rng0, (self.observation_space.shape[0],))
        noise = noise * self.noise
        next_state = next_state + noise
        next_state = jnp.clip(
            next_state, self.observation_space.low, self.observation_space.high
        )
        
        reward = jnp.zeros((), dtype=jnp.float32)
        for unsafe in self.unsafe_spaces:
            contain = jnp.all(
                jnp.logical_and(state >= unsafe.low, state <= unsafe.high)
            )
            reward += -10 * jnp.float32(contain)
            # next_step += 200 * jnp.int32(contain)
        for target in self.target_spaces:
            contain = jnp.all(
                jnp.logical_and(state >= target.low, state <= target.high)
            )
            center = 0.5 * (target.low + target.high)
            dist = jnp.sum(jnp.abs(center - next_state))
            dist = jnp.clip(dist, 0, 2)
            reward += 2 * (2.0 - dist)
            reward += jnp.float32(contain)
        
        next_step = jnp.minimum(step + 1, self.episode_length)
        
        next_packed = jnp.array([next_step] + [next_state[i] for i in range(2)])
        done = jnp.array(step >= self.episode_length, dtype=jnp.float32)
        truncation = jnp.where(
            done,
            jnp.ones_like(done, dtype=jnp.float32),
            _state.info['truncation']
        )
        info = {
            'rng': _state.info['rng'],
            'truncation': truncation,
            'steps': step + 1,
            'first_pipeline_state': _state.info['first_pipeline_state'],
            'first_obs': _state.info['first_obs'],
        }
        _state = _state.replace(
            pipeline_state=next_packed,
            obs=next_state,
            reward=reward,
            done=done,
            info=info
        )
        return _state


    @property
    def observation_size(self):
        return self.observation_space.shape[0]

    @property
    def action_size(self):
        return self.action_space.shape[0]

    @property
    def noise_bounds(self):
        return -self.noise, self.noise

    @property
    def lipschitz_constant(self):
        return 1.78

    @property
    def lipschitz_constant_linf(self):
        return 1.78

    @property
    def backend(self):
        return None

    @property
    def delta(self):
        return 0.1 + self.noise[0]

    def integrate_noise(self, a: list, b: list):
        dims = 2
        pmass = np.ones(a[0].shape[0])
        for i in range(dims):
            loc = self.noise_bounds[0][i]
            scale = self.noise_bounds[1][i] - self.noise_bounds[0][i]
            marginal_pmass = triang.cdf(b[i], c=0.5, loc=loc, scale=scale) - triang.cdf(
                a[i], c=0.5, loc=loc, scale=scale
            )
            pmass *= marginal_pmass
        return pmass

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng) -> State:
        rng0, rng1 = jax.random.split(rng, 2)
        rng1, rng2 = jax.random.split(rng1, 2)

        init_samples = jnp.array([space.sample() for space in self.init_spaces])
        i = jax.random.randint(rng2, (), 0, len(self.init_spaces))
        obs = init_samples[i]
        state = jnp.array([0, obs[0], obs[1]])
        metrics = {}

        info = {
            'rng': rng1,
            'truncation': jnp.zeros((), dtype=jnp.float32),
            'steps': jnp.zeros((), dtype=jnp.int32),
            'first_pipeline_state': state,
            'first_obs': obs,
        }
        return State(
            pipeline_state=state,
            obs=obs,
            reward=jnp.zeros((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.float32),
            metrics=metrics,
            info=info
        )

class b2LDSEnv(Env):
    name = f"b2lds"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.has_render = False
        self.episode_length = 200

        safe = np.array([0.2, 0.2], np.float32)
        self.target_spaces = [spaces.Box(low=-safe, high=safe, dtype=np.float32)]
        self.init_spaces = [
            spaces.Box(
                low=np.array([-0.25, -0.1]),
                high=np.array([-0.2, 0.1]),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([0.2, -0.1]),
                high=np.array([0.25, 0.1]),
                dtype=np.float32,
            ),
        ]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.5 * np.ones(2, dtype=np.float32),
            high=1.5 * np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )
        # self.noise = np.array([0.01, 0.005])
        self.noise = np.array([0.01, 0.005])
        self.unsafe_spaces = [
            spaces.Box(
                low=self.observation_space.low,
                high=np.array([self.observation_space.low[0] + 0.1, 0.0]),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([self.observation_space.high[0] - 0.1, 0.0]),
                high=self.observation_space.high,
                dtype=np.float32,
            ),
        ]

        self._jax_rng = jax.random.PRNGKey(777)
        self.v_next = jax.vmap(self.next, in_axes=(0, 0), out_axes=0)
        self.v_step = jax.jit(jax.vmap(self.step))
        self.v_reset = jax.jit(jax.vmap(self.reset))

    @property
    def noise_bounds(self):
        return -self.noise, self.noise

    @property
    def observation_size(self):
        return self.observation_space.shape[0]

    @property
    def action_size(self):
        return self.action_space.shape[0]

    @property
    def backend(self):
        return None

    @partial(jax.jit, static_argnums=(0,))
    def next(self, state, action):
        action = jnp.clip(action, -1, 1)
        new_y = state[1] * 0.9 + action[0] * 0.5
        new_x = state[0] * 1.0 + state[1] * 0.045 + action[0] * 0.45
        new_y = jnp.clip(
            new_y, self.observation_space.low[1], self.observation_space.high[1]
        )
        new_x = jnp.clip(
            new_x, self.observation_space.low[0], self.observation_space.high[0]
        )
        return jnp.array([new_x, new_y])

    def add_noise(self, state):
        self._jax_rng, rng = jax.random.split(self._jax_rng, 2)
        noise = triangular(rng, (self.observation_space.shape[0],))
        noise = noise * self.noise
        return state + noise

    @partial(jax.jit, static_argnums=(0,))
    def step(self, _state, action) -> State:  # pytype: disable=signature-mismatch

        state = _state.pipeline_state
        step = state[0]
        state = state[1:3]
        next_state = self.next(state, action)

        rng0, rng1 = jax.random.split(_state.info['rng'], 2)
        _state.info['rng'] = rng1

        noise = triangular(rng0, (self.observation_space.shape[0],))
        noise = noise * self.noise
        next_state = next_state + noise
        next_state = jnp.clip(
            next_state, self.observation_space.low, self.observation_space.high
        )
        reward = jnp.zeros((), dtype=jnp.float32)
        for unsafe in self.unsafe_spaces:
            contain = jnp.all(
                jnp.logical_and(next_state >= unsafe.low, next_state <= unsafe.high)
            )
            reward += -jnp.float32(contain)
        for target in self.target_spaces:
            contain = jnp.all(
                jnp.logical_and(next_state >= target.low, next_state <= target.high)
            )
            reward += jnp.float32(contain)

        reward -= 2 * jnp.mean(jnp.abs(next_state / self.observation_space.high))

        next_packed = jnp.array([step + 1, next_state[0], next_state[1]])
        done = jnp.array(step >= self.episode_length, dtype=jnp.float32)
        truncation = jnp.where(
            done,
            jnp.ones_like(done, dtype=jnp.float32),
            _state.info['truncation']
        )
        info = {
            'rng': _state.info['rng'],
            'truncation': truncation,
            'steps': step + 1,
            'first_pipeline_state': _state.info['first_pipeline_state'],
            'first_obs': _state.info['first_obs'],
        }
        _state = _state.replace(
            pipeline_state=next_packed,
            obs=next_state,
            reward=reward,
            done=done,
            info=info
        )
        return _state

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng) -> State:
        rng0, rng1 = jax.random.split(rng, 2)
        obs = jax.random.uniform(
            rng0,
            shape=(self.observation_space.shape[0],),
            minval=self.observation_space.low,
            maxval=self.observation_space.high,
        )
        state = jnp.array([0, obs[0], obs[1]])
        metrics = {}

        info = {
            'rng': rng1,
            'truncation': jnp.zeros((), dtype=jnp.float32),
            'steps': jnp.zeros((), dtype=jnp.int32),
            'first_pipeline_state': state,
            'first_obs': obs,
        }
        return State(
            pipeline_state=state,
            obs=obs,
            reward=jnp.zeros((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.float32),
            metrics=metrics,
            info=info
        )

    @property
    def lipschitz_constant(self):
        A = np.max(np.sum(np.array([[1, 0.045, 0.45], [0, 0.9, 0.5]]), axis=0))
        return A

    @property
    def lipschitz_constant_linf(self):
        A = np.max(np.sum(np.array([[1, 0.045, 0.45], [0, 0.9, 0.5]]), axis=1))
        return A

    @property
    def delta(self):
        return 0.1 + self.noise[0]

    def integrate_noise(self, a: list, b: list):
        dims = 2
        pmass = np.ones(a[0].shape[0])
        for i in range(dims):
            loc = self.noise_bounds[0][i]
            scale = self.noise_bounds[1][i] - self.noise_bounds[0][i]
            marginal_pmass = triang.cdf(b[i], c=0.5, loc=loc, scale=scale) - triang.cdf(
                a[i], c=0.5, loc=loc, scale=scale
            )
            pmass *= marginal_pmass
        return pmass



class b2LDSS(Env):

    name = "b2ldss"

    def __init__(self, num_dims=2):
        assert num_dims % 2 == 0
        self.has_render = False
        self.n = 2
        self._fig_id = 0
        self.num_dims = num_dims
        self.episode_length = 200

        safe = np.array([0.2] * num_dims, np.float32)
        self.target_spaces = [spaces.Box(low=-safe, high=safe, dtype=np.float32)]

        self.init_spaces = [
            spaces.Box(
                low=np.array([-0.25] * (num_dims // 2) + [-0.1] * (num_dims // 2)),
                high=np.array([-0.2] * (num_dims // 2) + [0.1] * (num_dims // 2)),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([0.2] * (num_dims // 2) + [-0.1] * (num_dims // 2)),
                high=np.array([0.25] * (num_dims // 2) + [0.1] * (num_dims // 2)),
                dtype=np.float32,
            ),
        ]

        self.action_space = spaces.Box(low=-1, high=1, shape=(num_dims // 2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.5 * np.ones(num_dims, dtype=np.float32),
            high=1.5 * np.ones(num_dims, dtype=np.float32),
            dtype=np.float32,
        )
        self.noise = np.array([0.01] * (num_dims // 2) + [0] * (num_dims // 2))
        self.unsafe_spaces = [
            spaces.Box(
                low=self.observation_space.low,
                high=np.array([self.observation_space.low[i_] + 0.1 for i_ in range(num_dims // 2)]
                              + [0] * (num_dims // 2)),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([self.observation_space.high[i_] - 0.1 for i_ in range(num_dims // 2)]
                             + [0] * (num_dims // 2)),
                high=self.observation_space.high,
                dtype=np.float32,
            ),
        ]

        self._jax_rng = jax.random.PRNGKey(777)
        self.v_next = jax.vmap(self.next, in_axes=(0, 0), out_axes=0)
        self.v_step = jax.jit(jax.vmap(self.step))
        self.v_reset = jax.jit(jax.vmap(self.reset))

    @partial(jax.jit, static_argnums=(0,))
    def next(self, state, action):
        action = jnp.clip(action, -1, 1)
        x, vx = jnp.split(state, 2)

        new_vx = vx * 0.9 + action * 0.5
        new_x = x + vx * 0.045 + action * 0.45

        new_state = jnp.concatenate([new_x, new_vx])

        new_state = jnp.clip(
            new_state, self.observation_space.low, self.observation_space.high
        )
        return new_state

    def add_noise(self, state):
        self._jax_rng, rng = jax.random.split(self._jax_rng, 2)
        noise = triangular(rng, (self.observation_space.shape[0],))
        noise = noise * self.noise
        return state + noise

    @partial(jax.jit, static_argnums=(0,))
    def step(self, _state, action) -> State:
        state = _state.pipeline_state
        step = state[0]
        state = state[1:self.num_dims + 1]
        next_state = self.next(state, action)

        rng0, rng1 = jax.random.split(_state.info['rng'], 2)
        _state.info['rng'] = rng1

        noise = triangular(rng0, (self.observation_space.shape[0],))
        noise = noise * self.noise
        next_state = next_state + noise
        next_state = jnp.clip(
            next_state, self.observation_space.low, self.observation_space.high
        )

        reward = jnp.zeros((), dtype=jnp.float32)
        for unsafe in self.unsafe_spaces:
            contain = jnp.all(
                jnp.logical_and(state >= unsafe.low, state <= unsafe.high)
            )
            reward += -10 * jnp.float32(contain)
            # next_step += 200 * jnp.int32(contain)
        for target in self.target_spaces:
            contain = jnp.all(
                jnp.logical_and(state >= target.low, state <= target.high)
            )
            center = 0.5 * (target.low + target.high)
            dist = jnp.sum(jnp.abs(center - next_state))
            dist = jnp.clip(dist, 0, 2)
            reward += 2 * (2.0 - dist)
            reward += jnp.float32(contain)

        next_step = jnp.minimum(step + 1, self.episode_length)

        next_packed = jnp.array([next_step] + [next_state[i] for i in range(self.num_dims)])
        done = jnp.array(step >= self.episode_length, dtype=jnp.float32)
        truncation = jnp.where(
            done,
            jnp.ones_like(done, dtype=jnp.float32),
            _state.info['truncation']
        )
        info = {
            'rng': _state.info['rng'],
            'truncation': truncation,
            'steps': step + 1,
            'first_pipeline_state': _state.info['first_pipeline_state'],
            'first_obs': _state.info['first_obs'],
        }
        _state = _state.replace(
            pipeline_state=next_packed,
            obs=next_state,
            reward=reward,
            done=done,
            info=info
        )
        return _state

    @property
    def noise_bounds(self):
        return -self.noise[:self.num_dims // 2], self.noise[:self.num_dims // 2]

    @property
    def observation_size(self):
        return self.observation_space.shape[0]

    @property
    def action_size(self):
        return self.action_space.shape[0]

    @property
    def backend(self):
        return None

    @property
    def lipschitz_constant(self):
        A = np.max(np.sum(np.array([[1, 0.045, 0.45]] * (self.num_dims // 2) +
                                   [[0, 0.9, 0.5]] * (self.num_dims // 2)), axis=0))
        return A

    @property
    def lipschitz_constant_linf(self):
        A = np.max(np.sum(np.array([[1, 0.045, 0.45]] * (self.num_dims // 2) +
                                   [[0, 0.9, 0.5]] * (self.num_dims // 2)), axis=1))
        return A

    @property
    def delta(self):
        return 0.1 + self.noise[0]

    def integrate_noise(self, a: list, b: list):
        pmass = np.ones(a[0].shape[0])
        for i in range(self.num_dims // 2):
            loc = self.noise_bounds[0][i]
            scale = self.noise_bounds[1][i] - self.noise_bounds[0][i]
            marginal_pmass = triang.cdf(b[i], c=0.5, loc=loc, scale=scale) - triang.cdf(
                a[i], c=0.5, loc=loc, scale=scale
            )
            pmass *= marginal_pmass
        return pmass

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng) -> State:
        rng0, rng1 = jax.random.split(rng, 2)
        obs = jax.random.uniform(
            rng0,
            shape=(self.observation_space.shape[0],),
            minval=self.observation_space.low,
            maxval=self.observation_space.high,
        )
        state = jnp.array([0] + [obs[i] for i in range(self.num_dims)])
        metrics = {}

        info = {
            'rng': rng1,
            'truncation': jnp.zeros((), dtype=jnp.float32),
            'steps': jnp.zeros((), dtype=jnp.int32),
            'first_pipeline_state': state,
            'first_obs': obs,
        }
        return State(
            pipeline_state=state,
            obs=obs,
            reward=jnp.zeros((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.float32),
            metrics=metrics,
            info=info
        )

