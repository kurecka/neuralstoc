import jax
import jax.numpy as jnp
import chex
import numpy as np

import gc

from functools import partial

from neuralstoc.utils import (
    batch_apply,
    pretty_number,
    v_contains,
    v_intersect,
)

import time
import tqdm

Grid = chex.Array
SizedGrid = chex.Array


import logging
logger = logging.getLogger("neuralstoc")



# def max_dp_step(A, shift):
#     for axis in range(A.ndim):
#         indices = jnp.arange(A.shape[axis])
#         A = jnp.maximum(A, A.take(indices + shift, axis=axis))
#         # A = jnp.maximum(jnp.maximum(A, A.take(indices + shift, axis=axis)), A.take(jnp.clip(indices - shift, 0, None), axis=axis))

#     return A


# def axis_max(A, axis, L):
#     def cond(i, data):
#         A, l = data
#         return 2*l < L

#     def body(i, data):
#         A, l = data
#         A = jnp.maximum(A, A.take(jnp.arange(A.shape[axis]) + l, axis=axis))
#         return A, 2*l

#     A, l = jax.lax.while_loop(
#         cond,
#         body,
#         (A, 1)
#     )

#     offset = L - l
#     return jnp.maximum(A, A.take(jnp.arange(A.shape[axis]) + offset, axis=axis))


# def axis_max_with_target(A, axis, prev_len, target_len):
#     offset = target_len - prev_len
#     return jnp.maximum(A, A.take(jnp.arange(A.shape[axis]) + offset, axis=axis))


# @partial(jax.jit, static_argnums=(1,))
# def fast_window_max(A, k, ratios):
#     """
#     Produce array B of shape A.shape + (k,) such that B[x1, x2, ..., xn, i] = max(A[x1-2^i+1:x1+2^i-1, ..., xn-2^i+1:xn+2^i-1])
#     """
#     fratios = ratios / ratios.min()
#     iratios = jnp.ceil(fratios).astype(jnp.int32)
#     for a in range(A.ndim):
#         A = axis_max(A, a, iratios[a])

#     B = jnp.empty((k,) + A.shape, dtype=A.dtype)
#     B = B.at[0].set(A)

#     def body_fun(i, data):
#         B, L, lengths = data
#         target_lengths = jnp.ceil(2 * L * fratios).astype(jnp.int32)
#         B = B.at[i].set(B[i - 1])
#         for a in range(A.ndim):
#             B = B.at[i].set(axis_max_with_target(B[i], a, lengths[a], target_lengths[a]))
        
#         return B, 2 * L, target_lengths
#     B, _, _ = jax.lax.fori_loop(1, k, body_fun, (B, 1, iratios))
#     return B


def fft(a):
    """
    Fast Fourier Transform (FFT) for arbitrary number of dimensions.
    """
    for i in range(a.ndim):
        a = jnp.fft.fft(a, axis=i)
    return a


def ifft(a):
    """
    Inverse Fast Fourier Transform (IFFT) for arbitrary number of dimensions.
    """
    for i in range(a.ndim):
        a = jnp.fft.ifft(a, axis=i)
    return a


@jax.jit
def cross_corelation(a, kernel):
    """
    Compute the cross-correlation of a and kernel using FFT (i.e. the convolution of a and reversed kernel).
    The kernel is assumed to be smaller than a.
    """
    kernel_shape = kernel.shape
    kernel = jnp.pad(kernel, [(0, a.shape[i] - kernel.shape[i]) for i in range(a.ndim)], mode='constant')
    kernel = jnp.roll(jnp.flip(kernel, axis=tuple(range(kernel.ndim))), shift=1, axis=tuple(range(kernel.ndim)))
    fta = fft(a)
    ftb = fft(kernel)
    valid_slice = tuple(slice(0, a.shape[i] - kernel_shape[i] + 1) for i in range(a.ndim))
    return ifft(fta * ftb).real[valid_slice]


@partial(jax.jit, static_argnums=(0))
def refine_point(coeff, base_step, center, scale, ):
    """Refine a single point in the grid by creating `coeff^{env_dim}` new points around the center."""
    old_step = base_step * scale
    new_scale = scale / coeff
    new_step = base_step * new_scale
    new_centers = DecreaseVerifier.spanning_grid(
        center - old_step / 2 + new_step / 2,
        center + old_step / 2 - new_step / 2,
        size = (coeff,) * len(center),
    )
    new_scales = jnp.full(coeff**len(center), new_scale)
    return new_centers, new_scales


class DecreaseVerifier:
    """
    Provides functionality to verify decrease conditions for neural networks on a given environment.
    
    Features:
    - Chunked verification: Partitions the state space into chunks and cluster those source points
    that are mapped to the same chunk by the deterministic part of the environment dynamics.
    - Direct expectation computation: Original approach to compute expectations.
    - FFT based expectation computation: Uses Fast Fourier Transform to compute expectations.
    First, the approach evaluates the value function on a grid of points, then computes the cross-correlation
    with the noise kernel to obtain expectations in the grid points. The expectation centered between the grid points
    is then computed using multilinear interpolation.
    """

    def __init__(
            self, env, chunk_size,
            policy_apply, policy_ibp,
            value_apply, value_ibp,
            get_lipschitz,
            target_grid_size,
            spec,
            fft_threshold=100000,
            silent=False,
            noise_partition_cells=10,
            norm='linf',
        ):
        """
        Initialize the DecreaseVerifier.

        Args:
            env: The environment to verify decrease conditions on.
            chunk_size: Size of the chunks to partition the state space.
            policy_apply: JAX function to apply the policy.
            policy_ibp: JAX function for interval bound propagation of the policy.
            value_apply: JAX function to apply the value function.
            value_ibp: JAX function for interval bound propagation of the value function.
            get_lipschitz: Function to get the Lipschitz constant.
            target_grid_size: The target size of the source point grid.
            spec: Specification type (e.g., "reach_avoid", "reachability", "safety", "stability").
            fft_threshold: Threshold for the number of points in a bucket to use FFT for expectation computation.
            silent: If True, suppresses progress bars and logs.
            noise_partition_cells: Number of cells to partition noise into. This also determines the size of the grid over the state space chunk.
            norm: Norm type ('linf' or 'l1').
        """
        self.env = env
        self.chunk_size = chunk_size
        self.policy_apply = jax.jit(policy_apply)
        self.policy_ibp = jax.jit(policy_ibp)
        self.value_apply = jax.jit(value_apply)
        self.value_ibp = jax.jit(value_ibp)
        self.get_lipschitz = get_lipschitz
        self.silent = silent
        self.norm = norm
        self.spec = spec
        self.noise_partition_cells = noise_partition_cells
        self.grid_size = int(target_grid_size**(1 / env.observation_dim))
        self.fft_threshold = fft_threshold

        assert self.spec in ["reach_avoid", "reachability", "safety", "stability"], f"Unknown spec: {self.spec}"
    
        self.partition_noise()

        self._cached_filtered_grid = None
    
    def partition_noise(self):
        """
        Partition the noise space into cells and prepare the integration grid for expectation computation.

        Sets the following attributes:
        - noise_lb: Lower coordinates of the partitioned noise cells. Shape: (num_cells, env_dim).
        - noise_ub: Upper coordinates of the partitioned noise cells. Shape: (num_cells, env_dim).
        - noise_int_lb: Lower bounds on the probability mass of the noise cells. Shape: (noise_partition_cells,)*env_dim.
        - noise_int_ub: Upper bounds on the probability mass of the noise cells. Shape: (noise_partition_cells,)*env_dim.
        - integration_grid: Coordinates of the centers of grid points for integration. Shape: (size_1, size_{env_dim}, env_dim).
        - integration_grid_step: Step size of the integration grid in each dimension. Shape: (env_dim,).
        """
        noise_low, noise_high = self.env.noise_bounds

        # Partition the noise
        cell_width = (noise_high - noise_low) / self.noise_partition_cells
        num_cells = jnp.array(self.noise_partition_cells * jnp.ones(len(cell_width)), dtype=int)

        noise_vertices = self.grid2rectangular(self.spanning_grid(
            noise_low + 0.5 * cell_width,
            noise_high - 0.5 * cell_width,
            size=num_cells
        ), num_cells)

        grid_shape = tuple(noise_vertices.shape[:-1])
        env_dim = noise_vertices.shape[-1]

        self.noise_lb = noise_vertices.reshape((-1, env_dim)) - 0.5 * cell_width
        self.noise_ub = noise_vertices.reshape((-1, env_dim)) + 0.5 * cell_width

        self.noise_int_lb = self.noise_int_ub = self.env.integrate_noise(self.noise_lb.T, self.noise_ub.T)
        self.noise_int_lb = self.noise_int_lb.reshape(grid_shape)
        self.noise_int_ub = self.noise_int_ub.reshape(grid_shape)

        # Prepare the integartion grid
        self.integration_grid_step = (noise_high - noise_low) / self.noise_partition_cells

        self.integration_grid, ig_shape = self.spanning_grid_by_step(
            self.integration_grid_step/2 + noise_low,
            self.chunk_size + (noise_high - noise_low) + self.integration_grid_step/2,
            self.integration_grid_step,
            return_size=True,
        )
        self.integration_grid = self.grid2rectangular(self.integration_grid.astype(jnp.float16), ig_shape)
        self.integration_grid_size = self.integration_grid.shape[:-1]
        self.integration_grid = self.integration_grid.reshape((-1, env_dim))

        logger.info(f"Noise kernel shape: {self.noise_int_ub.shape}")
        logger.info(f"Integration grid size: {self.integration_grid_size}")
        logger.info(f"Integration grid step: {self.integration_grid_step}")
        logger.info(f"Total IG vertices: {np.prod(ig_shape)}")
        logger.info(f"Total IG memory: {np.prod(self.integration_grid.shape) * 2 / 1024**3}GB")

    def check_dec_cond(self, value_params, policy_params, value_bounds=None, couterexamples="all"):
        """
        Check decrease conditions for the given value and policy parameters.

        If `value_bounds` is provided, the function will check only the points that are within the bounds.
        If `couterexamples` is "hard", it will return only the hard violations.
        If `couterexamples` is "all", it will return all violations (both hard and soft).

        The source grid is represented by the centers of the grid cells `grid`, the base step size `base_step`,
        and the relative scales of the grid cells `scales`. I.e. the tuple `(grid, base_step, scales)` represents the grid of points
        with lower bounds `grid - scales * base_step / 2` and upper bounds `grid + scales * base_step / 2`.

        The function iteratively refines the grid until there is a hard violation or no soft violations are found.

        Args:
            value_params: Parameters of the value function.
            policy_params: Parameters of the policy function.
            value_bounds: Optional tuple of (lower_bound, upper_bound) for the value function.
            couterexamples: Specifies which counterexamples to return. Can be "hard" or "all".

        Returns:
            A tuple containing:
            - Number of soft violations
            - Number of hard violations
            - Maximum soft decrease
            - Maximum decay
            - Lower bound of the value of a violating state
            - States that violate the decrease conditions (either hard violations or all violations based on `couterexamples`).
        """
        grid, base_step = self.get_filtered_grid(self.grid_size)
        scales = np.ones(len(grid), dtype=np.float32)

        while True:
            (
                hard_decreases,
                soft_decreases,
                decays,
                values_lb,
                values_center,
                grid, scales
            ) = self.get_decreases(grid, scales, base_step, value_params, policy_params, value_bounds)

            soft_violation_mask = soft_decreases >= 0
            soft_violations_num = np.sum(soft_violation_mask)
            hard_violations_num = np.sum(hard_decreases >= 0)
            logger.info(
                f"Decrease condition check: {soft_violations_num} soft violations, {hard_violations_num} hard violations"
            )

            if hard_violations_num > 0:
                logger.info("Found hard violations, skipping refinement.")
                break
            elif soft_violations_num == 0:
                logger.info("No decrease condition violations found.")
                break

            logger.info("Refining grid...")
            grid, scales = self.refine_grid(
                grid[soft_violation_mask],
                scales[soft_violation_mask],
                base_step,
                soft_decreases[soft_violation_mask],
                hard_decreases[soft_violation_mask],
            )
        
        if couterexamples == "hard":
            violation_states = grid[hard_decreases >= 0]
        elif couterexamples == "all":
            violation_states = grid[soft_decreases >= 0]
        
        if soft_violations_num > 0:
            violation_value_lb = values_lb[soft_violation_mask].min()
        else:
            violation_value_lb = np.inf if value_bounds is None else value_bounds[1]


        return soft_violations_num, hard_violations_num, soft_decreases.max(), decays.max(), violation_value_lb, violation_states

    def refine_grid(self, grid, scales, base_step, soft_decreases, hard_decreases, max_refinement=4):
        """
        Refine the grid based on the soft and hard decreases.
        The refinement is done by creating new points around the centers of the grid points that have soft decreases and adjusting the scales accordingly.

        The width of the input grid cells is represented by `base_step * scales`.
        
        All points in `grid` are assumed to satisfy hard decrease condition, i.e. `hard_decreases < 0` but violate the soft decrease condition, i.e. `soft_decreases >= 0`.

        Args:
            grid: The centers of the grid cells.
            scales: The scales of the grid cells.
            base_step: The reference step size for the grid.
            soft_decreases: The soft decrease value for each grid point.
            hard_decreases: The hard decrease for each grid point.

        Returns:
            A tuple containing:
            - Refined grid centers
            - Refined scales for the grid cells.
        """

        coefficients = np.ceil((soft_decreases - hard_decreases) / -(hard_decreases - 1e7) * 1.1).astype(np.int32).clip(2, max_refinement)
        grid_list = []
        scales_list = []
        for coeff in np.unique(coefficients):
            coeff_mask = coefficients == coeff
            logger.info(f"Grid refinement {coeff}x: {coeff_mask.sum()} points")
            new_centers, new_scales = batch_apply(
                refine_point,
                grid[coeff_mask],
                scales[coeff_mask],
                params=(coeff, base_step),
                batch_size=1024*1024,
                output_num=2,
            )
            grid_list.append(new_centers.reshape(-1, grid.shape[1]))
            scales_list.append(new_scales.flatten())
        grid = np.concatenate(grid_list, axis=0)
        scales = np.concatenate(scales_list, axis=0)

        logger.info(f"Refined grid size: {grid.shape}")
        return grid, scales


    def value_lb(self, value_params, base_step, x, scale):
        """
        Use provided IBP function to compute the lower bound of the value function on a cell.
        """
        state_lb = x - base_step * scale / 2
        state_ub = x + base_step * scale / 2
        return self.value_ibp(
            value_params,
            (state_lb, state_ub),
        )[0].reshape()

    def get_decreases(self, grid, scales, base_step, value_params, policy_params, value_bounds):
        """
        Compute hard and soft decreases for the given grid points.

        Args:
            grid: The centers of the grid cells.
            scales: The scales of the grid cells.
            base_step: The reference step size for the grid.
            value_params: Parameters of the value function.
            policy_params: Parameters of the policy function.
            value_bounds: Optional tuple of (lower_bound, upper_bound) for the value function.

        Returns:
            A tuple containing:
            - Hard decrease values
            - Soft decrease values
            - Decays
            - Lower bounds of the value function at the grid points
            - Centered values of the value function at the grid points
            - Grid points
            - Scales of the grid cells
        """
        values_lb = batch_apply(
            self.value_lb,
            grid,
            scales,
            params=(value_params, base_step),
            batch_size=1024*1024,
        )

        if value_bounds is not None:
            mask = (values_lb >= value_bounds[0]) & (values_lb < value_bounds[1])
            grid = grid[mask]
            values_lb = values_lb[mask]
            scales = scales[mask]
        
        values_center = batch_apply(
            self.value_apply,
            grid,
            params=(value_params,),
            batch_size=1024*1024,
        ).flatten()

        hard_exp, soft_exp = self.get_next_expectations(
            policy_params,
            value_params,
            grid,
            scales,
            base_step,
        )

        hard_desc = hard_exp - values_center
        soft_desc = soft_exp - values_center

        decays = soft_exp / values_center

        return hard_desc, soft_desc, decays, values_center, values_lb, grid, scales

    def get_next_expectations(self, policy_params, value_params, states, scales, step):
        """
        Compute the next step expectations for the given states and scales using either FFT or direct method.
        """
        gc.collect()

        logger.info(f"Computing next step expectations for {len(states)} points...")
        logger.info(f" - Compute actions")
        actions = batch_apply(
            self.policy_apply,
            states,
            params=(policy_params,),
            batch_size=1024*1024,
        )

        logger.info(f" - Compute next states")
        next_states = self.env.v_next(states, actions)
        del actions

        global_corner = next_states.min(axis=0)
        bucket_indices = np.floor((next_states-global_corner) / self.chunk_size)
        del next_states

        logger.info(f" - Compute order")
        t = time.time()
        ord = np.lexsort(np.flip(bucket_indices, -1).T)
        logger.info(f"Took {time.time() - t:.2f} seconds to compute order")
        t = time.time()
        logger.info(f" - Sort arrays")
        states = states[ord]
        bucket_indices = bucket_indices[ord]
        scales = scales[ord]
        logger.info(f"Took {time.time() - t:.2f} seconds to sort arrays")

        hard_expectations = np.zeros(len(states))
        soft_expectations = np.zeros(len(states))

        K = self.get_lipschitz()
        logger.info(f" - Compute buckets")
        buckets, starts, counts = np.unique(bucket_indices, return_index=True, return_counts=True, axis=0)
        skipped = np.zeros(len(states), dtype=bool)

        logger.info(f"Total number of buckets: {len(buckets)}")
        logger.info(f"Number of points in each bucket: {counts}")
        logger.info(f"Max bucket index: {buckets.max(axis=0)}")

        bar = tqdm.tqdm(total=len(list(filter(lambda c: c>self.fft_threshold, counts))), disable=self.silent)
        for bucket, start, count in zip(buckets, starts, counts):
            end = start + count
            if count <= self.fft_threshold:
                skipped[start:end] = True
                continue

            logger.info(f"{bucket}: {start}-{end} ({count})")
            bar.update(1)

            bucket_corner = bucket * self.chunk_size + global_corner
        
            hard_exp_batch, soft_exp_batch = self.bucket_compute_expectation_fft(
                bucket_corner,
                policy_params,
                value_params,
                states[start:end],
                scales[start:end],
                step,
                K,
            )
            hard_expectations[start:end] = hard_exp_batch
            soft_expectations[start:end] = soft_exp_batch
                
        bar.close()
        
        logger.info(f"Using direct method for {skipped.sum()} skipped points")

        if skipped.any():
            hard_exp_batch, soft_exp_batch = self.batch_compute_expectation_direct(
                policy_params,
                value_params,
                states[skipped],
                scales[skipped],
                step,
                K,
            )
            hard_expectations[skipped] = hard_exp_batch
            soft_expectations[skipped] = soft_exp_batch

        reversed_ord = np.argsort(ord)
        return hard_expectations[reversed_ord], soft_expectations[reversed_ord]

    def interpolate_multilinear(self, values: jnp.ndarray, point: jnp.ndarray):
        """
        Interpolate the value of the grid at the given point using multilinear interpolation.

        Args:
            values: A grid of values corresponding to a grid of points, where first point is (0, 0, ..., 0)
            and the spacing between points is defined by `self.integration_grid_step`. Shape is (size_1, size_2, ..., size_{env_dim}).
            point: The point at which to interpolate the value. Shape is (env_dim,).
        """
        step = self.integration_grid_step

        lb_indices = jnp.floor(point / step).astype(jnp.int32)
        ub_indices = lb_indices + 1
        lb_indices = jnp.clip(lb_indices, 0, jnp.array(values.shape) - 1)
        ub_indices = jnp.clip(ub_indices, 0, jnp.array(values.shape) - 1)

        indices_t = jnp.stack([lb_indices, ub_indices], axis=1)
        grid_indices = jnp.stack(jnp.meshgrid(*indices_t), axis=0).reshape(point.shape[0], -1)
        coordinates_t = indices_t * step[:, None]
        dists_t = jnp.abs(coordinates_t - point[:, None])
        sums = jnp.sum(dists_t, axis=1)
        weights_t = (dists_t / sums[:, None])[:, ::-1]
        grid_weights = jnp.stack(jnp.meshgrid(*weights_t), axis=0).reshape(point.shape[0], -1)
        grid_weights = jnp.prod(grid_weights, axis=0)

        return jnp.sum(values[tuple(grid_indices)] * grid_weights)

    @partial(jax.jit, static_argnums=(0,))
    def prepare_expectations(self, ig_values):
        return cross_corelation(ig_values, self.noise_int_ub)
    
    @partial(jax.jit, static_argnums=(0,))
    def ig_value_ub(self, value_params, bucket_corner, x):
        """
        Use provided IBP function to compute the upper bound of the value function on an integration grid cell.
        """
        state_lb = x + bucket_corner - self.integration_grid_step / 2
        state_ub = x + bucket_corner + self.integration_grid_step / 2
        return self.value_ibp(
            value_params,
            (state_lb, state_ub),
        )[1].reshape()

    def bucket_compute_expectation_fft(self, bucket_corner, policy_params, value_params, grid, scales, base_step, K):
        """
        Compute the next step expectations for the given grid and scales using FFT method.

        Args:
            bucket_corner: The corner of the current bucket in the integration grid.
            policy_params: Parameters of the policy function.
            value_params: Parameters of the value function.
            grid: The centers of the grid cells for which to compute next state expectations.
            scales: The scales of the grid cells.
            base_step: The reference step size for the grid.
            K: Lipschitz constant.
        
        Returns:
            A tuple containing:
            - Hard expectations
            - Soft expectations
        """
        t = time.time()
        ig_values = batch_apply(
            self.ig_value_ub,
            self.integration_grid,
            params=(value_params,bucket_corner),
            batch_size=1024*1024,
        )

        ig_values = ig_values.reshape(self.integration_grid_size)
        logger.info(f"Prepare values time: {time.time() - t}")
        t = time.time()

        ig_expectations = self.prepare_expectations(ig_values)
        del ig_values

        logger.info(f"Prepare expectations time: {time.time() - t}")
        t = time.time()

        exp_hard, exp_soft = batch_apply(
            self.compute_expectation_fft,
            grid,
            scales,
            params=(policy_params, value_params, ig_expectations, bucket_corner, K, base_step),
            batch_size=1024*1024,
            output_num=2,
        )

        logger.info(f"Compute expectations time: {time.time() - t}")
        return exp_hard, exp_soft

    @partial(jax.jit, static_argnums=(0,))
    def compute_expectation_fft(self, policy_params, value_params, ig_expectations, bucket_corner, K, base_step, state, scale):
        """Compute the next step expectation for a single state using FFT method."""
        action = self.policy_apply(policy_params, state)
        next_state_mean = self.env.next(state, action)

        ExpV_xPlus = self.interpolate_multilinear(ig_expectations, next_state_mean - bucket_corner)

        eps = base_step * scale / 2
        radius = self.eps2radii(eps, self.norm)

        ExpV_xPlus_soft = ExpV_xPlus + radius * K

        return ExpV_xPlus, ExpV_xPlus_soft

    def batch_compute_expectation_direct(self, policy_params, value_params, grid, scales, base_step, K):
        """
        Compute the next step expectations for the given grid and scales using direct method.
        Args:
            policy_params: Parameters of the policy function.
            value_params: Parameters of the value function.
            grid: The centers of the grid cells for which to compute expectations.
            scales: The scales of the grid cells.
            base_step: The reference step size for the grid.
            K: Lipschitz constant.
        Returns:
            A tuple containing:
            - Hard expectations
            - Soft expectations
        """
        return batch_apply(
            self.compute_expectation,
            grid,
            scales,
            params=(policy_params, value_params, K, base_step),
            batch_size=1024*2,
            output_num=2,
        )

    @partial(jax.jit, static_argnums=(0,))
    def compute_expectation(self, policy_params, value_params, K, base_step, state, scale):
        """Compute the next step expectation for a single state using direct method."""
        action = self.policy_apply(policy_params, state)
        next_state = self.env.next(state, action)

        V_noise_cell_ub = self.value_ibp(
            value_params,
            (self.noise_lb + next_state, self.noise_ub + next_state),
        )[1]
        ExpV_xPlus = jnp.dot(V_noise_cell_ub.ravel(), self.noise_int_ub.ravel())

        eps = base_step * scale / 2
        radius = self.eps2radii(eps, self.norm)

        ExpV_xPlus_soft = ExpV_xPlus + radius * K

        return ExpV_xPlus, ExpV_xPlus_soft

    @staticmethod
    def ticks2grid(ticks):
        """
            Generate grid points from ticks.

            Example:
                ticks = [jnp.array([0, 1, 2]), jnp.array([0, 1])]
                grid = self.ticks2grid(ticks)
                # grid will be [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]

            Args:
                List of arrays representing the ticks in each dimension.
            Return:
                Grid points. Shape: (product(size), len(ticks)).
        """
        grid = jnp.vstack(list(map(jnp.ravel, jnp.meshgrid(*ticks)))).T
        return grid

    @staticmethod
    def spanning_grid(low, high, size, return_step=False):
        """
            Generate grid points spanning the given range and the grid size in each dimension.

            Example:
                low = jnp.array([0, 0])
                high = jnp.array([1, 2])
                size = (2, 3)
                grid = self.spanning_grid(low, high, size)
                # grid will be [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]] 

            Args:
                low: Lower bounds of the grid.
                high: Upper bounds of the grid.
                size: Number of points in each dimension.
            
            Returns:
                Grid points. Shape: (product(size), space_dim).
        """
        ticks = [jnp.linspace(low[i], high[i], size[i], endpoint=True) for i in range(len(size))]
        if return_step:
            step = np.array([ticks[i][1] - ticks[i][0] for i in range(len(size))])
            return DecreaseVerifier.ticks2grid(ticks), step
        else:
            return DecreaseVerifier.ticks2grid(ticks)

    @staticmethod
    def spanning_grid_by_step(low, high, step, return_size=False):
        """
            Generate grid points spanning the given range with specified step sizes.

            Example:
                low = jnp.array([0, 0])
                high = jnp.array([1, 2])
                step = jnp.array([1, 1])
                grid = self.spanning_grid_by_step(low, high, step)
                # grid will be [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]

            Args:
                low: Lower bounds of the grid.
                high: Upper bounds of the grid.
                step: Step sizes in each dimension.
            
            Returns:
                Grid points. Shape is (product(size), space_dim).
        """
        ticks = [jnp.arange(low[i], high[i] + step[i] / 2, step[i]) for i in range(len(step))]
        if return_size:
            size = tuple(map(len, ticks))
            return DecreaseVerifier.ticks2grid(ticks), size
        else:
            return DecreaseVerifier.ticks2grid(ticks)
    

    @staticmethod
    def grid2rectangular(grid, size):
        """
            Reorder and reshape a grid of centers to that the coordinates correspond to their indices in a multi-dimensional array.

            Example:
                grid = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                size = (2, 2)
                rectangular_grid = self.grid2rectangular(grid, size)
                # rectangular_grid will be [[[0, 0], [1, 0]], [[0, 1], [1, 1]]]

            Args:
                grid: Grid points. Shape is (product(size), space_dim).
                size: Size of the grid in each dimension.
            
            Returns:
                Rectangular grid points. Shape is (size[0], size[1], ..., size[-1]).
        """
        order = jnp.lexsort(jnp.flip(grid, -1).T)
        return grid[order].reshape(tuple(size) + grid.shape[1:])
    
    @staticmethod
    def eps2radii(step, norm):
        """Convert radius vector of a grid cell into radius in the specified norm."""

        if norm == 'linf':
            return step.max()
        elif norm == 'l1':
            return step.sum()

    def get_filtered_grid(self, n=100):
        """
        Generate a grid over the state space, filtered based on the specification.
        
        Creates a uniform grid with n points in each dimension and filters out
        points based on the specification type (e.g., removing target points for 
        reachability specifications). The grid is cached for efficiency.
        
        Args:
            n: Number of points per dimension (default: 100)
            
        Returns:
            tuple: (grid, mask) where
                - grid: Array of filtered grid points
                - mask: Boolean mask indicating which points were kept
        """
        if self._cached_filtered_grid is not None:
            if n == self._cached_filtered_grid:
                logger.info(f"Using cached grid of n={n} ")
                return self._cached_filtered_grid[1], self._cached_filtered_grid[2]
            else:
                self._cached_filtered_grid = None
        import gc

        gc.collect()
        size_t = 4 * (n**self.env.observation_dim)
        dims = self.env.observation_space.shape[0]
        grid, steps = [], []
        for i in range(dims):
            samples, step = np.linspace(
                self.env.observation_space.low[i],
                self.env.observation_space.high[i],
                n,
                endpoint=False,
                retstep=True,
            )
            grid.append(samples)
            steps.append(step)
        logger.info(f"Allocating grid of n={n} ({pretty_number(size_t)} bytes) meshgrid with steps={steps}")
        grid = np.meshgrid(*grid)
        grid = [grid[i].flatten() + steps[i] / 2 for i in range(dims)]
        grid = np.stack(grid, axis=1)

        mask = np.zeros(grid.shape[0], dtype=np.bool_)
        if self.spec == "reach_avoid":
            for target_space in self.env.target_spaces:
                contains = v_contains(target_space, grid)
                mask = np.logical_or(
                    mask,
                    contains,
                )
            for unsafe_space in self.env.unsafe_spaces:
                contains = v_contains(unsafe_space, grid)
                mask = np.logical_or(
                    mask,
                    contains,
                )
        elif self.spec == "reachability":
            for target_space in self.env.target_spaces:
                contains = v_contains(target_space, grid)
                mask = np.logical_or(
                    mask,
                    contains,
                )
        elif self.spec == "safety":
            for unsafe_space in self.env.unsafe_spaces:
                contains = v_contains(unsafe_space, grid)
                mask = np.logical_or(
                    mask,
                    contains,
                )
        filtered_grid = grid[np.logical_not(mask)]
        steps = np.array(steps)
        self._cached_filtered_grid = (n, filtered_grid, steps)
        return filtered_grid, steps
    