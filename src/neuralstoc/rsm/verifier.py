import math
import os
import time
from functools import partial

from neuralstoc.utils import (
    pretty_time,
    pretty_number,
    compute_expected_l,
    v_contains,
    v_intersect,
    batch_apply,
)
from neuralstoc.rsm.lipschitz import lipschitz_l1_jax
from neuralstoc.rsm.train_buffer import TrainBuffer

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import tensorflow as tf
import jax.numpy as jnp

from tqdm import tqdm
import numpy as np

import logging
logger = logging.getLogger("neuralstoc")


def get_n_for_bound_computation(obs_dim, co_factor=1):
    if obs_dim == 2:
        n = int(200 * co_factor)
    elif obs_dim == 3:
        n = int(150 * co_factor)
    elif obs_dim == 4:
        n = int(100 * co_factor)
    else:
        n = int(25 * co_factor)
    return n


class RSMVerifier:
    """
    RSMVerifier implements the verifier component of the learner-verifier framework.
    
    The verifier module is responsible for formally checking whether the neural supermartingale 
    certificate satisfies the required conditions to prove that a specification is met with 
    at least the desired probability threshold.
    
    The verification process consists of:
    1. State space discretization: Dividing the state space into a grid of cells
    2. Using interval arithmetic abstract interpretation (IAAI) to check certificate conditions
    3. Computing local Lipschitz constants of neural networks to bound approximation errors
    4. Generating counterexamples when verification fails, to guide further learning
    
    The verifier supports different types of certificate verification:
    - Reach-avoid supermartingales (RASMs) for reach-avoid specifications
    - Ranking supermartingales (RSMs) for reachability specifications
    - Stochastic barrier functions (SBFs) for safety specifications
    - Stabilizing ranking supermartingales (sRSMs) for stability specifications
    
    Attributes:
        learner: The RSMLearner module containing the neural networks to be verified
        env: The stochastic environment
        prob: The probability threshold for verification
        grid_size: The size of the discretization grid
        norm: The norm used for Lipschitz calculations ('l1' or 'linf')
        spec: The specification type ('reach_avoid', 'reachability', 'safety', or 'stability')
        train_buffer: Buffer containing counterexamples for training
        bound_co_factor: Co-factor for the bound computation grid size
    """
    
    def __init__(
        self,
        rsm_learner,
        env,
        batch_size,
        prob,
        target_grid_size,
        dataset_type="all",
        norm="linf",
        n_local=10,
        buffer_size=6_000_000,
        spec='reach_avoid',
        bound_co_factor=1,
    ):
        """
        Initialize the RSMVerifier module.
        
        Args:
            rsm_learner: RSMLearner module containing the neural networks to be verified
            env: The stochastic environment
            batch_size: Batch size for verification operations
            prob: Probability threshold for the specification (0 < prob <= 1)
            target_grid_size: Desired total number of cells in the discretization grid
            dataset_type: Type of counterexamples to add to training buffer:
                - "all": Add both hard and soft violations (recommended)
                - "hard": Add only hard violations
            norm: Norm for Lipschitz calculations ("l1" or "linf")
            n_local: Grid size for local Lipschitz constant computation
            buffer_size: Maximum size of the counterexample buffer
            spec: Specification type ('reach_avoid', 'safety', 'reachability', or 'stability')
            bound_co_factor: Co-factor for the bound computation grid size
        """
        self.learner = rsm_learner
        self.env = env
        self.norm = norm
        self.prob = jnp.float32(prob)
        self.n_local = n_local

        self.batch_size = batch_size
        self.block_size = 8 * batch_size
        self.refinement_enabled = True
        self.cached_lip_l_linf = None
        self.cached_lip_p_linf = None
        self.bound_co_factor = bound_co_factor
        target_grid_size = target_grid_size
        self.grid_size = int(math.pow(target_grid_size, 1 / env.observation_dim))
        self._cached_pmass_grid = self.learner._cached_pmass_grid
        self._cached_filtered_grid = None
        self._debug_violations = None
        self.dataset_type = dataset_type
        self.hard_constraint_violation_buffer = None
        self.unfiltered_grid_with_step_buffer = None
        self.train_buffer = TrainBuffer(max_size=buffer_size)
        self._perf_stats = {
            "apply": 0.0,
            "loop": 0.0,
        }
        self.v_get_grid_item = jax.vmap(
            self.get_grid_item, in_axes=(0, None), out_axes=0
        )
        self._grid_shuffle_rng = jax.random.PRNGKey(333)
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


    def assign_local_lip(self, grids, K_l):
        """
        Assign local Lipschitz constants to grid points based on their indices.
        
        Args:
            grids: Grid points to assign constants to
            K_l: Array of precomputed local Lipschitz constants
            
        Returns:
            ndarray: Local Lipschitz constants for each grid point
        """
        indices = (grids + self.env.observation_space.low) / (
                self.env.observation_space.high - self.env.observation_space.low
        )
        lip_index = (indices * self.n_local).astype(int)
        lip_index = np.sum(
            lip_index * np.array([self.n_local ** (i - 1) for i in range(self.env.observation_dim, 0, -1)]),
            axis=1).astype(
            int)
        K_l = K_l[lip_index]
        return K_l

    def vassign_local_lip(self, grids, K_l):
        """
        JAX-compatible version of assign_local_lip for vectorized operations.
        
        Args:
            grids: Grid points to assign constants to
            K_l: Array of precomputed local Lipschitz constants
            
        Returns:
            jnp.ndarray: Local Lipschitz constants for each grid point
        """
        K_l = jnp.asarray(K_l)
        indices = (grids + self.env.observation_space.low) / (
                self.env.observation_space.high - self.env.observation_space.low
        )
        lip_index = jnp.array((indices * self.n_local).astype(int))
        lip_index = jnp.sum(
            lip_index * jnp.array([self.n_local ** (i - 1) for i in range(self.env.observation_dim, 0, -1)]),
            axis=1).astype(
            int)
        K_l = K_l[lip_index]
        return K_l

    def prefill_train_buffer(self):
        """
        Fills the train buffer with a coarse grid
        """
        buffer_size = 1_000_000
        n = int(math.pow(buffer_size, 1 / self.env.observation_dim))
        # state_grid, _, _ = self.get_unfiltered_grid(n=n)
        state_grid, _ = self.get_filtered_grid(n=n)
        self.train_buffer.append(np.array(state_grid))
        return (
            self.env.observation_space.high[0] - self.env.observation_space.low[0]
        ) / n

    @partial(jax.jit, static_argnums=(0, 2))
    def get_grid_item(self, idx, n):
        """
        Maps an integer cell index and grid size to the bounds of the grid cell
        :param idx: Integer between 0 and n**obs_dim
        :param n: Grid size
        :return: jnp.ndarray corresponding to the center of the idx cell
        """
        dims = self.env.observation_dim
        target_points = [
            jnp.linspace(
                self.env.observation_space.low[i],
                self.env.observation_space.high[i],
                n,
                retstep=True,
                endpoint=False,
            )
            for i in range(dims)
        ]
        target_points, retsteps = zip(*target_points)
        target_points = list(target_points)
        for i in range(dims):
            target_points[i] = target_points[i] + 0.5 * retsteps[i]
        inds = []
        for i in range(dims):
            inds.append(idx % n)
            idx = idx // n
        return jnp.array([target_points[i][inds[i]] for i in range(dims)])

    def get_refined_grid_template(self, steps, n):
        """
        Refines a grid with resolution delta into n smaller grid cells.
        The returned template can be added to cells to create the smaller grid
        """
        dims = self.env.observation_dim
        grid, new_steps = [], []
        for i in range(dims):
            samples, new_step = jnp.linspace(
                -0.5 * steps[i],
                +0.5 * steps[i],
                n,
                endpoint=False,
                retstep=True,
            )
            grid.append(samples.flatten() + new_step * 0.5)
            new_steps.append(new_step)
        grid = jnp.meshgrid(*grid)
        grid = jnp.stack(grid, axis=1)
        return grid, np.array(new_steps)

    def get_unfiltered_grid_with_step(self):
        """
        Generate a full grid over the state space without filtering, with step size.
        
        This method creates a uniform grid covering the entire state space with
        n_local points in each dimension, returning both the grid points and the step size.
        The grid is cached for efficiency.
        
        Returns:
            tuple: (grid, steps) where
                - grid: Array of grid points
                - steps: Step size in each dimension
        """
        if self.unfiltered_grid_with_step_buffer is not None:
            return self.unfiltered_grid_with_step_buffer[0], self.unfiltered_grid_with_step_buffer[1]

        dims = self.env.observation_dim
        grid, steps = [], []
        for i in range(dims):
            samples, step = np.linspace(
                self.env.observation_space.low[i],
                self.env.observation_space.high[i],
                self.n_local,
                endpoint=False,
                retstep=True,
            )
            grid.append(samples)
            steps.append(step)
        grid = np.meshgrid(*grid)
        grid = [grid[i].flatten() + steps[i] / 2 for i in range(dims)]
        grid = np.stack(grid, axis=1)
        tmp = grid[:, 1].copy()
        grid[:, 1] = grid[:, 0]
        grid[:, 0] = tmp
        steps = np.array(steps)
        self.unfiltered_grid_with_step_buffer = (grid, steps)

        return grid, steps

    def get_unfiltered_grid(self, n=100):
        """
        Generate a full grid over the state space returning centers, lower and upper bounds.
        
        Creates a uniform grid with n points in each dimension and returns the
        centers, lower bounds, and upper bounds of each grid cell.
        
        Args:
            n: Number of points per dimension (default: 100)
            
        Returns:
            tuple: (grid_centers, grid_lb, grid_ub) where
                - grid_centers: Array of grid cell centers
                - grid_lb: Array of grid cell lower bounds
                - grid_ub: Array of grid cell upper bounds
        """
        dims = self.env.observation_dim
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
        grid = np.meshgrid(*grid)
        grid_lb = [x.flatten() for x in grid]
        grid_ub = [grid_lb[i] + steps[i] for i in range(dims)]
        grid_centers = [grid_lb[i] + steps[i] / 2 for i in range(dims)]

        grid_lb = np.stack(grid_lb, axis=1)
        grid_ub = np.stack(grid_ub, axis=1)
        grid_centers = np.stack(grid_centers, axis=1)
        return grid_centers, grid_lb, grid_ub

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
                logger.info(f"Using cached grid of n={n} ", end="", flush=True)
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

    def compute_bound_init(self, n):
        """
        Compute lower and upper bounds of the certificate function over the initial set.
        
        Uses interval arithmetic abstract interpretation to bound the values of the
        certificate function over the initial set.
        
        Args:
            n: Number of samples to use for the bound computation (a too high value will cause a long runtime or out-of-memory errors)
            
        Returns:
            tuple: (lb_init, ub_init) where
                - lb_init: Lower bound on certificate values in the initial set
                - ub_init: Upper bound on certificate values in the initial set
        """
        _, grid_lb, grid_ub = self.get_unfiltered_grid(n)

        mask = np.zeros(grid_lb.shape[0], dtype=np.bool_)
        # Include if the grid cell intersects with any of the init spaces
        for init_space in self.env.init_spaces:
            intersect = v_intersect(init_space, grid_lb, grid_ub)
            mask = np.logical_or(
                mask,
                intersect,
            )
        # Exclude if both lb AND ub are in the target set
        for target_space in self.env.target_spaces:
            contains_lb = v_contains(target_space, grid_lb)
            contains_ub = v_contains(target_space, grid_ub)
            mask = np.logical_and(
                mask, np.logical_not(np.logical_and(contains_lb, contains_ub))
            )

        grid_lb = grid_lb[mask]
        grid_ub = grid_ub[mask]
        assert grid_ub.shape[0] > 0

        return self.compute_bounds_on_set(grid_lb, grid_ub)

    def compute_bound_unsafe(self, n):
        """
        Compute lower and upper bounds of the certificate function over the unsafe set.
        
        Uses interval arithmetic abstract interpretation to bound the values of the
        certificate function over the unsafe set.
        
        Args:
            n: Number of samples to use for the bound computation (a too high value will cause a long runtime or out-of-memory errors)
            
        Returns:
            tuple: (lb_unsafe, ub_unsafe) where
                - lb_unsafe: Lower bound on certificate values in the unsafe set
                - ub_unsafe: Upper bound on certificate values in the unsafe set
        """
        _, grid_lb, grid_ub = self.get_unfiltered_grid(n)

        # Include only if either lb OR ub are in one of the unsafe sets
        mask = np.zeros(grid_lb.shape[0], dtype=np.bool_)
        for unsafe_spaces in self.env.unsafe_spaces:
            intersect = v_intersect(unsafe_spaces, grid_lb, grid_ub)
            mask = np.logical_or(
                mask,
                intersect,
            )
        grid_lb = grid_lb[mask]
        grid_ub = grid_ub[mask]
        assert grid_ub.shape[0] > 0
        return self.compute_bounds_on_set(grid_lb, grid_ub)

    def compute_bound_target(self, n):
        """
        Compute lower and upper bounds of the certificate function over the target set.
        
        Uses interval arithmetic abstract interpretation to bound the values of the
        certificate function over the target set.
        
        Args:
            n: Number of samples to use for the bound computation (a too high value will cause a long runtime or out-of-memory errors)
            
        Returns:
            tuple: (lb_target, ub_target) where
                - lb_target: Lower bound on certificate values in the target set
                - ub_target: Upper bound on certificate values in the target set
        """
        _, grid_lb, grid_ub = self.get_unfiltered_grid(n)

        # Include only if either lb OR ub are in one of the unsafe sets
        mask = np.zeros(grid_lb.shape[0], dtype=np.bool_)
        for target_spaces in self.env.target_spaces:
            intersect = v_intersect(target_spaces, grid_lb, grid_ub)
            mask = np.logical_or(
                mask,
                intersect,
            )
        grid_lb = grid_lb[mask]
        grid_ub = grid_ub[mask]
        assert grid_ub.shape[0] > 0
        return self.compute_bounds_on_set(grid_lb, grid_ub)

    def compute_bound_domain(self, n):
        """
        Compute lower and upper bounds of the certificate function over the entire domain.
        
        Uses interval arithmetic abstract interpretation to bound the values of the
        certificate function over the entire state space.
        
        Args:
            n: Number of samples to use for the bound computation (a too high value will cause a long runtime or out-of-memory errors)
            
        Returns:
            tuple: (lb_domain, ub_domain) where
                - lb_domain: Lower bound on certificate values in the entire domain
                - ub_domain: Upper bound on certificate values in the entire domain
        """
        _, grid_lb, grid_ub = self.get_unfiltered_grid(n)

        if self.spec == "reachability" or self.spec == "reach_avoid":
            # Exclude if both lb AND ub are in the target set
            mask = np.zeros(grid_lb.shape[0], dtype=np.bool_)
            for target_space in self.env.target_spaces:
                contains_lb = v_contains(target_space, grid_lb)
                contains_ub = v_contains(target_space, grid_ub)
                mask = np.logical_or(
                    mask,
                    np.logical_and(contains_lb, contains_ub),
                )
            mask = np.logical_not(
                mask
            )  # now we have all cells that have both lb and both in a target -> invert for filtering
            grid_lb = grid_lb[mask]
            grid_ub = grid_ub[mask]
        assert grid_ub.shape[0] > 0
        return self.compute_bounds_on_set(grid_lb, grid_ub)

    def compute_bounds_on_set(self, grid_lb, grid_ub):
        """
        Compute bounds of the certificate function on a set of grid cells.
        
        Uses the IBP model to compute lower and upper bounds on the certificate value
        for each of the provided grid cells.
        
        Args:
            grid_lb: Lower bounds of grid cells
            grid_ub: Upper bounds of grid cells
            
        Returns:
            tuple: (lb, ub) where
                - lb: Lower bounds on certificate values for each grid cell
                - ub: Upper bounds on certificate values for each grid cell
        """
        global_min = jnp.inf
        global_max = -jnp.inf
        for i in range(int(np.ceil(grid_ub.shape[0] / self.batch_size))):
            start = i * self.batch_size
            end = np.minimum((i + 1) * self.batch_size, grid_ub.shape[0])
            batch_lb = jnp.array(grid_lb[start:end])
            batch_ub = jnp.array(grid_ub[start:end])
            lb, ub = self.learner.v_ibp.apply(
                self.learner.v_state.params, [batch_lb, batch_ub]
            )
            global_min = jnp.minimum(global_min, jnp.min(lb))
            global_max = jnp.maximum(global_max, jnp.max(ub))
        return float(global_min), float(global_max)

    @partial(jax.jit, static_argnums=(0,))
    def _check_dec_batch(self, l_params, p_params, f_batch, l_batch, K):
        """
        Check the expected decrease condition for a batch of states.
        
        This JIT-compiled function checks whether the certificate function decreases
        in expectation for each state in the batch, accounting for Lipschitz error bounds.
        
        Args:
            l_params: Parameters of the certificate network
            p_params: Parameters of the controller network
            f_batch: Batch of grid points to check
            l_batch: Certificate values for the batch
            K: Lipschitz term(s)
            
        Returns:
            v: Number of violating cells
            violating_indices: Array indicating which points violate the condition
            hard_v: Number of hard violating cells
            hard_violating_indices: Array indicating which points violate the condition
            decrease: Maximum decrease value
            decay: Maximum decay factor
        """
        a_batch = self.learner.p_state.apply_fn(p_params, f_batch)
        pmass, batched_grid_lb, batched_grid_ub = self._cached_pmass_grid
        # e = self.compute_expected_l(
        e = compute_expected_l(
            self.env,
            self.learner.v_ibp.apply,
            l_params,
            f_batch,
            a_batch,
            pmass,
            batched_grid_lb,
            batched_grid_ub,
        )
        e = e.flatten()
        l_batch = l_batch.flatten()

        decrease = e + K - l_batch
        if self.spec == "safety":
            violating_indices = decrease > 0
        else:
            violating_indices = decrease >= 0
        v = violating_indices.astype(jnp.int32).sum()
        hard_violating_indices = e - l_batch >= 0
        hard_v = hard_violating_indices.astype(jnp.int32).sum()
        decay = (e + K) / l_batch
        return (
            v,
            violating_indices,
            hard_v,
            hard_violating_indices,
            jnp.max(decrease),
            jnp.max(decay),
        )

    @partial(jax.jit, static_argnums=(0,))
    def normalize_rsm(self, l, ub_init, domain_min, K, target_ub):
        """
        Normalize the certificate values to satisfy the bound conditions.
        
        Applies a linear transformation to certificate values to ensure they satisfy
        the conditions required by the supermartingale certificate.
        
        Args:
            l: Certificate values
            ub_init: Upper bound on certificate values in the initial set
            domain_min: Minimum certificate value in the domain
            K: Lipschitz term(s)
            target_ub: Upper bound on certificate values in the target set
            
        Returns:
            jnp.ndarray: Normalized certificate values
        """
        if self.spec == "safety" or self.spec == "reach_avoid" or self.spec == "reachability":
            l = l - domain_min
            ub_init = ub_init - domain_min
            # now min = 0
            l = l / jnp.maximum(ub_init, 1e-6)
            # now init max = 1
            K = K / jnp.maximum(ub_init, 1e-6)
        else:
            l = l - domain_min
            target_ub = target_ub - domain_min

            # now min = 0
            l = l / jnp.maximum(target_ub, 1e-6)
            # now target max = 1
            K = K / jnp.maximum(target_ub, 1e-6)
        return l, K

    def check_dec_cond(self, lipschitz_k, ra_bounds=None):
        """
        Check whether the certificate satisfies the expected decrease condition.
        
        A wrapper around check_dec_cond_full that calls the full implementation.
        
        Args:
            lipschitz_k: Lipschitz term(s) of the entire system (environment, controller, RSM)
            ra_bounds: Optional bounds for checking specific value ranges of the certificate function
            
        Returns:
            tuple containing verification results (see check_dec_cond_full)
        """
        return self.check_dec_cond_full(lipschitz_k, ra_bounds=ra_bounds)

    def check_dec_cond_full(self, lipschitz_k, ra_bounds=None):
        """
        Fully check whether the certificate satisfies the expected decrease condition.
        
        This method verifies if the certificate function decreases in expectation for
        all states in the domain (outside the target set), accounting for approximation
        errors using Lipschitz analysis.
        
        Args:
            lipschitz_k: Lipschitz term(s) of the entire system (environment, controller, RSM)
            ra_bounds: Optional bounds for checking specific value ranges of the certificate function
            
        Returns:
            tuple: (violations, hard_violations, max_decrease, max_decay, violation_min_val)
                - violations: Total number of states violating the condition
                - hard_violations: Number of states with significant violations
                - max_decrease: Maximum decrease value observed
                - max_decay: Maximum decay factor
                - violation_min_val: Minimum certificate value among violations
        """
        dims = self.env.observation_dim
        grid_total_size = self.grid_size**dims

        verify_start_time = time.time()
        n = get_n_for_bound_computation(self.env.observation_dim, self.bound_co_factor)
        _, ub_init = self.compute_bound_init(n)
        domain_min, _ = self.compute_bound_domain(n)
        _, ub_target = self.compute_bound_target(n)
        logger.info(f"computed bounds done: {pretty_time(time.time()-verify_start_time)}")

        grid, steps = self.get_filtered_grid(self.grid_size)
        logger.info(f"allocated grid done: {pretty_time(time.time()-verify_start_time)}")
        if self.norm == "l1":
            delta = 0.5 * np.sum(steps)
            # l1-norm of the half the grid cell (=l1 distance from center to corner)
        elif self.norm == "linf":
            delta = 0.5 * np.max(steps)
        else:
            raise ValueError("Should not happen")
        K = lipschitz_k * delta

        number_of_cells = self.grid_size**self.env.observation_dim
        logger.info(
            f"Checking GRID with {pretty_number(number_of_cells)} cells"
        )
        K = jnp.float32(K)

        violations = 0
        hard_violations = 0
        total_cells_processed = 0
        failed_fast = False
        max_decrease = -jnp.inf
        max_decay = -jnp.inf
        violation_buffer = []
        hard_violation_buffer = []
        violation_min_val = jnp.inf

        logger.info(f"loop start: {pretty_time(time.time()-verify_start_time)}")
        kernel_start_time = time.perf_counter()


        if self.spec == "reachability" or self.spec == "reach_avoid":
            logger.info("Filtering out cells with too high potential")
            v = batch_apply(self.learner.v_state.apply_fn, grid, params=(self.learner.v_state.params,), batch_size=1024*128).flatten()
            K_local = self.assign_local_lip(grid, K)
            normalized_l, normalized_K = self.normalize_rsm(v, ub_init, domain_min, K_local, ub_target)
            grid = grid[normalized_l - normalized_K < 1 / (1 - self.prob)]


        pbar = tqdm(total=grid.shape[0], unit="cells")
        for start in range(0, grid.shape[0], self.batch_size):
            end = min(start + self.batch_size, grid.shape[0])
            x_batch = jnp.array(grid[start:end])
            v_batch = self.learner.v_state.apply_fn(
                self.learner.v_state.params, x_batch
            ).flatten()
            if self.norm == "linf":
                K_batch = self.assign_local_lip(x_batch, K)
            else:
                K_batch = K
            # normalize the RSM to obtain slightly better values
            normalized_l_batch, K_batch_ = self.normalize_rsm(v_batch, ub_init, domain_min, K_batch, ub_target)
            if self.spec == "reachability" or self.spec == "reach_avoid":
                # Next, we filter the grid cells that are > 1/(1-p)
                if self.prob < 1.0:
                    if ra_bounds is None:
                        less_than_p = normalized_l_batch - K_batch_ < 1 / (1 - self.prob)
                    else:
                        less_than_p = (ra_bounds[0] <= (normalized_l_batch - K_batch_)) & ((normalized_l_batch - K_batch_) < ra_bounds[1])
                    if jnp.sum(less_than_p.astype(np.int32)) == 0:
                        # If all cells are filtered -> can skip the expectation computation
                        pbar.update(end - start)
                        continue
                    x_batch = x_batch[less_than_p]
                    v_batch = v_batch[less_than_p]
                    if self.norm == "linf":
                        K_batch = K_batch[less_than_p]
            elif self.spec == "stability":
                more_1 = normalized_l_batch - K_batch_ >= 1
                mask = np.zeros(x_batch.shape[0], dtype=np.bool_)
                for unsafe_space in self.env.unsafe_spaces:
                    contains = v_contains(unsafe_space, x_batch)
                    mask = np.logical_or(
                        mask,
                        contains,
                    )
                more_1 = np.logical_or(mask, more_1)
                if jnp.sum(more_1.astype(np.int32)) == 0:
                    # If all cells are filtered -> can skip the expectation computation
                    pbar.update(end - start)
                    continue
                x_batch = x_batch[more_1]
                v_batch = v_batch[more_1]
                if self.norm == "linf":
                    K_batch = K_batch[more_1]

            # Finally, we compute the expectation of the grid cell
            (
                v,
                violating_indices,
                hard_v,
                hard_violating_indices,
                decrease,
                decay,
            ) = self._check_dec_batch(
                self.learner.v_state.params,
                self.learner.p_state.params,
                x_batch,
                v_batch,
                K_batch,
            )
            if jnp.sum(violating_indices) > 0:
                violation_min_val = jnp.minimum(violation_min_val, v_batch[violating_indices].min())
            max_decrease = jnp.maximum(max_decrease, decrease)
            max_decay = jnp.maximum(max_decay, decrease)
            # Count the number of violations and hard violations
            hard_violations += hard_v
            violations += v
            if v > 0:
                violation_buffer.append(np.array(x_batch[violating_indices]))
            if hard_v > 0:
                hard_violation_buffer.append(np.array(x_batch[hard_violating_indices]))
            total_kernel_time = time.perf_counter() - kernel_start_time
            total_cells_processed = end
            kcells_per_sec = total_cells_processed / total_kernel_time / 1000
            pbar.update(end - start)
            pbar.set_description(
                f"{pretty_number(violations)}/{pretty_number(total_cells_processed)} cell violating @ {kcells_per_sec:0.1f} Kcells/s"
            )

        pbar.close()
        logger.info(f"loop ends: {pretty_time(time.time()-verify_start_time)}")
        if failed_fast:
            logger.info(
                f"Failed fast after {pretty_number(total_cells_processed)}/{pretty_number(number_of_cells)} cells checked"
            )
        if len(violation_buffer) == 1:
            logger.info(f"violation_buffer[0][0]:", violation_buffer[0][0])
        self.hard_constraint_violation_buffer = (
            None
            if len(hard_violation_buffer) == 0
            else np.concatenate([np.array(g) for g in hard_violation_buffer])
        )
        if self.dataset_type in ["all", "soft"]:
            self.train_buffer.extend(violation_buffer)
            self.train_buffer.extend(hard_violation_buffer)
        elif self.dataset_type == "hard":
            self.train_buffer.extend(hard_violation_buffer)
        else:
            raise ValueError(f"Unknown dataset type {self.dataset_type}")
        logger.info(
            f"Verified {pretty_number(total_cells_processed)} cells ({pretty_number(violations)} violations, {pretty_number(hard_violations)} hard) in {pretty_time(time.time()-verify_start_time)}"
        )

        if (
            self.refinement_enabled
            and hard_violations == 0
            and not failed_fast
            and violations > 0
            and len(violation_buffer) > 0
        ):
            logger.info(
                f"Zero hard violations -> refinement of {pretty_number(grid_total_size)} soft violations"
            )
            refine_start = time.time()
            refinement_buffer = [np.array(g) for g in violation_buffer]
            refinement_buffer = np.concatenate(refinement_buffer)
            success, max_decrease, max_decay, violation_min_val = self.refine_grid(
                refinement_buffer, lipschitz_k, steps, ub_init, domain_min, ub_target, ra_bounds
            )
            if success:
                logger.info(
                    f"Refinement successful! (took {pretty_time(time.time()-refine_start)})"
                )
                return 0, 0, float(max_decrease), float(max_decay), float(violation_min_val)
            else:
                logger.info(
                    f"Refinement unsuccessful! (took {pretty_time(time.time()-refine_start)})"
                )

        return violations, hard_violations, float(max_decrease), float(max_decay), float(violation_min_val)


    def refine_grid(self, refinement_buffer, lipschitz_k, steps, ub_init, domain_min, ub_target=None, ra_bounds=None):
        """
        Refine the grid around states that violate the certificate conditions.
        
        This method takes states with violations, creates a finer grid around them,
        and checks whether the finer grid points also violate the conditions. The
        new violations are added to the counterexample buffer for further training.
        
        Args:
            refinement_buffer: States to refine around
            lipschitz_k: Lipschitz term(s)
            steps: Current grid step size
            ub_init: Upper bound on certificate values in the initial set
            domain_min: Minimum certificate value in the domain
            ub_target: Optional upper bound on certificate values in the target set
            ra_bounds: Optional bounds for checking specific value ranges of the certificate function   
            
        Returns:
            int: Number of new violations found
        """
        n_dims = self.env.observation_dim

        n = 10
        if self.env.observation_dim > 3:
            n = 6            
            if refinement_buffer.shape[0] > 1e6:
                n = 3
        elif self.env.observation_dim == 3 and refinement_buffer.shape[0] > 5e5:
            n = 3
        template_batch, new_steps = self.get_refined_grid_template(steps, n)
        if self.norm == "l1":
            new_delta = 0.5 * np.sum(new_steps)
            # l1-norm of the half the grid cell (=l1 distance from center to corner)
        elif self.norm == "linf":
            new_delta = 0.5 * np.max(new_steps)
        else:
            raise ValueError("Should not happen")

        batch_size = self.batch_size // (n ** n_dims)

        K = jnp.float32(lipschitz_k * new_delta)
        template_batch = template_batch.reshape((1, -1, n_dims))
        max_decrease = -jnp.inf
        max_decay = -jnp.inf
        violation_min_val = jnp.inf
        for i in tqdm(range(int(np.ceil(refinement_buffer.shape[0] / batch_size)))):
            start = i * batch_size
            end = np.minimum((i + 1) * batch_size, refinement_buffer.shape[0])
            s_batch = jnp.array(refinement_buffer[start:end])
            s_batch = s_batch.reshape((-1, 1, n_dims))
            r_batch = s_batch + template_batch
            r_batch = r_batch.reshape((-1, self.env.observation_dim))  # flatten
            if self.norm == "linf":
                K_batch = self.assign_local_lip(r_batch, K)
            else:
                K_batch = K

            l_batch = self.learner.v_state.apply_fn(
                self.learner.v_state.params, r_batch
            ).flatten()
            normalized_l_batch, K_batch_ = self.normalize_rsm(l_batch, ub_init, domain_min, K_batch, ub_target)
            if self.spec == "reachability" or self.spec == "reach_avoid":
                # Next, we filter the grid cells that are > 1/(1-p)
                if self.prob < 1.0:
                    if ra_bounds is None:
                        less_than_p = normalized_l_batch - K_batch_ < 1 / (1 - self.prob)
                    else:
                        less_than_p = (ra_bounds[0] <= (normalized_l_batch - K_batch_)) & ((normalized_l_batch - K_batch_) < ra_bounds[1])
                    if jnp.sum(less_than_p.astype(np.int32)) == 0:
                        # If all cells are filtered -> can skip the expectation computation
                        continue
                    r_batch = r_batch[less_than_p]
                    l_batch = l_batch[less_than_p]
                    if self.norm == "linf":
                        K_batch = K_batch[less_than_p]
            elif self.spec == "stability":
                more_1 = normalized_l_batch - K_batch_ >= 1
                mask = np.zeros(r_batch.shape[0], dtype=np.bool_)
                for unsafe_space in self.env.unsafe_spaces:
                    contains = v_contains(unsafe_space, r_batch)
                    mask = np.logical_or(
                        mask,
                        contains,
                    )
                more_1 = np.logical_or(mask, more_1)
                if jnp.sum(more_1.astype(np.int32)) == 0:
                    # If all cells are filtered -> can skip the expectation computation
                    continue
                r_batch = r_batch[more_1]
                l_batch = l_batch[more_1]
                if self.norm == "linf":
                    K_batch = K_batch[more_1]

            (
                v,
                violating_indices,
                hard_v,
                hard_violating_indices,
                decrease,
                decay,
            ) = self._check_dec_batch(
                self.learner.v_state.params,
                self.learner.p_state.params,
                r_batch,
                l_batch,
                K_batch,
            )
            if jnp.sum(violating_indices) > 0:
                violation_min_val = jnp.minimum(violation_min_val, l_batch[violating_indices].min())
            max_decrease = jnp.maximum(max_decrease, decrease)
            max_decay = jnp.maximum(max_decay, decay)
            if v > 0:
                return False, max_decrease, max_decay, violation_min_val
        return True, max_decrease, max_decay, violation_min_val


    def get_unsafe_complement_grid(self, n=100):
        """
        Generate a grid covering the state space excluding unsafe regions.
        
        Args:
            n: Number of points per dimension (default: 100)
            
        Returns:
            ndarray: Grid points in the safe region of the state space
        """
        # print(f"Allocating grid of n={n} ", end="", flush=True)
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
        # print(f" meshgrid with steps={steps} ", end="", flush=True)
        logger.info(f"Allocating grid of n={n} meshgrid with steps={steps}")
        grid = np.meshgrid(*grid)
        grid = [grid[i].flatten() + steps[i] / 2 for i in range(dims)]
        grid = np.stack(grid, axis=1)

        mask = np.zeros(grid.shape[0], dtype=np.bool_)
        for unsafe_space in self.env.unsafe_spaces:
            contains = v_contains(unsafe_space, grid)
            mask = np.logical_or(
                mask,
                contains,
            )

        filtered_grid = grid[np.logical_not(mask)]

        return filtered_grid, np.array(steps)

    def get_mask_in_and_safe(self, grid):
        """
        Create a mask for points that are both in the domain and safe.
        
        Args:
            grid: Grid points to check
            
        Returns:
            ndarray: Boolean mask where True indicates points both in domain and safe
        """
        mask = np.zeros(grid.shape[0], dtype=np.bool_)
        for unsafe_space in self.env.unsafe_spaces:
            contains = v_contains(unsafe_space, grid)
            mask = np.logical_or(mask, contains)

        contains = v_contains(self.env.observation_space, grid)
        mask = np.logical_or(mask, np.logical_not(contains))
        return np.logical_not(mask)

    def get_unsafe_d_grid(self, d, n=100):
        """
        Generate a grid over states that are d-distance away from unsafe regions.
        
        Args:
            d: Distance parameter from unsafe regions
            n: Number of points per dimension (default: 100)
            
        Returns:
            ndarray: Grid points that are d-distance from unsafe regions
        """
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
        # print(f"Allocating grid of n={n} ", end="", flush=True)
        # print(f" meshgrid with steps={steps} ", end="", flush=True)
        logger.info(f"Allocating grid of n={n} meshgrid with steps={steps}")
        grid = np.meshgrid(*grid)
        grid = [grid[i].flatten() + steps[i] / 2 for i in range(dims)]
        grid = np.stack(grid, axis=1)

        mask = np.zeros(grid.shape[0], dtype=np.bool_)
        for unsafe_space in self.env.unsafe_spaces:
            contains = v_contains(unsafe_space, grid)
            mask = np.logical_or(
                mask,
                contains,
            )

        filtered_grid = grid[mask]

        mask = np.zeros(filtered_grid.shape[0], dtype=np.bool_)

        for i in range(dims):
            sh = np.zeros(dims, dtype=np.float32)
            sh[i] = d

            grid_n = filtered_grid - sh
            grid_p = filtered_grid + sh

            mask = np.logical_or(mask, self.get_mask_in_and_safe(grid_n))
            mask = np.logical_or(mask, self.get_mask_in_and_safe(grid_p))

        filtered_grid = filtered_grid[mask]

        return filtered_grid, np.array(steps)

    @partial(jax.jit, static_argnums=(0,))
    def get_big_delta(self, grids, steps, ub_target, lb_domain):
        """
        Compute the big Delta parameter for stability analysis.
        
        For stability specifications, this computes the parameter used to bound
        the expected time to stability.
        
        Args:
            grids: Grid points for computation
            steps: Grid step sizes
            ub_target: Upper bound on certificate values in the target set
            lb_domain: Lower bound on certificate values in the domain
            
        Returns:
            float: The big Delta parameter value
        """
        p_params = self.learner.p_state.params
        v_params = self.learner.v_state.params

        if self.norm == "l1":
            delta = 0.5 * jnp.sum(
                steps
            )
            lip_f = self.env.lipschitz_constant
            lip_p = lipschitz_l1_jax(p_params, obs_normalization=self.learner.obs_normalization)
            lip_norm = lip_f * (lip_p + 1) * delta
            next_max_norm = -jnp.inf
            ub_target = ub_target - lb_domain
            for start in range(0, grids.shape[0], self.batch_size):
                end = min(start + self.batch_size, grids.shape[0])
                v_batch = self.learner.v_state.apply_fn(v_params, grids[start:end]).flatten()
                v_batch = (v_batch - lb_domain) / jnp.maximum(ub_target, 1e-6)
                v_batch = jnp.ceil(jnp.clip(jnp.maximum(1 - v_batch, 0), max=1))
                next_det_grids = self.env.v_next(grids[start:end],
                                                 self.learner.p_state.apply_fn(p_params, grids[start:end]))
                next_max_norm = jnp.maximum(
                    jnp.max(jnp.sum(jnp.abs(next_det_grids - grids[start:end]), axis=1) * v_batch), next_max_norm)

            noise_norm = jnp.sum(jnp.abs(self.env.noise))
        elif self.norm == "linf":
            delta = 0.5 * jnp.max(
                steps
            )
            lip_f = self.env.lipschitz_constant_linf
            lip_p = jnp.max(self.vassign_local_lip(grids, self.cached_lip_p_linf))
            lip_norm = lip_f * jnp.maximum(lip_p, 1) * delta
            next_max_norm = -jnp.inf
            ub_target = ub_target - lb_domain
            for start in range(0, grids.shape[0], self.batch_size):
                end = min(start + self.batch_size, grids.shape[0])
                v_batch = self.learner.v_state.apply_fn(v_params, grids[start:end]).flatten()
                v_batch = (v_batch - lb_domain) / jnp.maximum(ub_target, 1e-6)
                v_batch = jnp.ceil(jnp.clip(jnp.maximum(1 - v_batch, 0), max=1))
                next_det_grids = self.env.v_next(grids[start:end],
                                                 self.learner.p_state.apply_fn(p_params, grids[start:end]))
                next_max_norm = jnp.maximum(
                    jnp.max(jnp.max(jnp.abs(next_det_grids - grids[start:end]), axis=1) * v_batch), next_max_norm)

            noise_norm = jnp.max(jnp.abs(self.env.noise))
        else:
            raise ValueError("Should not happen")
        big_delta = next_max_norm + lip_norm + noise_norm

        return big_delta

    def get_m_d(self, d):
        """
        Used for stability time contour computation.
        """
        grids, steps = self.get_unsafe_d_grid(d, self.grid_size)

        l_params = self.learner.v_state.params

        if self.norm == "l1":
            delta = 0.5 * np.sum(steps)
            lip_l = lipschitz_l1_jax(l_params)
        elif self.norm == "linf":
            delta = 0.5 * np.max(steps)
            lip_l = np.max(self.assign_local_lip(grids, self.cached_lip_l_linf))
        else:
            raise ValueError("Should not happen")

        n = get_n_for_bound_computation(self.env.observation_dim, self.bound_co_factor)
        _, ub_target = self.compute_bound_target(n)
        lb_domain, _ = self.compute_bound_domain(n)
        ub_target = ub_target - lb_domain
        l_grids = self.learner.v_state.apply_fn(l_params, grids)

        return (np.max(l_grids - lb_domain) + delta * lip_l) / np.maximum(ub_target, 1e-6)