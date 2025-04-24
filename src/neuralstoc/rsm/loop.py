import copy
import os
import sys

import jax.random
import pylab as pl
from gym import spaces
from tqdm import tqdm

from neuralstoc.utils import (
    lipschitz_l1_jax,
    triangular,
    pretty_time,
    pretty_number,
    lipschitz_linf_jax, compute_local_lipschitz,
)

from neuralstoc.rsm.verifier import get_n_for_bound_computation
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp


import logging
logger = logging.getLogger("neuralstoc")


class RSMLoop:
    """
    RSMLoop implements the learner-verifier framework for neural stochastic control with certificates.
    
    The learner-verifier loop is the core component of NeuralStoc, responsible for iteratively:
    1. Training a neural supermartingale certificate (and optionally a neural controller)
    2. Verifying whether the certificate satisfies the conditions required to prove the specification
    3. If verification fails, generating counterexamples to refine the training
    
    The loop continues until either:
    - A valid certificate is found (and controller, if in control synthesis mode)
    - A timeout is reached
    
    Attributes:
        learner: The RSMLearner module for training neural networks
        verifier: The RSMVerifier module for checking certificate conditions
        env: The stochastic environment
        train_p: When to start training the policy (iteration number)
        min_iters: Minimum number of loop iterations to enable refinement of the grids in verifier
        soft_constraint: Whether to terminate even if a lower bound is found
        policy_rollback: Whether to enable policy rollback if training diverges
        exp_name: Name of the experiment for saving artifacts
    """
    
    def __init__(
            self,
            learner,
            verifier,
            env,
            plot,
            soft_constraint=False,
            train_p=0,
            min_iters=0,
            exp_name="test",
            policy_rollback=True,
            rollback_threshold=0.99,
            no_train=False,
            skip_first=False,
    ):
        """
        Initialize the Learner-Verifier loop
        
        Args:
            learner: The RSMLearner module for training neural networks
            verifier: The RSMVerifier module for checking certificate conditions
            env: The stochastic environment
            plot: Whether to generate plots
            soft_constraint: If True, only the expected decrease condition will be checked,
                reach-avoid probability will be optimized in the training and computed but not enforced,
                i.e., the loop will terminate even if the reach-avoid threshold specified in the verifier 
                module is not achieved.
            train_p: Integer indicating after which loop iteration the policy parameters will be optimized.
            min_iters: Integer specifying the minimum number of iterations that the loop should run to enable refinement of the grids in verifier.
            exp_name: Name of the experiment for saving artifacts (default: "")
            policy_rollback: Whether to enable policy rollback if training diverges (default: False)
            rollback_threshold: Threshold for determining policy divergence (default: 0.1)
            no_train: Whether to skip training and only verify (default: False)
            skip_first: Whether to skip the first iteration training (default: False)
        """
        self.env = env
        self.learner = learner
        self.verifier = verifier
        self.train_p = train_p
        self.min_iters = min_iters
        self.soft_constraint = soft_constraint
        self.K_p = None
        self.K_l = None
        self.policy_rollback = policy_rollback
        self.rollback_threshold = rollback_threshold
        self.no_train = no_train
        self.skip_first = skip_first

        os.makedirs(exp_name, exist_ok=True)
        os.makedirs(f"{exp_name}/plots", exist_ok=True)
        os.makedirs(f"{exp_name}/saved", exist_ok=True)
        os.makedirs(f"{exp_name}/loop", exist_ok=True)
        os.makedirs(f"{exp_name}/checkpoints", exist_ok=True)
        self.train_omega = 256
        self.best_prob = 0
        self.plot = plot
        self.prefill_delta = 0
        self.iter = 0
        self.info = {}
        self.exp_name = exp_name

    def learn(self):

        train_ds = self.verifier.train_buffer.as_tfds(batch_size=self.learner.batch_size)
        current_delta = self.prefill_delta
        start_metrics = None
        self.learner.grid_size = self.verifier.grid_size
        num_epochs = (
            50 if self.iter > 0 else 200
        )  # in the first iteration we train a bit longer
        logger.info(f"Learning for {num_epochs} epochs. {train_ds}")
        if self.iter > 2:
            self.learner.init_with_static = False

        start_time = time.time()
        train_p = self.train_p >= 0 and self.iter >= self.train_p and self.learner.task == "control"

        pbar = tqdm(total=num_epochs, unit="epochs")
        for epoch in range(num_epochs):
            # we always train the RSM
            train_v = True
            metrics = self.learner.train_epoch(
                train_ds, current_delta, train_v, train_p, omega=self.train_omega
            )

            if start_metrics is None:
                start_metrics = metrics
            pbar.update(1)
            pbar.set_description_str(
                f"Train [v={train_v}, p={train_p}]: loss={metrics['loss']:0.3g}, dec_loss={metrics['dec_loss']:0.3g}, violations={metrics['train_violations']:0.3g}, kp_param={metrics['kp_param']}"
            )
        pbar.close()
        self.info["ds_size"] = len(self.verifier.train_buffer)

        training_time = pretty_time(time.time() - start_time)

        print(
            f"Trained on {pretty_number(len(self.verifier.train_buffer))} samples, start_loss={start_metrics['loss']:0.3g}, end_loss={metrics['loss']:0.3g}, start_violations={start_metrics['train_violations']:0.3g}, end_violations={metrics['train_violations']:0.3g} in {training_time}"
        )

    def get_lipschitz_k(self):
        """
        Compute the Lipschitz term for the verification.
        
        This function computes either global or local Lipschitz constants for both
        the controller and certificate networks, based on the selected norm (L1 or L-infinity).
        
        Returns:
            ndarray: The Lipschitz term(s) to be used in verification
        """
        if self.verifier.norm == "l1":
            K_p = lipschitz_l1_jax(self.learner.p_state.params, obs_normalization=self.learner.obs_normalization).item()
            K_l = lipschitz_l1_jax(self.learner.v_state.params).item()
            K_f = self.env.lipschitz_constant
            lipschitz_k = K_l * K_f * (1 + K_p) + K_l

        else:
            if self.learner.model == "tmlp":
                self.learner.update_tmodels()

                grid, steps = self.verifier.get_unfiltered_grid_with_step()
                eps = 0.5 * np.sqrt(np.sum(steps ** 2))
                if self.learner.K_p is None:
                    K_p = compute_local_lipschitz(self.learner.p_tnet, grid, eps, out_dim=self.learner.action_dim, obs_normalization=self.learner.obs_normalization)
                else:
                    K_p = self.learner.K_p
                K_l = compute_local_lipschitz(self.learner.v_tnet, grid, eps)
                self.verifier.cached_lip_l_linf = jnp.float32(K_l)
                self.verifier.cached_lip_p_linf = jnp.float32(K_p)
                print('local K_l:', K_l)
                print('local K_p:', K_p)
                K_f = self.env.lipschitz_constant_linf
                lipschitz_k = K_l * K_f * np.maximum(1, K_p) + K_l
                print('lip_k: ', lipschitz_k)

                self.log(eps=eps)
                global_K_l = lipschitz_linf_jax(self.learner.v_state.params).item()
                self.log(K_l=global_K_l)
                global_K_p = lipschitz_linf_jax(self.learner.p_state.params, obs_normalization=self.learner.obs_normalization).item()
                self.log(K_p=global_K_p)
                self.learner.lip_lambda_l = np.max(K_l) / global_K_l
                self.learner.lip_lambda_p = np.max(K_p) / global_K_p
            else:
                K_p = lipschitz_linf_jax(self.learner.p_state.params, obs_normalization=self.learner.obs_normalization).item()
                K_l = lipschitz_linf_jax(self.learner.v_state.params).item()
                K_f = self.env.lipschitz_constant_linf
                lipschitz_k = K_l * K_f * np.maximum(1, K_p) + K_l
                lipschitz_k = float(lipschitz_k)
                self.log(lipschitz_k=lipschitz_k)
                lipschitz_k = np.array([lipschitz_k])
                self.verifier.cached_lip_l_linf = np.array([jnp.float32(K_l)])
                self.verifier.cached_lip_p_linf = np.array([jnp.float32(K_p)])
                self.learner.lip_lambda_l = 1
                self.learner.lip_lambda_p = 1
                self.log(K_p=K_p)
                self.log(K_f=K_f)
                self.log(K_l=K_l)
        if self.verifier.norm != "linf":

            self.log(K_p=K_p)
            self.log(K_f=K_f)
            self.log(K_l=K_l)

            lipschitz_k = float(lipschitz_k)
            self.log(lipschitz_k=lipschitz_k)

        return lipschitz_k

    def check_decrease_condition(self, lipschitz_k):
        """
        Check if the expected decrease condition of the supermartingale certificate is satisfied.
        
        This method verifies whether the certificate function decreases in expectation over
        the entire state space (excluding the target set).
        
        Args:
            lipschitz_k: The Lipschitz term(s)
            
        Returns:
            tuple: (satisfied, max_decrease) where
                - satisfied (bool): True if the decrease condition is satisfied
                - max_decrease (float): The maximum allowed decrease value
        """
        logger.info("Checking decrease condition...")
        (
            violations,
            hard_violations,
            max_decrease,
            max_decay,
            violation_min_val
        ) = self.verifier.check_dec_cond(lipschitz_k)
        self.log(violations=int(violations))
        self.log(hard_violations=int(hard_violations))
        self.log(max_decrease=max_decrease)
        self.log(max_decay=max_decay)
        self.log(violation_min_val=violation_min_val)

        if violations == 0:
            return True, max_decrease
        if (hard_violations == 0 and self.iter > self.min_iters) or self.no_train:
            if self.env.observation_space.shape[0] == 2:
                self.verifier.grid_size *= 1.1
                self.verifier.grid_size = int(self.verifier.grid_size)
                print(f"Increasing grid resolution -> {self.verifier.grid_size}")
            elif self.env.observation_space.shape[0] == 3:
                if self.no_train:
                    self.verifier.grid_size *= 1.05
                    self.verifier.grid_size = int(self.verifier.grid_size)
                    print(f"Increasing grid resolution -> {self.verifier.grid_size}")
            else:
                if self.no_train:
                    self.verifier.grid_size *= 1.01
                    self.verifier.grid_size = int(self.verifier.grid_size)
                    print(f"Increasing grid resolution -> {self.verifier.grid_size}")

        return False, max_decrease

    def verify(self):
        """
        Verify whether the neural supermartingale certificate satisfies all conditions.
        
        This method checks if the certificate satisfies all the required conditions:
        - For reachability: probability of reaching the target
        - For reach-avoid: probability of reaching the target while avoiding unsafe states
        - For safety: probability of always avoiding unsafe states
        - For stability: probability of eventually staying in the target set
        
        Returns:
            float or None: The probability bound if verification succeeds, None otherwise
        """
        if self.iter < self.min_iters:
            self.verifier.refinement_enabled = False
        else:
            self.verifier.refinement_enabled = True
        lipschitz_k = self.get_lipschitz_k()
        dec_sat, max_decrease = self.check_decrease_condition(lipschitz_k)

        self.learner.save(f"{self.exp_name}/saved/{self.env.name}_loop_{self.iter}.jax")
        print("[SAVED]")
        if dec_sat:
            print("Decrease condition fulfilled!")

            n = get_n_for_bound_computation(self.env.observation_dim)

            if self.verifier.spec == "reachability":
                return self.verifier.prob
            elif self.verifier.spec == "reach_avoid":
                _, ub_init = self.verifier.compute_bound_init(n)
                lb_unsafe, _ = self.verifier.compute_bound_unsafe(n)
                lb_domain, _ = self.verifier.compute_bound_domain(n)
                _, ub_target = self.verifier.compute_bound_target(n)
                self.log(ub_init=ub_init)
                self.log(lb_unsafe=lb_unsafe)
                self.log(lb_domain=lb_domain)
                self.log(ub_target=ub_target)
                if lb_unsafe < ub_init:
                    print(
                        "WARNING: RSM is lower at unsafe than in init. No Reach-avoid/Safety guarantees can be obtained."
                    )
                    if float(self.verifier.prob) < 1.0:
                        return None
                # normalize to lb_domain -> 0
                ub_init = ub_init - lb_domain
                lb_unsafe = lb_unsafe - lb_domain
                # normalize to ub_init -> 1
                lb_unsafe = lb_unsafe / ub_init
                actual_prob = 1 - 1 / np.clip(lb_unsafe, 1e-9, None)
                if actual_prob > self.verifier.prob:
                    (violations,
                     hard_violations,
                     max_decrease,
                     max_decay,
                     violation_min_val
                     ) = self.verifier.check_dec_cond(lipschitz_k,
                                                      ra_bounds=(1 / (1 - self.verifier.prob), lb_unsafe))
                    self.log(max_decrease=np.maximum(max_decrease, self.info["max_decrease"]))
                    self.log(max_decay=np.maximum(max_decay, self.info["max_decay"]))
                    if violation_min_val < lb_unsafe:
                        lb_unsafe = violation_min_val
                        actual_prob = 1 - 1 / np.clip(lb_unsafe, 1e-9, None)
                self.log(old_prob=actual_prob)
                self.log(actual_prob=actual_prob)

                num = -2 * (-self.info["max_decrease"]) * (lb_unsafe - 1)
                denom = np.square(self.info["K_l"]) * np.square(self.env.delta)
                other_prob = 1 - np.exp(num / np.clip(denom, 1e-9, None))
                self.log(other_prob=other_prob)
                best_reach_bound = np.maximum(actual_prob, other_prob)

                N = np.floor((lb_unsafe - 1) / (self.info["K_l"] * self.env.delta))
                improved_bound = 1 - 1 / np.clip(lb_unsafe, 1e-9, None) * (
                        self.info["max_decay"] ** N
                )
                self.log(improved_bound=improved_bound)
                best_reach_bound = np.maximum(best_reach_bound, improved_bound)
                self.best_prob = np.maximum(self.best_prob, best_reach_bound)

                with open(f"log_new_bound", "a") as f:
                    f.write(f"\n#### {self.exp_name} ####\n")
                    f.write(f"orig_bound     = {actual_prob}\n")
                    f.write(f"lambda  = {lb_unsafe:0.4g}\n")
                    f.write(f"epsilon = {-self.info['max_decrease']:0.4g}\n")
                    f.write(f"LV      = {self.info['K_l']:0.4g}\n")
                    f.write(f"delta   = {self.env.delta:0.4g}\n")
                    f.write(f"num     = {num:0.4g}\n")
                    f.write(f"denom   = {denom:0.4g}\n")
                    f.write(f"frac    = {num / np.clip(denom, 1e-9, None):0.4g}\n")
                    f.write(f"exp     = {np.exp(num / np.clip(denom, 1e-9, None)):0.4g}\n")
                    f.write(f"bound   = {other_prob:0.4g}\n")
                    f.write(f"------------------------------\n")
                    f.write(f"max_decay      = {self.info['max_decay']:0.4g}\n")
                    f.write(f"N              = {N}\n")
                    f.write(f"improved_bound = {improved_bound}\n")
                    f.write(f"best_prob = {self.best_prob}\n")
                    f.write(f"iter = {self.iter}\n")

                if (
                        self.soft_constraint or actual_prob >= self.verifier.prob
                ) and self.iter >= self.min_iters:
                    return best_reach_bound
                return None
            elif self.verifier.spec == "safety":
                _, ub_init = self.verifier.compute_bound_init(n)
                lb_unsafe, _ = self.verifier.compute_bound_unsafe(n)
                lb_domain, _ = self.verifier.compute_bound_domain(n)
                self.log(ub_init=ub_init)
                self.log(lb_unsafe=lb_unsafe)
                self.log(lb_domain=lb_domain)
                if lb_unsafe < ub_init:
                    print(
                        "WARNING: RSM is lower at unsafe than in init. No Reach-avoid/Safety guarantees can be obtained."
                    )
                    if float(self.verifier.prob) < 1.0:
                        return None
                # normalize to lb_domain -> 0
                ub_init = ub_init - lb_domain
                lb_unsafe = lb_unsafe - lb_domain
                # normalize to ub_init -> 1
                lb_unsafe = lb_unsafe / ub_init
                actual_prob = 1 - 1 / np.clip(lb_unsafe, 1e-9, None)
                if actual_prob > self.verifier.prob:
                    (violations,
                     hard_violations,
                     max_decrease,
                     max_decay,
                     violation_min_val
                     ) = self.verifier.check_dec_cond(lipschitz_k,
                                                      ra_bounds=(1 / (1 - self.verifier.prob), lb_unsafe))
                    self.log(max_decrease=np.maximum(max_decrease, self.info["max_decrease"]))
                    self.log(max_decay=np.maximum(max_decay, self.info["max_decay"]))
                    if violation_min_val < lb_unsafe:
                        lb_unsafe = violation_min_val
                        actual_prob = 1 - 1 / np.clip(lb_unsafe, 1e-9, None)
                self.log(old_prob=actual_prob)
                self.log(actual_prob=actual_prob)

                num = -2 * (-self.info["max_decrease"]) * (lb_unsafe - 1)
                denom = np.square(self.info["K_l"]) * np.square(self.env.delta)
                other_prob = 1 - np.exp(num / np.clip(denom, 1e-9, None))
                self.log(other_prob=other_prob)
                best_reach_bound = np.maximum(actual_prob, other_prob)

                N = np.floor((lb_unsafe - 1) / (self.info["K_l"] * self.env.delta))
                improved_bound = 1 - 1 / np.clip(lb_unsafe, 1e-9, None) * (
                        self.info["max_decay"] ** N
                )
                self.log(improved_bound=improved_bound)
                best_reach_bound = np.maximum(best_reach_bound, improved_bound)
                self.best_prob = np.maximum(self.best_prob, best_reach_bound)

                with open(f"log_new_bound", "a") as f:
                    f.write(f"\n#### {self.exp_name} ####\n")
                    f.write(f"orig_bound     = {actual_prob}\n")
                    f.write(f"lambda  = {lb_unsafe:0.4g}\n")
                    f.write(f"epsilon = {-self.info['max_decrease']:0.4g}\n")
                    f.write(f"LV      = {self.info['K_l']:0.4g}\n")
                    f.write(f"delta   = {self.env.delta:0.4g}\n")
                    f.write(f"num     = {num:0.4g}\n")
                    f.write(f"denom   = {denom:0.4g}\n")
                    f.write(f"frac    = {num / np.clip(denom, 1e-9, None):0.4g}\n")
                    f.write(f"exp     = {np.exp(num / np.clip(denom, 1e-9, None)):0.4g}\n")
                    f.write(f"bound   = {other_prob:0.4g}\n")
                    f.write(f"------------------------------\n")
                    f.write(f"max_decay      = {self.info['max_decay']:0.4g}\n")
                    f.write(f"N              = {N}\n")
                    f.write(f"improved_bound = {improved_bound}\n")
                    f.write(f"best_prob = {self.best_prob}\n")
                    f.write(f"iter = {self.iter}\n")

                if (
                        self.soft_constraint or actual_prob >= self.verifier.prob
                ) and self.iter >= self.min_iters:
                    return best_reach_bound
                return None
            elif self.verifier.spec == "stability":
                lb_unsafe, _ = self.verifier.compute_bound_unsafe(n)
                lb_domain, _ = self.verifier.compute_bound_domain(n)
                _, ub_target = self.verifier.compute_bound_target(n)
                self.log(lb_unsafe=lb_unsafe)
                self.log(lb_domain=lb_domain)
                self.log(ub_target=ub_target)
                if self.verifier.norm == "l1":
                    lip_l = lipschitz_l1_jax(self.learner.v_state.params).item()
                else:
                    lip_l = np.max(self.verifier.cached_lip_l_linf)
                grids, steps = self.verifier.get_unsafe_complement_grid(self.verifier.grid_size)
                big_d = self.verifier.get_big_delta(grids, steps, ub_target, lb_domain)
                self.log(big_d=big_d)

                ub_target = ub_target - lb_domain
                lb_unsafe = lb_unsafe - lb_domain
                lb_unsafe = lb_unsafe / np.maximum(ub_target, 1e-6)
                lip_l = lip_l / np.maximum(ub_target, 1e-6)
                if lb_unsafe <= 1 + lip_l * big_d:
                    print(
                        "{Unsafe states are not greater than the bound. the actual lb: ",
                        lb_unsafe,
                        ", the desired lower bound: ",
                        1 + lip_l * big_d,
                        ", lip_v: ",
                        lip_l,
                        ", big delta: ",
                        big_d,
                        "}"
                    )
                    return None
                p = (1 + lip_l * big_d) / lb_unsafe
                self.log(p=p)
                return 1
        return None

    def log(self, **kwargs):
        """
        Log information about the current iteration.
        
        Args:
            **kwargs: Key-value pairs to add to the info dictionary
        """
        for k, v in kwargs.items():
            self.info[k] = v

    def run(self, timeout):
        """
        Run the learner-verifier loop until a valid certificate is found or timeout is reached.
        
        The loop consists of the following steps:
        1. Learn a neural certificate (and controller if in control mode)
        2. Verify whether the certificate satisfies all required conditions
        3. If verification fails, generate counterexamples and refine training
        
        Args:
            timeout: Maximum time in seconds to run the loop
            
        Returns:
            float or None: The probability bound if verification succeeds, None otherwise
        """
        start_time = time.time()
        self.prefill_delta = self.verifier.prefill_train_buffer()
        while True:
            runtime = time.time() - start_time
            self.log(runtime=runtime)
            self.log(iter=self.iter)

            if runtime > timeout:
                print("Timeout!")
                return False
            print(
                f"\n#### Iteration {self.iter} (runtime: {pretty_time(runtime)}) #####"
            )
            if not self.no_train and (not self.skip_first or (self.skip_first and self.iter > 0)):
                p_state_copy = copy.deepcopy(self.learner.p_state)
                self.learn()
                if self.plot:
                    logger.info("Plotting")
                    self.plot_l(f"{self.exp_name}/loop/{self.env.name}_{self.iter:04d}_{self.exp_name}_pre.png", is_pre=True)
                _, res = self.learner.evaluate_rl()
                if res['num_end_in_target'] / res['num_traj'] < self.rollback_threshold and self.policy_rollback:
                    self.learner.p_state = p_state_copy

            actual_prob = self.verify()

            self.log(runtime=time.time() - start_time)
            print("Log=", str(self.info))
            sys.stdout.flush()

            if self.plot:
                logger.info("Plotting")
                self.plot_l(f"{self.exp_name}/loop/{self.env.name}_{self.iter:04d}_{self.exp_name}.png")

            if actual_prob is not None:
                if self.verifier.spec == "reach_avoid":
                    print(
                        f"Probability of reaching the target safely is at least {actual_prob * 100:0.3f}%"
                    )
                elif self.verifier.spec == "stability":
                    print(
                        f"Stability bound is {actual_prob * 100:0.3f}%"
                    )
                elif self.verifier.spec == "safety":
                    print(
                        f"Safety bound is at least {actual_prob * 100:0.3f}%"
                    )
                elif self.verifier.spec == "reachability":
                    print(
                        f"Reachability bound is at least {actual_prob * 100:0.3f}%"
                    )
                if self.verifier.spec == 'stability' and self.plot:
                    try:
                        self.plot_stability_time_contour(self.info['p'],
                                                         -self.info['max_decrease'],
                                                         self.info['ub_target'] - self.info['lb_domain'],
                                                         self.info['big_d'],
                                                         self.info['lb_domain'],
                                                         f"{self.exp_name}/plots/{self.env.name}_contour_lines.pdf")
                    except Exception as e:
                        print(f"Error plotting stability time contour: Computation difficulty")

                return True

            self.iter += 1

    def rollout(self, seed=None):
        """
        Generate trajectories with the learned controller.
        
        Args:
            seed: Random seed
            
        Returns:
            traces: ndarray of system trajectories
        """
        if seed is None:
            seed = np.random.default_rng().integers(0, 10000)
        rng = jax.random.PRNGKey(seed)
        space = spaces.Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            dtype=np.float32,
        )

        obs = space.sample()
        trace = [np.array(obs)]
        for i in range(self.env.episode_length):
            action = self.learner.p_state.apply_fn(self.learner.p_state.params, obs)
            next_state = self.env.next(obs, action)
            rng, rng0 = jax.random.split(rng)
            noise = triangular(rng0, (self.env.observation_dim,))
            noise = noise * self.env.noise
            obs = next_state + noise
            trace.append(np.array(obs))
        return np.stack(trace, axis=0)

    def plot_l(self, filename, is_pre=False):
        """
        Plot the neural supermartingale certificate function over the state space.
        
        Creates a visualization of the certificate function, highlighting regions 
        like target sets, unsafe sets, and initial states. For 2D environments, 
        it plots a heatmap of the certificate values. For higher dimensions, 
        it projects to the specified plot dimensions.
        
        Args:
            filename: Path to save the plot
            is_pre: Whether this is a pre-verification plot (default: False)
        """
        i_, j_ = self.env.plot_dims
        grid, _, _ = self.verifier.get_unfiltered_grid(n=50)
        for target_ind, source_ind in self.env.plot_dim_map.items():
            grid[:, target_ind] = grid[:, source_ind]
        l = self.learner.v_state.apply_fn(self.learner.v_state.params, grid).flatten()
        l = np.array(l)
        sns.set()
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(grid[:, i_], grid[:, j_], marker="s", c=l, zorder=1, alpha=0.7)
        fig.colorbar(sc)
        ax.set_title(f"L at iter {self.iter} for {self.env.name}")

        terminals_x, terminals_y = [], []
        for i in range(30):
            trace = self.rollout(seed=i)
            ax.plot(
                trace[:, i_],
                trace[:, j_],
                color=sns.color_palette()[0],
                zorder=2,
                alpha=0.3,
            )
            ax.scatter(
                trace[:, i_],
                trace[:, j_],
                color=sns.color_palette()[0],
                zorder=2,
                marker=".",
            )
            terminals_x.append(float(trace[-1, i_]))
            terminals_y.append(float(trace[-1, j_]))
        ax.scatter(terminals_x, terminals_y, color="white", marker="x", zorder=5)
        if not is_pre and self.verifier.hard_constraint_violation_buffer is not None:
            ax.scatter(
                self.verifier.hard_constraint_violation_buffer[:, i_],
                self.verifier.hard_constraint_violation_buffer[:, j_],
                color="green",
                marker="s",
                alpha=0.7,
                zorder=6,
            )
        if self.verifier._debug_violations is not None:
            ax.scatter(
                self.verifier._debug_violations[:, i_],
                self.verifier._debug_violations[:, j_],
                color="cyan",
                marker="s",
                alpha=0.7,
                zorder=6,
            )
        for init in self.env.init_spaces:
            x = [
                init.low[i_],
                init.high[i_],
                init.high[i_],
                init.low[i_],
                init.low[i_],
            ]
            y = [
                init.low[j_],
                init.low[j_],
                init.high[j_],
                init.high[j_],
                init.low[j_],
            ]
            ax.plot(x, y, color="cyan", alpha=0.5, zorder=7)
        for unsafe in self.env.unsafe_spaces:
            x = [
                unsafe.low[i_],
                unsafe.high[i_],
                unsafe.high[i_],
                unsafe.low[i_],
                unsafe.low[i_],
            ]
            y = [
                unsafe.low[j_],
                unsafe.low[j_],
                unsafe.high[j_],
                unsafe.high[j_],
                unsafe.low[j_],
            ]
            ax.plot(x, y, color="magenta", alpha=0.5, zorder=7)
        for target_space in self.env.target_spaces:
            x = [
                target_space.low[i_],
                target_space.high[i_],
                target_space.high[i_],
                target_space.low[i_],
                target_space.low[i_],
            ]
            y = [
                target_space.low[j_],
                target_space.low[j_],
                target_space.high[j_],
                target_space.high[j_],
                target_space.low[j_],
            ]
            ax.plot(x, y, color="green", alpha=0.5, zorder=7)
        ax.set_xlim(
            [self.env.observation_space.low[i_], self.env.observation_space.high[i_]]
        )
        ax.set_ylim(
            [self.env.observation_space.low[j_], self.env.observation_space.high[j_]]
        )
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)

    def plot_stability_time_contour(self, p, eps, ub_target, big_d, lb_domain, filename):
        """
        Plot contours showing the expected time to stability for different regions.
        
        This function creates a visualization for stability specifications, showing
        expected time to reach and remain in the target set.
        
        Args:
            p: The probability of leaving the target set
            eps: The maximum decrease value
            ub_target: Upper bound on certificate values in the target set
            big_d: The "big Delta" parameter for stability analysis
            lb_domain: Lower bound on certificate values in the domain
            filename: Path to save the plot
        """
        if self.env.observation_dim > 2:
            return
        m_d = self.verifier.get_m_d(big_d)
        n = 100

        states, _, _ = self.verifier.get_unfiltered_grid(n=n)
        l = self.learner.v_state.apply_fn(self.learner.v_state.params, states).flatten()
        stab_exp = np.array(((l - lb_domain) / ub_target + (p / (1 - p)) * m_d) / eps)

        plt.figure(figsize=(6, 6))

        contours = plt.contour(np.reshape(states[:, 0], (n, n)), np.reshape(states[:, 1], (n, n)), np.reshape(stab_exp, (n, n)))
        plt.clabel(contours, inline=1, fontsize=12)

        plt.xlabel('x1')
        plt.ylabel('x2')

        plt.savefig(filename)
        pl.close()
