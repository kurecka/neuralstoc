import copy
import os
import sys

import jax.random
import pylab as pl
from gym import spaces
from tqdm import tqdm

from neuralstoc.utils import (
    triangular,
    pretty_time,
    pretty_number,
)

LEGACY = False

from neuralstoc.rsm.lipschitz import lipschitz_l1_jax, get_lipschitz_k

from neuralstoc.rsm.verifier import get_n_for_bound_computation
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp


import logging
logger = logging.getLogger("neuralstoc")

class RSMTransform:
    """
    RSMTransform is a utility class for transforming supermartingale certificate into a normalized form.
    
    The transformation is defined in the affine form:
    f(x) = (x - low) / (high - low)

    For safety, reach-avoid, reachability:
    - `low` is the lower bound of the domain, and
    - `high` is the upper bound of the initial state.

    For stability:
    - `low` is the lower bound of the domain, and
    - `high` is the upper bound of the target set.
    """
    
    def __init__(self, domain_lb, init_ub, target_ub, spec):
        high = init_ub if spec in ["safety", "reach_avoid", "reachability"] else target_ub
        low = domain_lb
        self.translation = - low
        self.scale = 1 / jnp.maximum(high - low, 1e-6)  # Avoid division by zero
    
    def __call__(self, x):
        return (x + self.translation) * self.scale

    def apply_inverse(self, x):
        return x / self.scale - self.translation
    
    def apply_scale(self, x):
        return x * self.scale
    
    def apply_translation(self, x):
        return x + self.translation


def get_rsm_normalization_coeffs(domain_lb, init_ub, target_ub, spec):
    low = domain_lb
    if spec == "safety" or spec == "reach_avoid" or spec == "reachability":
        high = init_ub
    else:
        high = target_ub
    shift = -low
    scale = 1 / jnp.maximum(high - low, 1e-6)
    return shift, scale


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
            descent_verifier,
            env,
            monitor,
            plot,
            soft_constraint=False,
            train_p=0,
            min_iters=0,
            exp_name="test",
            policy_rollback=True,
            rollback_threshold=0.99,
            no_train=False,
            skip_first=False,
            epochs=50,
            initial_epochs=200,
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
        self.descent_verifier = descent_verifier
        self.monitor = monitor
        self.train_p = train_p
        self.min_iters = min_iters
        self.soft_constraint = soft_constraint
        self.K_p = None
        self.K_l = None
        self.policy_rollback = policy_rollback
        self.rollback_threshold = rollback_threshold
        self.no_train = no_train
        self.skip_first = skip_first
        self.epochs = epochs
        self.initial_epochs = initial_epochs

        # TODO: Add to monitor
        os.makedirs(exp_name, exist_ok=True)
        self.train_omega = 256
        self.best_prob = 0
        self.plot = plot
        self.prefill_delta = 0
        self.exp_name = exp_name
        self.iter = 0
        self.log(iter=self.iter)
        self.max_prob = -1

    def learn(self):
        train_ds = self.verifier.train_buffer.as_tfds(batch_size=self.learner.batch_size)
        current_delta = self.prefill_delta
        start_metrics = None
        self.learner.grid_size = self.verifier.grid_size
        num_epochs = (
            self.epochs if self.iter > 0 else self.initial_epochs
        )  # in the first iteration we train a bit longer

        if num_epochs == 0:
            logger.info("Skipping training")
            return

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
        self.log(ds_size=len(self.verifier.train_buffer))

        training_time = pretty_time(time.time() - start_time)

        logger.info(
            f"Trained on {pretty_number(len(self.verifier.train_buffer))} samples, start_loss={start_metrics['loss']:0.3g}, end_loss={metrics['loss']:0.3g}, start_violations={start_metrics['train_violations']:0.3g}, end_violations={metrics['train_violations']:0.3g} in {training_time}"
        )

    def check_decrease_condition(self, value_bounds=None, lipschitz_k=None):
        """
        Check if the expected decrease condition of the supermartingale certificate is satisfied.
        
        This method verifies whether the certificate function decreases in expectation over
        the entire state space (excluding the target set).
        
        Args:
            local_lipschitz_k: The Lipschitz term(s)
            
        Returns:
            tuple: (satisfied, max_decrease) where
                - satisfied (bool): True if the decrease condition is satisfied
                - max_decrease (float): The maximum allowed decrease value
                - max_decay (float): The maximum decay value
                - violation_min_val (float): The minimum value of the violation
        """
        logger.info("Checking decrease condition...")
        if LEGACY:
            assert lipschitz_k is not None, "Lipschitz constant must be provided for legacy verifier"
            (
                violations,
                hard_violations,
                max_decrease,
                max_decay,
                violation_min_val
            ) = self.verifier.check_dec_cond(lipschitz_k, ra_bounds=value_bounds)

        else:
            (
                violations,
                hard_violations,
                max_decrease,
                max_decay,
                violation_min_val,
                counterexamples,
            ) = self.descent_verifier.check_dec_cond(
                self.learner.v_state.params,
                self.learner.p_state.params,
                value_bounds=value_bounds,
            )
        
            self.verifier.train_buffer.append(counterexamples)

        self.log(
            violations=int(violations),
            hard_violations=int(hard_violations),
            max_decrease=max_decrease,
            max_decay=max_decay,
            violation_min_val=violation_min_val,
        )
        return violations == 0, max_decrease, max_decay, violation_min_val

    def verify(self):
        if LEGACY:
            return self.verify_legacy()
        else:
            return self.verify_new()

    def verify_new(self):
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
        n = get_n_for_bound_computation(self.env.observation_dim, self.verifier.bound_co_factor)
        _, ub_init = self.verifier.compute_bound_init(n)
        lb_domain, _ = self.verifier.compute_bound_domain(n)
        _, ub_target = self.verifier.compute_bound_target(n)
        self.log(ub_init=ub_init)
        self.log(lb_domain=lb_domain)
        self.log(ub_target=ub_target)

        transform = RSMTransform(
            domain_lb=lb_domain,
            init_ub=ub_init,
            target_ub=ub_target,
            spec=self.verifier.spec,
        )

        dec_sat, max_decrease, _, _ = self.check_decrease_condition(value_bounds=(-np.inf, transform.apply_inverse(1 / (1 - self.verifier.prob))))

        self.learner.save(self.monitor.file_path(f"saved/{self.env.name}_loop_{self.iter}.jax"))
        print("[SAVED]")

        if dec_sat:
            print("Decrease condition fulfilled!")


            if self.verifier.spec == "reachability":
                self.max_prob = np.maximum(self.max_prob, self.verifier.prob)
                self.log(max_actual_prob=self.max_prob)
                return self.verifier.prob
            elif self.verifier.spec == "reach_avoid":
                lb_unsafe, _ = self.verifier.compute_bound_unsafe(n)
                self.log(lb_unsafe=lb_unsafe)

                if lb_unsafe < ub_init:
                    logger.warning(
                        "WARNING: RSM is lower at unsafe than in init. No Reach-avoid/Safety guarantees can be obtained."
                    )
                    if float(self.verifier.prob) < 1.0:
                        return None

                normalized_lb_unsafe = transform(lb_unsafe)
                actual_prob = 1 - 1 / np.clip(normalized_lb_unsafe, 1e-9, None)
                if actual_prob > self.verifier.prob:
                    (
                        violations,
                        max_decrease,
                        max_decay,
                        violation_min_val,
                    ) = self.check_decrease_condition(
                        value_bounds=(transform.apply_inverse(1 / (1 - self.verifier.prob)), normalized_lb_unsafe)
                    )

                    self.log(max_decrease=np.maximum(transform(max_decrease), self.info["max_decrease"]))
                    self.log(max_decay=np.maximum(max_decay, self.info["max_decay"]))  # TODO: This is just not correct due to the affine normalization. The offset changes the relative descent.
                    if violation_min_val < lb_unsafe:
                        lb_unsafe = violation_min_val
                        normalized_lb_unsafe = transform(lb_unsafe)
                        actual_prob = 1 - 1 / np.clip(normalized_lb_unsafe, 1e-9, None)
                self.log(old_prob=actual_prob)
                self.log(actual_prob=actual_prob)
                self.max_prob = np.maximum(self.max_prob, actual_prob)
                self.log(max_actual_prob=self.max_prob)
                num = -2 * (-self.info["max_decrease"]) * (lb_unsafe - 1)
                denom = np.square(self.info["K_l"]) * np.square(self.env.delta)
                other_prob = 1 - np.exp(num / np.clip(denom, 1e-9, None))
                self.log(other_prob=other_prob)
                best_reach_bound = np.maximum(actual_prob, other_prob)

                # TODO: Fix together with decay
                # N = np.floor((lb_unsafe - 1) / (self.info["K_l"] * self.env.delta))
                # improved_bound = 1 - 1 / np.clip(lb_unsafe, 1e-9, None) * (
                #         self.info["max_decay"] ** N
                # )
                # self.log(improved_bound=improved_bound)
                # best_reach_bound = np.maximum(best_reach_bound, improved_bound)
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
                    # f.write(f"max_decay      = {self.info['max_decay']:0.4g}\n")
                    # f.write(f"N              = {N}\n")
                    # f.write(f"improved_bound = {improved_bound}\n")
                    f.write(f"best_prob = {self.best_prob}\n")
                    f.write(f"iter = {self.iter}\n")

                if (
                        self.soft_constraint or self.best_prob >= self.verifier.prob
                ) and self.iter >= self.min_iters:
                    return best_reach_bound
                if self.no_train:
                    return best_reach_bound
                return None
            else:
                raise ValueError(f"Specification {self.verifier.spec} is not supported in the new verifier yet.")
        return None

    def verify_legacy(self):
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
        lipschitz_k = get_lipschitz_k(self.verifier, self.learner, log=self.monitor.log)
        dec_sat, max_decrease, _, _ = self.check_decrease_condition(lipschitz_k=lipschitz_k)

        self.learner.save(self.monitor.file_path(f"saved/{self.env.name}_loop_{self.iter}.jax"))
        print("[SAVED]")
        if dec_sat:
            print("Decrease condition fulfilled!")

            n = get_n_for_bound_computation(self.env.observation_dim, self.verifier.bound_co_factor)

            if self.verifier.spec == "reachability":
                self.max_prob = np.maximum(self.max_prob, self.verifier.prob)
                self.log(max_actual_prob=self.max_prob)
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
                self.max_prob = np.maximum(self.max_prob, actual_prob)
                self.log(max_actual_prob=self.max_prob)
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
                if self.no_train:
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
                self.max_prob = np.maximum(self.max_prob, actual_prob)
                self.log(max_actual_prob=self.max_prob)

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
                if self.no_train:
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
                self.max_prob = 1
                self.log(max_actual_prob=self.max_prob)
                return p
        return None

    def log(self, **kwargs):
        self.monitor.log(**kwargs)
    
    @property
    def info(self):
        return self.monitor.info

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
            self.log(
                runtime=runtime,
                iter=self.iter,
            )

            if runtime > timeout:
                logger.warning("Timeout!")
                return False
            logger.info(f"#### Iteration {self.iter} (runtime: {pretty_time(runtime)}) #####")
            try:
                if not self.no_train and (not self.skip_first or (self.skip_first and self.iter > 0)):
                    p_state_copy = copy.deepcopy(self.learner.p_state)
                    self.learn()
                    if self.plot:
                        logger.info("Plotting")
                        self.monitor.plot_l(
                            self.env,
                            self.verifier,
                            self.learner,
                            f"loop/{self.env.name}_{self.iter:04d}_{self.exp_name}_pre.png",
                        )
                    _, res = self.learner.evaluate_rl()
                    if res['num_end_in_target'] / res['num_traj'] < self.rollback_threshold and self.policy_rollback:
                        self.learner.p_state = p_state_copy

                actual_prob = self.verify()

                self.log(runtime=time.time() - start_time)
                print("Log=", str(self.info))
                sys.stdout.flush()

                if self.plot:
                    logger.info("Plotting")
                    self.monitor.plot_l(
                        self.env,
                        self.verifier,
                        self.learner,
                        f"loop/{self.env.name}_{self.iter:04d}_{self.exp_name}.png",
                    )
            except Exception as e:
                print(f"Error in loop: {e}")
                print("If you are facing memory issues, try adjusting the batch sizes, grid size, local Lipschitz computation factor, and/or putting a smaller timeout to prevent extensive grid refinement.")
                return False

            if actual_prob is not None:
                if self.verifier.spec == "reach_avoid":
                    logger.info(
                        f"Probability of reaching the target safely is at least {actual_prob * 100:0.3f}%"
                    )
                elif self.verifier.spec == "stability":
                    logger.info(
                        f"Stability bound is {actual_prob * 100:0.3f}%"
                    )
                elif self.verifier.spec == "safety":
                    logger.info(
                        f"Safety bound is at least {actual_prob * 100:0.3f}%"
                    )
                elif self.verifier.spec == "reachability":
                    logger.info(
                        f"Reachability bound is at least {actual_prob * 100:0.3f}%"
                    )
                if self.verifier.spec == 'stability' and self.plot:
                    try:
                        self.plot_stability_time_contour(
                            self.env,
                            self.verifier,
                            self.learner,
                            f"plots/{self.env.name}_contour_lines.pdf"
                        )
                    except Exception as e:
                        logger.error(f"Error plotting stability time contour: Computation difficulty")

                return True

            self.iter += 1
            self.log(iter=self.iter)

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
