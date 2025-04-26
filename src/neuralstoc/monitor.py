from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys


class ExperimentMonitor:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.experiment_path = Path('outputs') / experiment_name
        self.experiment_path.mkdir(parents=True, exist_ok=True)

        self.info = {}

    def log(self, **kwargs):
        for key, value in kwargs.items():
            self.info[key] = value
    
    def write_results(self, args, returned_txt, sat):
        os.makedirs("study_results", exist_ok=True)
        env_name = "_".join(args.env.split("_")[:2])
        cmd_line = " ".join(sys.argv)
        with open(f"study_results/info_{env_name}.log", "a") as f:
            f.write(f"python3 {cmd_line}\n")
            f.write("    args=" + str(vars(args)) + "\n")
            f.write("    return =" + returned_txt + "\n")
            f.write("    info=" + str(self.info) + "\n")
            f.write("    sat=" + str(sat) + "\n")
            f.write("\n\n")
        with open(f"global_summary.txt", "a") as f:
            f.write(f"{cmd_line}\n")
            f.write("    args=" + str(vars(args)) + "\n")
            f.write("    return =" + returned_txt + "\n")
            f.write("    info=" + str(self.info) + "\n")
            f.write("    sat=" + str(sat) + "\n")
            f.write("\n\n")
    
    def file_path(self, filename):
        """
            Ensure the parent directory exists and return the full path to the file.
        """
        path = self.experiment_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def plot_l(self, env, verifier, learner, filename, is_pre=False):
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
        i_, j_ = env.plot_dims
        grid, _, _ = verifier.get_unfiltered_grid(n=50)
        for target_ind, source_ind in env.plot_dim_map.items():
            grid[:, target_ind] = grid[:, source_ind]
        l = learner.v_state.apply_fn(learner.v_state.params, grid).flatten()
        l = np.array(l)
        sns.set()
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(grid[:, i_], grid[:, j_], marker="s", c=l, zorder=1, alpha=0.7)
        fig.colorbar(sc)
        if 'iter' in self.info:
            ax.set_title(f"L at iter {self.info['iter']} for {env.name}")
        else:
            ax.set_title(f"L for {env.name}")

        # terminals_x, terminals_y = [], []
        # for i in range(30):
        #     trace = self.rollout(seed=i)
        #     ax.plot(
        #         trace[:, i_],
        #         trace[:, j_],
        #         color=sns.color_palette()[0],
        #         zorder=2,
        #         alpha=0.3,
        #     )
        #     ax.scatter(
        #         trace[:, i_],
        #         trace[:, j_],
        #         color=sns.color_palette()[0],
        #         zorder=2,
        #         marker=".",
        #     )
        #     terminals_x.append(float(trace[-1, i_]))
        #     terminals_y.append(float(trace[-1, j_]))
        # ax.scatter(terminals_x, terminals_y, color="white", marker="x", zorder=5)
        if not is_pre and verifier.hard_constraint_violation_buffer is not None:
            ax.scatter(
                verifier.hard_constraint_violation_buffer[:, i_],
                verifier.hard_constraint_violation_buffer[:, j_],
                color="green",
                marker="s",
                alpha=0.7,
                zorder=6,
            )
        if verifier._debug_violations is not None:
            ax.scatter(
                verifier._debug_violations[:, i_],
                verifier._debug_violations[:, j_],
                color="cyan",
                marker="s",
                alpha=0.7,
                zorder=6,
            )
        for init in env.init_spaces:
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
        for unsafe in env.unsafe_spaces:
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
        for target_space in env.target_spaces:
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
            [env.observation_space.low[i_], env.observation_space.high[i_]]
        )
        ax.set_ylim(
            [env.observation_space.low[j_], env.observation_space.high[j_]]
        )
        fig.tight_layout()
        fig.savefig(self.file_path(filename))
        plt.close(fig)


    def plot_stability_time_contour(self, env, verifier, learner, filename):
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
        p = self.info['p'],
        eps = -self.info['max_decrease'],
        ub_target = self.info['ub_target'] - self.info['lb_domain'],
        big_d = self.info['big_d'],
        lb_domain = self.info['lb_domain'],

        if env.observation_dim > 2:
            return
        m_d = verifier.get_m_d(big_d)
        n = 100

        states, _, _ = verifier.get_unfiltered_grid(n=n)
        l = learner.v_state.apply_fn(learner.v_state.params, states).flatten()
        stab_exp = np.array(((l - lb_domain) / ub_target + (p / (1 - p)) * m_d) / eps)

        plt.figure(figsize=(6, 6))

        contours = plt.contour(np.reshape(states[:, 0], (n, n)), np.reshape(states[:, 1], (n, n)), np.reshape(stab_exp, (n, n)))
        plt.clabel(contours, inline=1, fontsize=12)

        plt.xlabel('x1')
        plt.ylabel('x2')

        plt.savefig(self.file_path(filename))
        plt.close()
