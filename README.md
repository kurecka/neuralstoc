# NeuralStoc: A Tool for Neural Stochastic Control and Verification

This repository contains the supplementary code for the paper "NeuralStoc: A Tool for Neural Stochastic Control and Verification" (CAV 2025), by Matin Ansaripour, Krishnendu Chatterjee, Thomas A. Henzinger, Mathias Lechner, Abhinav Verma, and Đorđe Žikelić.

TODO[![DOI]()]()

## Introduction

NeuralStoc is a tool for neural controller synthesis and verification in discrete-time stochastic dynamical systems. The tool implements and builds upon the first learner-verifier framework for neural stochastic control with certificates, by jointly learning and/or formally verifying a neural controller together with a neural supermartingale certificate of its correctness. NeuralStoc provides a unified interface for analyses with respect to reachability, safety, reach-avoidance, and stability specifications.


The key features of NeuralStoc include:

- **Optimizations**: Introduces several optimizations including modified reach-avoid supermartingales (RASMs), local Lipschitz analysis, and controller rollback, leading to significant improvements in practical performance and scalability.
- **Scalability**: The first tool able to solve neural stochastic control and verification tasks in 4-dimensional environments (4D state space + 2D control input space).

## System Requirements

- Python 3.10 or higher
- CUDA 12 (for GPU acceleration — ensure a compatible cuDNN installation is correctly configured)
- NVIDIA GPU with CUDA support and sufficient VRAM for your intended use
- dequate RAM capacity for your intended use
- CPU with sufficient cores and frequency suitable for your intended use


## Installation

### Option 1: Using Docker (Recommended)

The Docker installation method is recommended as it ensures all dependencies are correctly configured. Please install the docker first.

On ?link to Zenodo?, we provided two pre-build docker images with the names `neuralstoc-artifact.tar.zip` (built with `Dockerfile`) and `neuralstoc-sim-artifact.tar.zip` (built with `Dockerfile-sim`). Please see [Replication of the results in the paper](#replication-of-the-results-in-the-paper) to understand the difference between these two images. To run the image, you can use the following instruction:

1. Unzip the downloaded file:
```bash
unzip neuralstoc-artifact.tar.zip
```

2. Build the Docker image:
```bash
docker load < neuralstoc-artifact.tar
```

3. Run the container with GPU support:
```bash
docker run --gpus all -it neuralstoc
```

If you face any issues running the image or want to build the image from scratch, you can follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/matinansaripour/neuralstoc.git
cd neuralstoc
```

2. Build the Docker image:
```bash
docker build -t neuralstoc .
```

3. Run the container with GPU support:
```bash
docker run --gpus all -it neuralstoc
```


### Option 2: Direct Installation

1. Clone the repository:
```bash
git clone https://github.com/matinansaripour/neuralstoc.git
cd neuralstoc
```

2. Create and activate a virtual environment:
```bash
python3.10 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -U pip setuptools wheel
pip install pyyaml
pip install -e .
```

**Note:** The project has specific dependencies including JAX, Flax, Brax, and other machine learning libraries. Make sure you have CUDA 12 installed for GPU acceleration. These dependencies are listed in the `pyproject.toml` file.

## Tool Overview

NeuralStoc addresses two key problems:

1. **Verification Problem**: Given a neural controller π, prove that the probability of satisfying a specification φ is at least p for all initial states.
2. **Control Problem**: Learn a neural controller π such that the probability of satisfying a specification φ is at least p for all initial states.

The tool supports four types of specifications:
- **Reachability**: Reaching a target
- **Safety**: Avoiding unsafe states
- **Reach-Avoid**: Reaching a target set while avoiding unsafe states
- **Stability**: Converging to and remaining in a target set with probability 1

## Project Structure

```
neuralstoc/
├── scripts/                   # Main entry points and example commands
│   ├── rsm_loop.py            # Main script for running experiments
│   ├── commands.sh            # Example command configurations to replicate the paper results
│   └── config.yaml            # Default configuration file template
└── src/neuralstoc/            # Core library
    ├── environments/          # Environment implementations
    │   ├── vrl_environments.py # environments
    │   └── brl_environments.py # Brax environments for SAC RL training
    ├── rsm/                   # Ranking Supermartingale components
    │   ├── learner.py         # Implementation of the learner module
    │   ├── verifier.py        # Implementation of the verifier module
    │   ├── loop.py            # Learner-verifier loop implementation
    │   └── ibp.py             # Interval Bound Propagation utilities
    ├── rl/                    # Reinforcement learning components
    │   ├── ppo.py             # PPO implementation
    │   └── sac.py             # SAC implementation
    └── utils.py               # Utility functions            
```

## Usage

The main entry point is the `rsm_loop.py` script in the `scripts` directory:

```bash
python3 scripts/rsm_loop.py --env [environment] --task [task_type] \
                           --spec [specification] --prob [probability] \
                           --initialize [RL_algorithm] [[additional options]]
```

The detailed description of available options alongside the example commands are available in [Command Line Arguments](#command-line-arguments).

Each environment is defined as a `Gym` environment with:
- Rectangular state space
- Initial set of states
- Target set (for reachability, reach-avoidance, and stability)
- Unsafe set (for safety and reach-avoidance)
- Control action space

To use SAC initialization, you should also create a Brax environment. Please see `src/neuralstoc/environments/` to see the predefined environments and use them as examples. As an example, you can see how to input the environments to NeuralStoc in `scripts/rsm_loop.py`.

### Using Configuration Files

NeuralStoc supports YAML configuration files to manage experiment parameters. This approach makes it easier to manage and share experiment configurations.

To use a configuration file:

```bash
python3 scripts/rsm_loop.py --load_config --config_path scripts/config.yaml --env v2cavoid --model tmlp --spec reach_avoid --exp_name cavoid_new
```

A template configuration file is available at `scripts/config.yaml`.

### Configuration File Format

Configuration files use YAML format. Here's a simplified example:

```yaml
# Experiment Settings
prob: 0.97
plot: true

# Training Parameters
v_lr: 0.0001
p_lr: 0.000001
lip_lambda: 0.2

# Other parameters...
```


## Command Line Arguments

### Configuration File Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--load_config` | - | Flag to load configuration from a file |
| `--config_path` | `None` | Path to the configuration file |

### General Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--exp_name` | `"rasm_lds"` | Name of the experiment for saving artifacts |
| `--model` | `"tmlp"` | Network architecture type ('mlp' or 'tmlp' for local Lipschitz calculations) |
| `--env` | `"vlds"` | Environment to use (e.g., 'vlds', 'vpend', 'vhuman2') |
| `--task` | `"control"` | Task type ('control' or 'verification') |
| `--spec` | `"reach_avoid"` | Specification type ('reach_avoid', 'safety', 'reachability', or 'stability') |
| `--timeout` | `24 * 60` | Maximum runtime in minutes |
| `--plot` | - | Flag to enable plotting during the run |
| `--env_dim` | `None` | Environment dimension (optional - for LDS 4D) |

### Available Environments

| Environment Argument (`--env`) | Description |
|----------|-------------|
| `v2cavoid` | 2D Collision Avoidance Environment (RASM paper) |
| `vcavoid` | Collision Avoidance Environment (harder) |
| `v2ldss` | Linear Dynamical System with Specified Dimensions (for 4D experiment - extension on RASM paper) |
| `vldss` | Linear Dynamical System with Specified Dimensions (for 4D experiment - extension on sRSM paper) |
| `v2lds` | 2D Linear Dynamical System Environment (RASM paper) |
| `vlds` | 2D Linear Dynamical System Environment (sRSM paper) |
| `v2pend` | 2D Inverted Pendulum Environment (RASM paper) |
| `vpend` | 2D Inverted Pendulum Environment (sRSM paper) |
| `vhuman2` | 2-link Humanoid Balance Environment |
| `vtri` | Triple Integrator Environment |

### Probability and Verification Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--prob` | `0.3` | Target probability threshold for the specification |
| `--eps` | `0.001` | Desired expected decrease value for the certificate |
| `--norm` | `"linf"` | Norm for Lipschitz calculations ('l1' or 'linf') |
| `--min_iters` | `0` | Minimum number of iterations to enable refinement of verification grids |
| `--soft_constraint` | - | If set, only the expected decrease condition will be checked, and the loop will terminate even if the reach-avoid threshold is not achieved |
| `--grid_size` | `"16M"` | Target size of the verification grid (supports k/M/G suffixes) |

### Network and Learning Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden_v` | `128` | Hidden layer size for the certificate network |
| `--hidden_p` | `128` | Hidden layer size for the controller network |
| `--num_layers_v` | `2` | Number of hidden layers in the certificate network |
| `--num_layers_p` | `2` | Number of hidden layers in the controller network |
| `--v_act` | `"relu"` | Activation function for the certificate network |
| `--v_lr` | `0.0005` | Learning rate for the certificate network |
| `--p_lr` | `0.00005` | Learning rate for the controller network |
| `--c_lr` | `0.0005` | Learning rate for the critic network (used in PPO pre-training) |
| `--lip_lambda` | `0.001` | Regularization factor for Lipschitz loss (old loss) or estimation of local vs. global Lipschitz constant ratio (improved loss) |
| `--p_lip` | `3.0` | Maximum allowed Lipschitz constant for the controller |
| `--v_lip` | `10.0` | Maximum allowed Lipschitz constant for the certificate |
| `--train_p` | `10` | After which iteration to start training the policy parameters |
| `--n_local` | `10` | Grid size for computing local Lipschitz constants |
| `--improved_loss` | - | Flag to use the optimized loss function |
| `--estimate_expected_via_ibp` | - | Flag to use interval bound propagation for expected value estimation |

### Initialization Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--initialize` | `"ppo"` | Method to initialize the controller ('ppo' or 'sac'). If 'sac' used, 'load_from_brax' should be passed. |
| `--load_from_brax` | - | Flag to load parameters from Brax |
| `--skip_initialize` | - | Flag to skip the initialization step |
| `--only_initialize` | - | Flag to exit after initialization |
| `--continue_rsm` | `0` | If > 0, continue from a previous run with this multiplier for grid size |
| `--rsm_path` | `None` | Path to a pre-trained RSM model to load when using the `continue_rsm` option |
| `--no_train` | - | Flag to skip training and only run verification |
| `--policy_rollback` | - | Flag to enable policy rollback if training diverges |
| `--rollback_threshold` | `0.3` | Threshold for determining policy divergence |
| `--policy_path` | `None` | Path to load a pre-trained policy. If not provided, a predefined one will be used.` |
| `--init_with_static` | - | Flag to initialize with the old loss function |

### Batch and Buffer Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | `"16k"` | Batch size for verification (supports k/M/G suffixes) |
| `--learner_batch_size` | `"16k"` | Batch size for learning (supports k/M/G suffixes) |
| `--buffer_size` | `3000000` | Maximum size of the counterexample buffer |
| `--ds_type` | `"all"` | Type of counterexamples for training ('all' or 'hard') |

### PPO Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--ppo_iters` | `100` | Number of PPO iterations for controller pre-training |
| `--n_step` | `1` | Number of environment steps per training iteration |
| `--std_start` | `1.0` | Initial standard deviation for PPO's Gaussian policy |
| `--std_end` | `0.05` | Final standard deviation for PPO's Gaussian policy |
| `--normalize_r` | `0` | Whether to normalize rewards in PPO (1 = yes, 0 = no) |
| `--normalize_a` | `1` | Whether to normalize advantages in PPO (1 = yes, 0 = no) |

## Notes on Size Arguments

Size arguments like `--batch_size`, `--learner_batch_size`, and `--grid_size` support suffixes:
- `k` or `K`: Multiply by 1,024 (kilobyte)
- `M`: Multiply by 1,048,576 (megabyte)
- `G`: Multiply by 1,073,741,824 (gigabyte)

You can also use multiplication, e.g., `2*8*1k` is interpreted as 16,384. 


## Output and Artifacts

NeuralStoc generates the following outputs:

1. **Terminal Output**: Provides information on the verification/learning process, including the final proven probability bound.

2. **Artifacts Directory**: For each experiment (specified by `--exp_name`), the tool creates a directory containing:
   - Model checkpoints (`.jax` files) for the neural certificate and controller
   - Plots showing the value function across the state space with unsafe/target sets highlighted
   - Log files with detailed information about the verification process


## Note on Optimizations

NeuralStoc includes several key optimizations that improve its performance and scalability:

1. **Modified RASMs**: A slightly stricter variant of RASM certificates that simplifies the loss function used for learning.

2. **Local Lipschitz Analysis**: Replaces global Lipschitz analysis with local Lipschitz analysis, significantly simplifying the verification task and removing the Lipschitz regularization term from the loss function.

3. **Controller Rollback**: If learning fails to converge, the verifier utilizes a controller network from the last successful training iteration.


## Note on the obtained bounds

After the conditions are checked by the verifier module, the RASM network is normalized such that the sup. of V at the initial set is 1 and the inf. of V on the entire domain is 0. 
This normalization allows us to obtain even slightly better bounds than the verifier concluded.

## Smoke test and logging into the file

To check that the program runs smoothly, you can use the following command:
```bash
python3 scripts/rsm_loop.py --exp_name spend_old --initialize sac --plot --learner_batch_size 2 --batch_size 2 --env vpend_200 --model mlp --norm linf --ds_type all --v_lr 0.0001 --p_lr 0.000001 --lip_lambda 0.001 --train_p 3 --eps 0.01 --p_lip 4 --v_lip 15 --prob 1 --min_iters 0 --hidden_p 128 --hidden_v 128 --grid_size 4 --buffer_size 2 --spec stability --ppo_iters 1 --timeout 1
```

To output the logs properly in a file, you can follow the following example from the command in the `scripts/smoke_test.sh`:

```bash
bash scripts/smoke_test.sh > smoke_test.out
```

You can find the set of commands to replicate the results in the paper in the `scripts/commands.sh` file. You can follow the above command to log the output of each command in `scripts/commands.sh` in a file. To run a single experiment from the paper, select the corresponding command in `scripts/commands.sh` and comment out the rest. 


## Replication of the results in the paper

We ran our experiments each on Debian 12 with 16 `Intel Xeon Gold 6438Y+ 2 GHz` CPUS, 256 GB RAM, and a single L40S GPU with 48 GB VRAM. We used CUDA 12.6.3 and CudNN 8.9.5.30. We suggest the default docker file (`Dockerfile`) for a smooth run since the settings and packages are more compatible. However, because of the stochastic nature of the tool, you might not get the exact results as mentioned in the paper. For that reason, we provided another docker file (`Dockerfile-sim`) and its image (`neuralstoc-sim-artifact.tar.zip`) to get as close as possible to the setting of our experiments. You can follow the same instruction in [Installation](#installation) for running the docker container and instead build the docker image with this command (if you are building from the scratch):

```bash
docker build -f Dockerfile-sim -t neuralstoc .
```

### Using Provided Checkpoints

To facilitate evaluation without requiring full execution of experiments, we provide checkpoint files in the `scripts/checkpoints` folder. These checkpoints include:
- Models after the first successful learner-verifier iteration (named with pattern `*_loop_0.jax`)
- Final verified models at the end of the experiment (named with patterns like `*_loop_5.jax`, `*_loop_14.jax`, etc.)

Each checkpoint consists of two files:
- The main checkpoint file (e.g., `tri_new_stability_loop_0.jax`)
- A corresponding observation normalization file for SAC (e.g., `tri_new_stability_loop_0_obs_normalization.jax`)

When using the checkpoint commands, both files are automatically loaded when you specify the main checkpoint file path using the `--rsm_path` argument. Make sure the observation normalization file is near the main file with extra `_obs_normalization` suffix in the file name.

Due to the stochastic nature of the underlying algorithms (neural network training, optimization, reinforcement learning initialization, stochastic environments) and the dependency on specific hardware (GPU) and software drivers (CUDA/cuDNN versions), the exact runtime of experiments may vary between runs and different systems. These checkpoints allow you to bypass these kind of issues and lengthy runs and focus on specific parts of the learner-verifier loop or verify the final provided networks.

### Command Files for Different Stages

We provide specialized command files to help you evaluate different stages of the process:
- `scripts/commands_first_loop.sh`: Commands to run experiments from the first iteration checkpoints (using `--continue_rsm 1 --skip_initialize` flags and initial checkpoint files like `scripts/checkpoints/*_loop_0.jax`)
- `scripts/commands_final_loop.sh`: Commands to verify final models without retraining (using `--continue_rsm 1 --skip_initialize --no_train` flags and final checkpoint files)
- `scripts/commands.sh`: Original full commands for complete execution

The key differences between these command files:
1. `commands_first_loop.sh` loads checkpoints after the first successful iteration and continues training
2. `commands_final_loop.sh` loads the final checkpoints and runs only verification (with `--no_train` flag)

To evaluate our results using the provided checkpoints:

1. For quick verification of final results:
   ```bash
   # Choose a specific experiment from commands_final_loop.sh
   python3 scripts/rsm_loop.py --exp_name stri_new --load_from_brax --initialize sac --plot --learner_batch_size 8k --batch_size 32k --env vtri --model tmlp --norm linf --ds_type all --v_lr 0.0001 --p_lr 0.000001 --lip_lambda 0.2 --rollback_threshold 0.99 --prob 1 --min_iters 0 --train_p 0 --hidden_p 256 --hidden_v 256 --grid_size 32M --buffer_size 6000000 --n_local 4 --improved_loss --policy_rollback --estimate_expected_via_ibp --spec stability --continue_rsm 1 --skip_initialize --no_train --rsm_path scripts/checkpoints/tri_new_stability_loop_1.jax
   ```

2. To continue training from the first iteration:
   ```bash
   # Choose a specific experiment from commands_first_loop.sh
   python3 scripts/rsm_loop.py --exp_name stri_new --load_from_brax --initialize sac --plot --learner_batch_size 8k --batch_size 32k --env vtri --model tmlp --norm linf --ds_type all --v_lr 0.0001 --p_lr 0.000001 --lip_lambda 0.2 --rollback_threshold 0.99 --prob 1 --min_iters 0 --train_p 0 --hidden_p 256 --hidden_v 256 --grid_size 32M --buffer_size 6000000 --n_local 4 --improved_loss --policy_rollback --estimate_expected_via_ibp --spec stability --continue_rsm 1 --skip_initialize --rsm_path scripts/checkpoints/tri_new_stability_loop_0.jax
   ```

### Validating Tool Results

You have three options to validate the results of NeuralStoc:

1. **Full Execution**: Run the complete commands from `scripts/commands.sh` to replicate the entire experiment from scratch (requires significant time and resources):
   ```bash
   # Example command for stability specification with triple integrator
   python3 scripts/rsm_loop.py --exp_name stri_new --load_from_brax --initialize sac --plot --learner_batch_size 8k --batch_size 32k --env vtri --model tmlp --norm linf --ds_type all --v_lr 0.0001 --p_lr 0.000001 --lip_lambda 0.2 --rollback_threshold 0.99 --prob 1 --min_iters 0 --train_p 0 --hidden_p 256 --hidden_v 256 --grid_size 32M --buffer_size 6000000 --n_local 4 --improved_loss --policy_rollback --estimate_expected_via_ibp --spec stability
   ```

2. **Partial Training**: Continue training from the first successful iteration using commands in `scripts/commands_first_loop.sh` (less variance compared to our results - still requires significant time and resources)

3. **Quick Verification**: Verify final models without retraining using commands in `scripts/commands_final_loop.sh` (fastest option)

If you want to output your logs in a file, read the previous section ([Link to section](#smoke-test-and-logging-into-the-file)) to see how to output the log in a file. For all approaches, after execution:
- Check the terminal output for the final probability bound
- Examine the generated plots in the experiment directory to visualize the certificate function and sampled trajectories
- Review learning curves and verification grids in the experiment artifacts

Since our experiments were on cluster servers, you probably get memory errors if you run the same experiments locally. Consider using a lower learner batch size (`learner_batch_size`) and verifier batch size (`batch_size`) to resolve the GPU memory issue. To fix RAM out-of-memory errors, lower the grid size (`grid_size`) and buffer size (`buffer_size`) too.

We ran our experiments with a time limit of 24 hours (not including the controller initialization step). We strongly suggest running the experiments with a system configuration near to what we used for ours. Otherwise, it is not probably feasible to replicate the results in a reasonable time frame. The two following experiments have the least time to run based on our results. Consider running the following commands (with lower batch sizes and grid size if it doesn't fit in your system) if you want to run a subset of experiments to verify functionality of the tool.

### Computational Requirements and Stochastic Nature

It's important to note several characteristics of NeuralStoc that affect reproducibility:

1. **High Resource Requirements**: Our experiments require substantial computational resources:
   - CPU: 16 Intel Xeon Gold 6438Y+ 2 GHz CPUs
   - RAM: 256 GB
   - GPU: NVIDIA L40S with 48 GB VRAM
   - Specific CUDA (12.6.3) and cuDNN (8.9.5.30) versions
   
   Running these experiments on computers with lower specifications will be challenging and may require parameter adjustments that could affect outcomes.

2. **Non-Deterministic Behavior**: Due to the stochastic nature of:
   - Neural network training and optimization
   - Reinforcement learning initialization
   - Stochastic environments
   
   The exact runtime and numerical results may vary between runs and on different systems, even with identical parameters.

3. **Long Runtimes**: Our full experiments were run with a 24-hour timeout (excluding controller initialization). Even our fastest representative experiments took approximately 5-9 hours on our high-end cluster.

To address these challenges, we've provided:
- Docker images with specific environments
- Checkpoint files from different stages of successful runs
- Commands for evaluating partial or full results
- Guidance on parameter adjustments for lower-resource machines

When using reduced batch sizes, grid sizes, buffer size, or other parameters to accommodate lower-resource machines, expect both longer run times and potentially different final results compared to those reported in the paper.

**Reach-Avoid Example:**
```bash
python3 scripts/rsm_loop.py --exp_name cavoid_new --load_from_brax --initialize sac --plot --learner_batch_size 32k --batch_size 32k --env v2cavoid --model tmlp --norm linf --ds_type all --v_lr 0.0001 --p_lr 0.000001 --lip_lambda 0.2 --rollback_threshold 0.97 --prob 0.97 --min_iters 0 --train_p 0 --hidden_p 256 --hidden_v 256 --grid_size 32M --buffer_size 6000000 --n_local 10 --improved_loss --policy_rollback --estimate_expected_via_ibp --spec reach_avoid
```

**Stability Example:**
```bash
python3 scripts/rsm_loop.py --exp_name stri_new --load_from_brax --initialize sac --plot --learner_batch_size 8k --batch_size 32k --env vtri --model tmlp --norm linf --ds_type all --v_lr 0.0001 --p_lr 0.000001 --lip_lambda 0.2 --rollback_threshold 0.99 --prob 1 --min_iters 0 --train_p 0 --hidden_p 256 --hidden_v 256 --grid_size 32M --buffer_size 6000000 --n_local 4 --improved_loss --policy_rollback --estimate_expected_via_ibp --spec stability
```

Based our system configuration, these two experiments will take around 7-9 hours to run.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use NeuralStoc in your research, please cite our paper:

```
@inproceedings{neuralstoc2025,
  title={NeuralStoc: A Tool for Neural Stochastic Control and Verification},
  author={Ansaripour, Matin and Chatterjee, Krishnendu and Henzinger, Thomas A. and Lechner, Mathias and Verma, Abhinav and Žikelić, Đorđe},
  booktitle={Computer Aided Verification},
  year={2025}
}
```