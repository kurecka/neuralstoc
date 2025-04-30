# NeuralStoc: A Tool for Neural Stochastic Control and Verification

This repository contains the supplementary code for the paper "NeuralStoc: A Tool for Neural Stochastic Control and Verification" (CAV 2025), by Matin Ansaripour, Krishnendu Chatterjee, Thomas A. Henzinger, Mathias Lechner, Abhinav Verma, and Đorđe Žikelić.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15220506.svg)](https://doi.org/10.5281/zenodo.15220506)

## Introduction

NeuralStoc is a tool for neural controller synthesis and verification in discrete-time stochastic dynamical systems. The tool implements and builds upon the first learner-verifier framework for neural stochastic control with certificates, by jointly learning and/or formally verifying a neural controller together with a neural supermartingale certificate of its correctness. NeuralStoc provides a unified interface for analyses with respect to reachability, safety, reach-avoidance, and stability specifications.


The key features of NeuralStoc include:

- **Optimizations**: Introduces several optimizations including modified reach-avoid supermartingales (RASMs), local Lipschitz analysis, and controller rollback, leading to significant improvements in practical performance and scalability.
- **Scalability**: The first tool able to solve neural stochastic control and verification tasks in 4-dimensional environments (4D state space + 2D control input space).

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
   - [Option 1: Using Docker (Recommended)](#option-1-using-docker-recommended)
   - [Option 2: Direct Installation](#option-2-direct-installation)
4. [Common Issues](#common-issues)
5. [Smoke Test and Logging](#smoke-test-and-logging)
   - [Smoke-test 1: Toy Example](#smoke-test-1-toy-example)
   - [Smoke-test 2: Experiment from the Paper](#smoke-test-2-experiment-from-the-paper)
   - [Smoke-test 3: Checkpoint Loading](#smoke-test-3-checkpoint-loading)
6. [Usage](#usage)
   - [Using Configuration Files](#using-configuration-files)
   - [Configuration File Format](#configuration-file-format)
7. [Replication of the Results in the Paper](#replication-of-the-results-in-the-paper)
   - [System Specifications for Our Experiments](#system-specifications-for-our-experiments)
   - [Reproducibility Considerations](#reproducibility-considerations)
   - [1. Full Experiment Execution — Reproduce Results from Scratch](#1-full-experiment-execution--reproduce-results-from-scratch)
   - [2. Partial Experiment Execution](#2-partial-experiment-execution)
     - [2.1. Continue from First Successful Iteration](#21-continue-from-first-successful-iteration)
     - [2.2. Verification of Final Results](#22-verification-of-final-results)
   - [Output Validation](#output-validation)
8. [Tool Overview](#tool-overview)
9. [Project Structure](#project-structure)
10. [Command Line Arguments](#command-line-arguments)
   - [Configuration File Settings](#configuration-file-settings)
   - [General Settings](#general-settings)
   - [Available Environments](#available-environments)
   - [Probability and Verification Settings](#probability-and-verification-settings)
   - [Network and Learning Settings](#network-and-learning-settings)
   - [Initialization Settings](#initialization-settings)
   - [Batch and Buffer Settings](#batch-and-buffer-settings)
   - [PPO Settings](#ppo-settings)
   - [Notes on Size Arguments](#notes-on-size-arguments)
11. [Output and Artifacts](#output-and-artifacts)
    - [1. Streaming terminal logs](#1-streaming-terminal-logs)
    - [2. Persistent artifacts on disk](#2-persistent-artifacts-on-disk)
    - [3. Run summary logs](#3-run-summary-logs)
12. [The bound_co_factor Parameter](#the-bound_co_factor-parameter)
13. [Note on Optimizations](#note-on-optimizations)
14. [Note on the Obtained Bounds](#note-on-the-obtained-bounds)
15. [License](#license)
16. [Citation](#citation)

## System Requirements

- Python 3.10 or higher
- CUDA 12 (for GPU acceleration — ensure a compatible cuDNN installation is correctly configured)
- NVIDIA GPU with CUDA support and sufficient VRAM for your intended use
- Adequate RAM capacity for your intended use
- CPU with sufficient cores and frequency suitable for your intended use


## Installation

### Option 1: Using Docker (Recommended)

The Docker installation method is recommended as it ensures all dependencies are correctly configured. Please install the Docker first.

On [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15220506.svg)](https://doi.org/10.5281/zenodo.15220506), we provided a pre-built docker image with the name `neuralstoc.tar.zip` (built with `Dockerfile`). In our Docker image, we use the same versions of all software that were used in our experiments. To run the image, you can use the following instruction:

1. Unzip the downloaded file:
```bash
unzip neuralstoc.tar.zip
```

2. Load the Docker image:
```bash
docker load < neuralstoc.tar
```

3. Run the container with GPU support:
```bash
docker run --gpus all -it -v $(pwd):/app neuralstoc
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
docker run --gpus all -it -v $(pwd):/app neuralstoc
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

## Common Issues

Below we document frequently reported obstacles together with recommended remedies.  If you encounter problems not covered here please open an issue on GitHub so that we can extend this list.

### 1. CUDA "factory already registered" messages when using the Docker image

**Symptoms**

When running any script inside the Docker container, the following messages appear repeatedly and the smoke tests seem to *fail*:

```
E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
E0000 00:00:1745341703.916274      48 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1745341703.923530      48 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
```

**Explanation & Fix**

These *are only warnings*, not errors.  They stem from JAX initialising CUDA primitives (via XLA and TensorFlow bindings) more than once.  The program continues normally after printing them.  You can safely ignore the messages.


### 2. Very large Docker image (slow download / transfer)

**Symptoms**

Downloading `neuralstoc.tar.zip` takes several hours and moving the uncompressed `*.tar` file to another machine is similarly slow.

**Explanation & Fix**

The image is large (~10 GB) because it bundles:

* CUDA 12 runtime and cuDNN libraries
* A minimal Linux distribution
* All Python dependencies (JAX, Flax, Brax, PyTorch, …)

This creates a replication of the environment at the cost of size.  To shorten the overall process, download the archive directly on the target server (after logging in via SSH) and unpack it there:

```bash
curl -L -o neuralstoc.tar.zip "https://zenodo.org/record/15313260/files/neuralstoc.tar.zip?download=1"
unzip neuralstoc.tar.zip
```

### 3. Missing NVIDIA CUDA toolkit on the machine

**Symptoms**

Inside or outside the Docker container the command

```bash
nvcc --version
```

does not exist or fails, indicating that the CUDA toolkit is absent.

**Fix**

Install the toolkit (Ubuntu/Debian example):

```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

After installation, restart the terminal so that the `nvcc` binary becomes available in your `PATH`.

### 4. `ImportError: ncclCommRegister` when importing PyTorch

**Symptoms**

Running the smoke test yields:

```
ImportError: /opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so: undefined symbol: ncclCommRegister
```

**Explanation & Fix**

This indicates a mismatch between the installed PyTorch wheels and the version of NCCL / CUDA in the container (rarely triggered when additional packages are installed manually).  Re-install PyTorch 2.1.1, which is compatible with CUDA 12 used in the image:

```bash
pip install --no-cache-dir torch==2.1.1
```

Note: swapping out PyTorch (or any major dependency) can introduce numerical differences from the paper because the tool and environments are inherently stochastic.

## Smoke Test and Logging

To verify that NeuralStoc runs correctly on your system, we provide three smoke tests to test our tool.

### Smoke-test 1: Toy Example

The first smoke test is a simple toy example that should complete in under 20 minutes with GPU acceleration, after which we manually impose a timeout. This smoke test can be used to test if the tool is running properly and printing out the desired output. Note that for this smoke test we are using slightly different parameter values compared to defaults in Table 3 and 4 in the paper, which is done in order to speed up the test.

```bash
python3 scripts/rsm_loop.py --env vpend --initialize ppo --task control --spec stability --prob 1 --num_layers_v 2 --num_layers_p 2 --hidden_p 64 --hidden_v 64 --exp_name spend --plot --learner_batch_size 2 --batch_size 2 --n_local 2 --grid_size 64 --buffer_size 4 --ppo_iters 1 --timeout 1 --no_config --improved_loss --policy_rollback --estimate_expected_via_ibp
```

To output the logs properly in a file, you can follow the following example from the command in the `scripts/smoke_test.sh`:

```bash
bash scripts/smoke_test.sh > smoke_test.out
```

When running the smoke test, you should observe the following sequence:
1. One iteration of PPO for controller initialization
2. Policy evaluation with reporting of the percentage of trajectories reaching the target
3. First learner-verifier iteration:
   - Training of controller and certificate
   - Computation and reporting of local Lipschitz constants
   - Verification with reporting of specification violations
4. Logging of computed parameters (e.g., Lipschitz constants)
5. Termination with "Timeout!" message

### Smoke-test 2: Experiment from the Paper

If preferred, you can run an actual experiment from our paper (2D Linear benchmark from Table 2 in the paper, stability experiments). As reported in Table 2, running this experiment on our system takes 2-3 hours.

```bash
python3 scripts/rsm_loop.py --spec stability --task control --prob 1 --initialize sac --hidden_p 256 --hidden_v 256 --num_layers_p 2 --num_layers_v 2 --env v2lds --exp_name slds_new --load_from_brax
```

When this experiment successfully verifies the controller, you should see the message:
```
Stability bound is 100.0%
```

### Smoke-test 3: Checkpoint Loading

Reproducing our experimental results requires higher system requirements and longer runtimes. Hence, in [Replication of the Results in the Paper](#replication-of-the-results-in-the-paper) we will also provide an option that streamlines experiment reproduction by using checkpoints from our experiments. This smoke test runs a comprehensive series of checkpoint loading tests (28 tests in total) to verify that all checkpoint variations work correctly:

```bash
python3 scripts/checkpoints_smoke_test.py
```

The test script:
- Tests multiple environments (e.g., pendulum, LDS, collision avoidance, triple integrator)
- Validates both specification types (reach-avoid and stability)
- Checks both first iteration checkpoints (loop_0) and final checkpoints
- Generates detailed logs for each test in the `logs` directory

If successful, you'll see "Smoke test passed!" with a summary of all passing tests. Otherwise, the script will report which specific tests failed. The full test suite takes approximately 30 minutes to complete, depending on your system specifications.


## Usage

The main entry point is the `rsm_loop.py` script in the `scripts` directory:

```bash
python3 scripts/rsm_loop.py --env [environment] --task [task_type] --spec [specification] --prob [probability] --initialize [RL_algorithm] [[additional options]]
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

NeuralStoc supports YAML configuration files to manage experiment parameters. This approach makes it easier to manage and share experiment configurations. The program loads the config from the default path `scripts/config.yaml`. If you want not to use the config file, you can pass `--no_config`. The default path can be changed using `--config_path` option.

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


## Replication of the Results in the Paper

### System Specifications for Our Experiments

Our experiments were conducted using the following high-performance computing resources:
- **CPU**: 16 Intel Xeon Gold 6438Y+ 2 GHz CPUs
- **RAM**: 256 GB
- **GPU**: NVIDIA L40S with 48 GB VRAM
- **CUDA**: Version 12.6.3
- **cuDNN**: Version 8.9.5.30

### Reproducibility Considerations

It's important to note several factors that affect reproducibility of our results:

1. **High Resource Requirements**: Running these experiments on computers with lower specifications will be challenging and may require parameter adjustments that could affect runtime and numerical results. When using reduced batch sizes (`--batch_size`, `--learner_batch_size`), grid sizes (`--grid_size`), buffer size (`--buffer_size`), or bound computation grid size (via `--bound_co_factor`) to accommodate lower-resource machines, expect both longer run times and potentially different final runtime and numerical results compared to those reported in the paper depending on your system specifications.

Reducing these parameters has two fundamental effects: (i) **learner-side information loss** – fewer samples and counter-examples are processed in each learner–verifier iteration, so the learner sees a less informative view of the state space and may converge more slowly or to a different solution; and (ii) **verifier-side precision loss** – coarser verification and bound-computation grids yield looser over-approximations of the certificate function, which directly influences the tightness of the probability bounds that can be proven.

For runs involving the NeuralStoc tool results (Tables 1 and 2, NeuralStoc columns), these parameters can be adjusted to potentially accommodate lower-resource machines, although be aware that this might affect runtime and numerical results. You can lower these parameters by editing these lines in the `scripts/config.yaml` file: 
   ```yaml  
   # Batch, Grid, and Bound Computation Sizes - Timeout (in minutes)
   batch_size: "32k"  # Batch size for verification
   learner_batch_size: "32k"  # Batch size for learning
   grid_size: "32M"  # Grid size for verification
   buffer_size: 6000000  # Buffer size
   bound_co_factor: 1  # Co-factor for the bound computation grid size
   timeout: 1440  # Timeout in minutes (default: 24 hours) 
   ```

For example, the following configuration **might** be suitable for a GPU with around 12 GB VRAM and 64 GB RAM, potentially requiring a long timeout (e.g., `--timeout 720` for 12 hours):
   ```yaml
   # Batch, Grid, and Bound Computation Sizes - Timeout (in minutes)
   batch_size: "8k"  # Batch size for verification
   learner_batch_size: "4k"  # Batch size for learning
   grid_size: "4M"  # Grid size for verification
   buffer_size: 3000000  # Buffer size
   bound_co_factor: 0.5  # Co-factor for the bound computation grid size
   timeout: 720 # Timeout in minutes
   ```
For baseline runs (Tables 1 and 2, Baseline columns), since the baseline does not use the configuration file, you should pass or edit the same arguments directly in the command line provided in the corresponding `.sh` script (e.g., add `--batch_size 8k --learner_batch_size 4k --grid_size 4M --buffer_size 3000000 --bound_co_factor 0.5 --timeout 720`).

2. **Non-Deterministic Behavior**: The exact runtime and numerical results may vary between runs and on different systems due to:
   - Stochastic nature of neural network training and optimization
   - Randomness in reinforcement learning initialization
   - Inherent stochasticity in the environments

3. **Long Runtimes**: Our full experiments were run with a 24-hour timeout (excluding controller initialization). Even our fastest experiments took approximately 5-9 hours on our high-end computing cluster.

To accommodate these challenges, we provide three following approaches for reproducing our results, with varying levels of computational requirements. For each we provide bash files that you can run using this command:

```bash
bash <path_to_file>
```

Since each file consists of several commands, if you want to run a specific experiment, please find its command from its corresponding file and run it in the terminal or use this way to output in a file:

```bash
<command> > log.out
```

### 1. Full Experiment Execution — Reproduce Results from Scratch

This approach runs the complete experiment pipeline from initialization to final verification, requiring the longest runtime and might give different final runtime and numerical results compared to those reported in the paper depending on your system specifications (as mentioned before).

**Time Required**: Approximately 6-24+ hours per experiment (using our system setting or higher - full set: approximately 240 hours)

**Scripts**:
- **Table 1 NeuralStoc**: `scripts/table1_neuralstoc_full.sh`
- **Table 1 Baseline**: `scripts/table1_baseline_full.sh`
- **Table 2 NeuralStoc**: `scripts/table2_neuralstoc_full.sh`
- **Table 2 Baseline**: `scripts/table2_baseline_full.sh`

The two fastest experiments for `NeuralStoc` based on our results are for Collision Avoidance (reach-avoid) and Triple Integrator (stability). Consider running their commands (see "High Resource Requirements" in [Reproducibility Considerations](#reproducibility-considerations) if it doesn't fit in your system) if you want to run a subset of experiments to verify functionality of the tool.


### 2. Partial Experiment Execution

To facilitate evaluation without requiring full execution of experiments, we provide checkpoint files in the `scripts/checkpoints` folder. These checkpoints include:
- Models after the first successful learner-verifier iteration (named with pattern `*_loop_0.jax`)
- Final verified models at the end of the experiment (named with patterns like `*_loop_5.jax`, `*_loop_14.jax`, etc.)

Each checkpoint consists of two files:
- The main checkpoint file (e.g., `tri_new_stability_loop_0.jax`)
- A corresponding observation normalization file for SAC (e.g., `tri_new_stability_loop_0_obs_normalization.jax`)

When using the checkpoint commands, both files are automatically loaded when you specify the main checkpoint file path using the `--rsm_path` argument. Make sure the observation normalization file is near the main file with the extra `_obs_normalization` suffix in the file name.

If, after loading a checkpoint, the policy appears not to load correctly – the initial empirical evaluation printed by the tool shows that no trajectory reaches the target **AND** the verificatoin was not successful – add the flag `--load_scratch` to the command. This forces the program to rebuild the model skeleton from scratch and then copy the parameters from the checkpoint.

Note: The empirical evaluation of policies may be random due to random initialization and the stochastic nature of the environment. Use the verification results instead for more accurate results.

#### 2.1. Continue from First Successful Iteration

This set of experiments load the model after the first learning iteration and starts the loop with verification (it continues with further learner-verifier loop as normal). The purpose is to have the same initialization to reduce the impact of stochasticity of the system and for the experiments to get as close as possible to the runtime and numerical results of the paper. You still need significant time and resources to run the experiments.

**Time Required**: Approximately 5-24+ hours per experiment (using our system setting or higher - full set: approximately 220 hours)

**Scripts**:
- **Table 1 NeuralStoc**: `scripts/table1_neuralstoc_first_loop.sh`
- **Table 1 Baseline**: `scripts/table1_baseline_first_loop.sh`
- **Table 2 NeuralStoc**: `scripts/table2_neuralstoc_first_loop.sh`
- **Table 2 Baseline**: `scripts/table2_baseline_first_loop.sh`


#### 2.2. Verification of Final Results

This option loads the *final* checkpoint files that we provide and **re-runs only the verification phase** – no additional learning is performed.  Consequently, it is the fastest way to validate the numerical results reported in the paper.  If a checkpoint cannot be verified on the first attempt the loop automatically starts another verification iteration, each time enlarging the grid by 10 % per dimension.  We recommend running at least two iterations.  As soon as the desired probability bound (or any bound you consider satisfactory) has been proven you can safely stop the program – see "Examining results" in the "Output Validation" subsection below for how to decide when to interrupt.

**Time Required**: Approximately 2-4 hours per experiment (full set: approximately 30-50 hours)

**Scripts**:
- **Table 1 NeuralStoc**: `scripts/table1_neuralstoc_final_loop.sh`
- **Table 1 Baseline**: `scripts/table1_baseline_final_loop.sh`
- **Table 2 NeuralStoc**: `scripts/table2_neuralstoc_final_loop.sh`
- **Table 2 Baseline**: `scripts/table2_baseline_final_loop.sh`

### Output Validation

To validate the output and compare with the results in the paper:

1. **Outputting logs to a file**: When running any experiment, you can redirect the output to a file:
   ```bash
   bash scripts/table2_neuralstoc_final_loop.sh > results.out
   ```
   
   Or for a specific command:
   ```bash
   python3 scripts/rsm_loop.py [arguments] > results.out
   ```

2. **Examining results**:
   - Watch the terminal output for either a success message such as `Stability bound is 100.0%` or the fallback message `Timeout!`.
   - After every iteration the complete `info` dictionary is printed.  The same dictionary is appended to `study_results/info_<exp_name>.log` at the end of the run.  The field `max_actual_prob` shows the best formally-proven probability bound so far (`-1` means no bound yet).
   - For *reach-avoid* and *safety* specifications intermediate bounds are continuously written to the file `log_new_bound`; inspect the column `actual_reach_prob` to monitor progress.  If this value already meets your requirement you can interrupt the process right after the tool prints `SAVED` and starts the next iteration since it has already saved your model for that iteration.
   - All controller/certificate checkpoints and diagnostic plots are stored in the experiment folder created by the tool.
   - For a detailed description of all logs and files written by the tool see the section [Output and Artifacts](#output-and-artifacts).


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
│   ├── various .sh files      # Example command configurations
│   └── config.yaml            # Default configuration file template
└── src/neuralstoc/            # Core library
    ├── environments/          # Environment implementations
    │   ├── vrl_environments.py # Gym environments
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

## Command Line Arguments

### Configuration File Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--no_config` | - | Flag not to load configuration from a file |
| `--config_path` | `scripts/config.yaml` | Path to the configuration file |

### General Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--exp_name` | `"rasm_lds"` | Name of the experiment for saving artifacts |
| `--model` | `"tmlp"` | Network architecture type ('mlp' or 'tmlp' for local Lipschitz calculations) |
| `--env` | `"vlds"` | Environment to use (e.g., 'vlds', 'vpend', 'vhuman2') |
| `--task` | `"control"` | Task type ('control' or 'verification') |
| `--spec` | `"reach_avoid"` | Specification type ('reach_avoid', 'safety', 'reachability', or 'stability') |
| `--timeout` | `1440` | Maximum runtime in minutes |
| `--plot` | - | Flag to enable plotting during the run |
| `--env_dim` | `None` | Environment dimension (optional - for LDS 4D) |

### Available Environments

| Environment Argument (`--env`) | Description |
|----------|-------------|
| `v2cavoid` | Collision Avoidance Environment (RASM paper) |
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
| `--soft_constraint` | - | If set, the loop terminates if a bound is verified even lower than the one passed as `prob`, otherwise it continues the learner-verifier loop until it either reaches the timeout or verify a probability threshold higher than `prob`|
| `--grid_size` | `"16M"` | Target size of the verification grid (supports k/M/G suffixes) |
| `--bound_co_factor` | `1` | Co-factor to scale the grid size used for bound computation of certification value on the regions (reduces memory usage when < 1 with more loose bounds). See [The bound\_co\_factor Parameter](#the-bound_co_factor-parameter) for details. |

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
| `--train_p` | `3` | After which iteration to start training the policy parameters |
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
| `--rollback_threshold` | `0.99` | Threshold for determining policy divergence |
| `--policy_path` | `None` | Path to load a pre-trained policy. If not provided, a predefined one will be used. |
| `--load_scratch` | - | Flag to load the model skeleton from scratch and copy the parameters when loading a controller from a checkpoint. Use this if the controller seems not to load correctly when using SAC. |
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

NeuralStoc produces the following outputs:

### 1. Streaming terminal logs
* During each learner–verifier iteration the tool prints training losses, Lipschitz bounds, number of violations found, etc.
* When training for the current iteration finishes, the full `info` dictionary – capturing statistic used by the verifier – is printed.
* Upon termination the tool prints either a success line, e.g.
  ```
  Stability bound is 100.0%
  ```
  or, if the wall-clock limit given by `--timeout` has been reached, simply `Timeout!`.

### 2. Persistent artifacts on disk
For every experiment (`--exp_name`) a dedicated directory is created that contains
* **Model checkpoints** (`*.jax`) for both the neural certificate and the controller are saved after every iteration;
* **Plots** that visualise the certificate surface together with unsafe/target sets and sampled trajectories (for high-dimensional tasks the plot uses a user-defined projection, see the `plot_l` function in `loop.py`).

### 3. Run summary logs
* The `info` dictionary of the last completed iteration is appended to
  `study_results/info_<exp_name>.log`.  The entry `max_actual_prob` records the highest probability bound proven so far (-1 for none).
* For specifications where the loop may continue improving a bound (reach-avoid and safety) each newly proven bound is also appended to `log_new_bound` alongside the experiment name.  The column `actual_reach_prob` exposes the the bound that is obtained.

Tip: If the probability in `log_new_bound` already satisfies your needs you may abort the run right after the tool prints `SAVED` and starts the next iteration – the corresponding checkpoints have been written.  Alternatively, you can
* start the run with `--soft_constraint` to halt automatically after the first proven bound, or
* reduce `--timeout` to impose a stricter wall-clock limit.

## The bound_co_factor Parameter

The `--bound_co_factor` argument allows scaling the grid density used specifically for computing bounds on the certificate function's values (e.g., `compute_bound_init`, `compute_bound_domain` in the `RSMVerifier`). This is distinct from the main verification grid size controlled by `--grid_size`. It provides an efficient way to trade off memory usage and verification precision related to bound computation without reconfiguring all grid-related parameters:

- **Values < 1**: Reduce the bound computation grid density. This lowers memory consumption during bound calculation steps but might lead to looser bounds and potentially affect the precision of the overall verification numerical result or require more iterations.
- **Values > 1**: Increase the bound computation grid density. This improves the precision of the computed bounds at the cost of higher memory usage during those steps.
- **Default value = 1**: Uses the standard bound computation grid sizes.

The parameter works by scaling a base number of points per dimension (`n`) differently depending on the state space dimension (`env_dim`), and then creating a grid of size `n ** env_dim`. The base `n` is determined as follows within the `get_n_for_bound_computation` function:
- 2D environments: `n = int(200 * bound_co_factor)`
- 3D environments: `n = int(150 * bound_co_factor)`
- 4D environments: `n = int(100 * bound_co_factor)`
- Higher dimensions: `n = int(25 * bound_co_factor)`

Adjusting `--bound_co_factor` can be particularly useful on memory-constrained systems.

## Note on Optimizations

NeuralStoc includes several key optimizations that improve its performance and scalability:

1. **Modified RASMs**: A slightly stricter variant of RASM certificates that simplifies the loss function used for learning.

2. **Local Lipschitz Analysis**: Replaces global Lipschitz analysis with local Lipschitz analysis, significantly simplifying the verification task and removing the Lipschitz regularization term from the loss function.

3. **Controller Rollback**: If learning fails to converge, the verifier utilizes a controller network from the last successful training iteration.


## Note on the Obtained Bounds

After the conditions are checked by the verifier module, the RASM network is normalized such that the sup. of V at the initial set is 1 and the inf. of V on the entire domain is 0. 
This normalization allows us to obtain even slightly better bounds than the verifier concluded.

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