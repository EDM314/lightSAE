# lightSAE: Lightweight Shared-Auxiliary Embedding for Multivariate Time Series Forecasting


## 🎯 Project Overview

This repository implements a novel approach to address **channel heterogeneity** in Multivariate Time Series Forecasting (MTSF). Our work introduces the **Shared-Auxiliary Embedding (SAE)** decomposition framework and its parameter-efficient variant **lightSAE**, specifically designed to model the heterogeneous characteristics across different channels in multivariate time series data.

### 🔬 Core Innovation

- **SAE Architecture**: Combines shared base embedding layers with channel-specific auxiliary embedding layers
- **lightSAE Module**: Parameter-efficient implementation leveraging low-rank structure and channel clustering properties
- **Heterogeneous Modeling**: Effectively captures and models the inherent differences between channels in IoT and time series systems

### 🏗️ Key Components

- **HeEmb Layer** (`layers/HeEmb.py`): Core heterogeneous embedding module implementing SAE architecture variants
- **Enhanced Models**: Integration with mainstream forecasting models (RLinear_Emb, RMLP_Emb, PatchTST_Emb, iTransformer_Emb)
- **Hierarchical Experiments**: Configuration-driven experiment management with multi-GPU parallel execution
- **Analysis Tools**: Weight analysis and visualization tools for studying low-rank structure and clustering properties

## 🚀 Quick Start

### Environment Setup

```bash
# Create conda environment
conda create -n lightsae python=3.8
conda activate lightsae

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. Download datasets from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)
2. Create dataset directory: `mkdir ./dataset`
3. Place CSV files directly in `./dataset/` (e.g., `./dataset/electricity.csv`)

## 🔧 Usage & Experiment Runner (`master_runner.py`) Guide

This guide explains how to use the `master_runner.py` script to run time series forecasting experiments defined by Python configuration files. The script supports hierarchical configurations, path-based parameter inference, and parallel execution across multiple GPUs.

## Overview

`master_runner.py` is designed to execute `run_longExp.py` with various parameter combinations. It first collects all experiment iterations from the provided configuration files into a central task pool. These tasks are then dynamically distributed among a specified number of worker processes, configured to run across one or more GPUs, for parallel execution. The script provides progress tracking, including success and failure counts for tasks.

## Directory Structure

A typical setup might look like this:

```
SimpleEmb/
├── run_longExp.py
├── logs/
│   └── master_runner.log # Generated log file
├── configs/
│   ├── master_runner.py
│   ├── utils.py
│   ├── dataset_defaults.py
│   ├── README.md  # This file
│   └── Exp1_sl96/
│       ├── ModelA_Emb/
│       │   └── IndEmb/
│       │       └── etth1_config.py
│       │       └── weather_config.py
│       └── ModelB_Emb/
│           └── SharedEmb/
│               └── traffic_config.py
└── dataset/
    ├── ETTh1.csv
    ├── weather.csv
    └── traffic.csv
```

- **`run_longExp.py`**: The main script that trains/tests a model. Located in the project root (`SimpleEmb/`).
- **`logs/master_runner.log`**: A log file automatically generated in the project root, containing detailed logs for all executed tasks.
- **`configs/master_runner.py`**: This script, which orchestrates the experiments.
- **`configs/utils.py`**: Contains helper functions, such as `generate_parameter_combinations`, for use in configuration files.
- **`configs/dataset_defaults.py`**: Located alongside `master_runner.py`. Defines default parameters for common datasets (e.g., paths, `enc_in`, `features`, `data_short_name`).
- **Experiment Configuration Files (`*_config.py`)**: Python files defining specific experiments. They are located in subdirectories within `configs` (e.g., `configs/Exp1_sl96/ModelA_Emb/IndEmb/etth1_config.py`).

## Running Experiments

Execute `master_runner.py` from the project root directory (`SimpleEmb/`).

**Command Syntax:**

```bash
python configs/master_runner.py [options] <config_file_path_1> [<config_file_path_2> ...]
```

**Arguments:**

-   `<config_file_path_...>`: One or more paths to your experiment configuration Python files.

**Options:**

-   `--parse_path`: (Optional) If provided, the script will attempt to infer `model_name`, `embedding_type`, and `dataset_short_name` from the path of the configuration file. This requires a specific directory structure for config files (see "Path Parsing Mode").
-   `--num_gpus <N>`: (Optional) Specifies the number of physical GPUs to utilize for parallel execution. Defaults to `1`. Tasks are distributed across these GPUs (e.g., GPU 0 to GPU N-1).
-   `--tasks_per_gpu <M>`: (Optional) Specifies the number of worker processes to run in parallel on *each* physical GPU specified by `--num_gpus`. Defaults to `1`. The total number of parallel worker processes will be `N * M`. For instance, with `--num_gpus 2` and `--tasks_per_gpu 3`, there will be 6 worker processes in total, with 3 processes assigned to physical GPU 0 and 3 to physical GPU 1.
-   `--smoke_test`: (Optional) If provided, runs all tasks in a smoke test mode. This typically means `run_longExp.py` will execute for a minimal number of epochs/batches. The `--smoke_test` flag is passed to `run_longExp.py`.
-   `--use_data_cache`: (Optional) If provided, enables the use of preprocessed dataset caching for `run_longExp.py`. When active, `run_longExp.py` will attempt to load `Dataset` objects from a cache directory. If a cache file for a specific dataset configuration doesn't exist or is invalid, `run_longExp.py` will process the data normally and then save the `Dataset` object to the cache for future runs. This can significantly speed up experiments by avoiding repeated data preprocessing, especially for large datasets or complex preprocessing steps. For example, if data preprocessing for each task takes approximately 2 seconds, and you have 1024 tasks (e.g., 4 models × 8 embedding types × 8 datasets × 4 configurations per dataset), enabling this feature could save around 2048 seconds (approximately 34 minutes).
-   `--force_refresh_cache`: (Optional) If provided along with `--use_data_cache`, forces `run_longExp.py` to re-process and overwrite existing cache files for all datasets involved in the current run. This is useful when the underlying data or preprocessing logic has changed, and you want to ensure the cache is updated. This flag is passed to `run_longExp.py`.
-   `--data_cache_path <path>`: (Optional) Specifies the directory where `run_longExp.py` should store and look for cached `Dataset` objects. Defaults to `./datacache/` relative to where `run_longExp.py` is executed (usually the project root). This path is passed to `run_longExp.py`.
-   `--max_numpy_threads_per_task <T>`: (Optional) Specifies the maximum number of threads that libraries like NumPy, Pandas (and underlying libraries like MKL, OpenBLAS, OpenMP) can use within each individual experiment task (`run_longExp.py` process). Defaults to `0`, which means no explicit limit is set, and libraries use their default behavior (often trying to use all available cores). Setting this to a small integer (e.g., `1` or `2`) can be very beneficial when running multiple tasks in parallel (especially on multi-GPU setups or when `tasks_per_gpu` > 1). It helps prevent CPU over-subscription, where too many threads compete for CPU resources, leading to reduced overall efficiency. This ensures smoother parallel execution and can improve GPU utilization by preventing CPU bottlenecks in data loading/preprocessing stages of each task.

**Examples:**

1.  **Run a single experiment serially (path parsing disabled):**
    ```bash
    python configs/master_runner.py configs/Exp1_sl96/ModelA_Emb/IndEmb/etth1_config.py
    ```
    In this mode, `etth1_config.py` must be comprehensive or specify `dataset_short_name` in its `base_config` to load defaults from `dataset_defaults.py`.

2.  **Run a single experiment with path parsing enabled on 2 GPUs:**
    ```bash
    python configs/master_runner.py --parse_path --num_gpus 2 configs/Exp1_sl96/ModelA_Emb/IndEmb/etth1_config.py
    ```
    The script will attempt to derive parameters from the path and distribute the experiment iterations (if multiple) across 2 GPUs.

3.  **Run multiple experiment files in parallel using 4 GPUs:**
    ```bash
    python configs/master_runner.py --num_gpus 4 configs/Exp1_sl96/ModelA_Emb/IndEmb/etth1_config.py configs/Exp1_sl96/ModelB_Emb/SharedEmb/traffic_config.py
    ```

4.  **Run all config files in a directory (using shell expansion, path parsing, on 2 GPUs):**
    ```bash
    python configs/master_runner.py --parse_path --num_gpus 2 configs/Exp1_sl96/ModelA_Emb/IndEmb/*.py
    ```

## User Confirmation and Progress

-   **Path Validation**: Before processing, the script validates all provided configuration file paths. It prints a summary of found and not-found files and asks for user confirmation (y/n) before proceeding.
-   **Progress Bar**: During execution, a `tqdm` progress bar displays the overall progress of tasks, along with real-time counts of successful and failed tasks.
-   **Logging**: A central log file, `logs/master_runner.log`, is created in the project root. This file provides comprehensive details for each task, including its start time, final status (SUCCEEDED/FAILED), completion time, duration, the full set of parameters passed to `run_longExp.py`, and the complete `stdout` and `stderr` output. The console output remains clean, focusing on the real-time progress bar, while all diagnostic information is preserved in the log file for detailed review and debugging.

## Load Balancing and Parallel Execution Strategy

`master_runner.py` employs several strategies to manage and optimize the parallel execution of experiment tasks, particularly when dealing with mixed workloads that include resource-intensive operations (e.g., processing large datasets like `traffic`). The goal is to achieve better load balancing and overall resource utilization:

1.  **Task Pooling and Dynamic Dispatch**:
    -   All individual experiment tasks are first generated from the provided configuration files and collected into a central pool.
    -   Multiple `gpu_worker` processes are launched. Each worker, upon becoming idle, dynamically fetches the next available task from this pool. This helps keep GPUs busy as long as there are pending tasks.

2.  **GPU-based Task Parallelization**:
    -   Users can specify the number of physical GPUs (`--num_gpus`) and the number of concurrent tasks per GPU (`--tasks_per_gpu`).
    -   The script assigns tasks to specific GPUs by setting the `CUDA_VISIBLE_DEVICES` environment variable for each worker process, ensuring that computational workloads are distributed as intended.

3.  **Task Shuffling for Homogenization**:
    -   Before tasks are placed into the execution queue, the entire list of generated tasks is **randomly shuffled**.
    -   **Purpose**: This strategy aims to break up any sequential ordering of tasks from the configuration files that might lead to resource bottlenecks (e.g., many CPU-heavy or I/O-heavy tasks running consecutively on the same workers).
    -   **Effect**: By shuffling, resource-intensive tasks are more likely to be interspersed with lighter ones across the workers and over time. This promotes a more even distribution of load, reducing idle times and potentially improving overall throughput.

4.  **Limiting Intra-Task Parallelism for CPU-bound Operations**:
    -   The `--max_numpy_threads_per_task <T>` option allows users to restrict the maximum number of CPU cores that can be used by libraries like NumPy, Pandas, and their underlying multi-threaded backends (e.g., MKL, OpenBLAS, OpenMP) *within each individual `run_longExp.py` task process*.
    -   **Context**: When data loading and preprocessing (e.g., in `Dataset_Custom` with `num_workers=0`) involve CPU-intensive operations on large datasets, each task might try to utilize all available CPU cores by default.
    -   **Purpose**: When running multiple `run_longExp.py` tasks in parallel (e.g., on multiple GPUs or with `tasks_per_gpu > 1`), setting `<T>` to a small value (e.g., 1 or 2) prevents these parallel tasks from collectively overwhelming the CPU.
    -   **Effect**: This mitigates CPU over-subscription, ensures that CPU resources remain available for other essential operations (like data transfer to GPUs), and can lead to smoother, more stable parallel execution with potentially better GPU utilization, as CPU-bound data preparation stages are less likely to become a shared bottleneck.

By combining these approaches, `master_runner.py` strives for robust and efficient execution of numerous experiments, balancing diverse computational demands.

## Experiment Configuration Files (`*_config.py`)

Each experiment configuration file (e.g., `etth1_config.py`) is a Python script that defines the parameters for a set of experiments. It should primarily define two main variables:

1.  **`base_config` (dict):**
    -   Defines parameters shared across all iterations within this specific experiment file.
    -   These parameters override any defaults loaded from `dataset_defaults.py` or inferred from the path (if `--parse_path` is used).
    -   **Important for non-path-parsing mode**: If `--parse_path` is NOT used, `base_config` should typically define:
        -   `model`: The model name (e.g., `"ModelA_Emb"`).
        -   `model_type`: The embedding type (e.g., `"IndEmb"`).
        -   `dataset_short_name`: (Optional, but recommended) The short name of the dataset (e.g., `"etth1"`). If provided, `master_runner.py` will try to load defaults for this dataset from `dataset_defaults.py`. If not provided, `base_config` must supply all necessary dataset parameters (like `root_path`, `data_path`, `data`, etc.).
    -   `model_id_prefix`: (Optional) If you want to customize the first part of the generated `model_id`. If not provided, `master_runner.py` will attempt to use the `data_short_name`.

2.  **`specific_iterations` (list of dicts):**
    -   This list defines all the individual parameter combinations for the experiments to be run from this file. Each dictionary in the list represents one experiment iteration and contains parameters specific to that iteration.
    -   This list is typically generated using the `generate_parameter_combinations` helper function located in `configs/utils.py`. `master_runner.py` automatically adds the `configs` directory to Python's system path, so you can import the function directly within your configuration files, regardless of their subdirectory depth.
        ```python
        # Correct way to import the helper function in any *_config.py file:
        from utils import generate_parameter_combinations
        ```
    -   The `generate_parameter_combinations` function takes two main arguments:
        *   **`tuning_branch_definitions` (list of dicts):**
            This is a list where each dictionary defines a "tuning branch." A branch represents a set of tuning parameters that will be combined using a Cartesian product.
            -   Each branch dictionary must have a `'params'` key. The value is another dictionary holding the actual tuning parameters for this branch (e.g., `{'learning_rate': [0.001, 0.0005], 'optimizer': 'adam'}`).
            -   For each parameter within a branch's `'params'` (e.g., `learning_rate`), you provide a list of values (e.g., `[0.001, 0.0005]`) or a single value (e.g., `'adam'`, which the function treats as `['adam']`).
            -   All parameter-value lists within a single branch's `'params'` are combined using a Cartesian product. For example, if a branch has `{'lr': [0.1, 0.01], 'opt': ['sgd', 'adam']}`, it will generate 4 tuning combinations for that branch.
            -   Each branch dictionary can also have an optional `'name'` key (e.g., `'learned_router_settings'`) for logging or identification purposes.
            -   If `tuning_branch_definitions` is an empty list, or if all defined branches have empty `'params'` dictionaries, then the "tuning" part contributes no parameters (equivalent to a single empty tuning set). This is useful if you only want to vary `specific_params_dict`.

        *   **`current_specific_params_dict` (dict):**
            This dictionary defines parameters that are combined in an element-wise (or "zipped") manner.
            -   Keys are parameter names (e.g., `'pred_len'`, `'hidden_dim'`).
            -   Values are lists of parameter values (e.g., `'pred_len': [96, 192, 336]`, `'hidden_dim': [128, 256, 512]`).
            -   **Crucially, all lists provided as values in this dictionary must have the same length** (or all be empty). The combinations are formed by taking the first element from each list for the first specific combination, the second element from each list for the second, and so on.
            -   If this dictionary is empty or all its value lists are empty, it contributes no specific parameters (equivalent to a single empty specific set).

    -   **Generating `specific_iterations`:**
        ```python
        # Example:
        tuning_defs = [
            {'name': 'branch1', 'params': {'lr': [0.01, 0.005], 'model_variant': ['A']}},
            {'name': 'branch2', 'params': {'lr': [0.001], 'model_variant': ['B', 'C']}}
        ]
        specific_defs = {'pred_len': [96, 192], 'seq_len': [48, 96]}

        specific_iterations = generate_parameter_combinations(tuning_defs, specific_defs)
        # This would result in (2*1 + 1*2) * 2 = 4 * 2 = 8 total iterations.
        ```
    -   The `master_runner.py` script will then take each dictionary from the `specific_iterations` list, merge it with the `base_config` (where `specific_iterations` parameters take precedence in case of overlap), and queue an experiment for each resulting complete configuration.