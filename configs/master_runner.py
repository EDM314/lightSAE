import argparse
import importlib.util
import os
import subprocess
import sys
import tqdm
import multiprocessing
import random
import datetime
import pprint
import uuid

# --- Constants and Project Root Setup ---

def get_project_root(script_path):
    """Determines the project root. Assumes master_runner.py is in configs/ within the project root."""
    return os.path.dirname(os.path.dirname(script_path))

PROJECT_ROOT = get_project_root(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RUN_LONG_EXP_SCRIPT_PATH = os.path.join(PROJECT_ROOT, "run_longExp.py")

try:
    from dataset_defaults import DATASET_DEFAULTS
except ImportError:
    print("Warning: dataset_defaults.py not found. Proceeding without dataset defaults.", file=sys.stderr)
    DATASET_DEFAULTS = {}

# --- Logging Infrastructure ---

class RunLogger:
    """Encapsulates all logging operations for a master_runner execution."""
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.summary_log_path = os.path.join(log_dir, 'summary.log')
        self.all_outputs_log_path = os.path.join(log_dir, 'all_tasks_output.log')
        
        self._summary_f = open(self.summary_log_path, 'w')
        self._all_outputs_f = open(self.all_outputs_log_path, 'w')
        self._write_headers()

    def _write_headers(self):
        self._summary_f.write(f"--- Task Execution Summary ---\nLog Directory: {os.path.abspath(self.log_dir)}\n\n")
        self._all_outputs_f.write(f"--- Consolidated Task Outputs ---\nLog Directory: {os.path.abspath(self.log_dir)}\n\n")
        self.flush()

    def log_task_result(self, result: dict):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        task_data = result.get('task_data', {})
        meta_params = task_data.get('meta_params', {})
        all_params = task_data.get('all_params', {})
        
        # Use the descriptive log_display_id from meta_params
        log_display_id = meta_params.get('log_display_id', 'UnknownTask')
        gpu_id = result.get('gpu_id', '?')
        
        # Use the more descriptive status from the result dictionary
        default_status = "FAILURE" if not result.get('success') else "SUCCESS"
        status_str = result.get('status', default_status).upper()

        # 1. Console Log
        status_msg = f"[{timestamp}] [{status_str}] Task '{log_display_id}' on GPU {gpu_id}"
        if result['success']:
            tqdm.tqdm.write(status_msg, file=sys.stdout)
        else:
            tqdm.tqdm.write(f"{status_msg}. See details in: {self.all_outputs_log_path}", file=sys.stderr)

        # 2. Summary Log
        self._summary_f.write(status_msg + '\n')
        
        # 3. Detailed Log for all tasks
        self._all_outputs_f.write(f"\n{'='*20} Task: {log_display_id} | Status: {status_str} | Timestamp: {timestamp} {'='*20}\n")
        self._all_outputs_f.write(f"GPU ID: {gpu_id}\n\n")
        
        self._all_outputs_f.write("Complete Parameters (for debugging and reproduction):\n")
        self._all_outputs_f.write(pprint.pformat(all_params, indent=2))
        self._all_outputs_f.write('\n\n')

        stdout = result.get('stdout', '')
        stderr = result.get('stderr', '')
        if stdout:
            self._all_outputs_f.write("--- STDOUT ---\n" + stdout + '\n\n')
        if stderr:
            self._all_outputs_f.write("--- STDERR ---\n" + stderr + '\n\n')
        self.flush()

    def log_timeout(self, batch_num: int, total_tasks: int):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg = f"Timeout waiting for result from worker. Task batch {batch_num}/{total_tasks}."
        tqdm.tqdm.write(msg, file=sys.stderr)
        self._summary_f.write(f"\n[{timestamp}] {msg}\n")
        self.flush()

    def flush(self):
        self._summary_f.flush()
        self._all_outputs_f.flush()

    def close(self):
        self._summary_f.close()
        self._all_outputs_f.close()

# --- Task Generation Logic ---

def parse_config_path(config_file_abs_path: str, base_dir: str) -> tuple[str | None, str | None, str | None]:
    """Parses a config file path to extract model, embedding, and dataset name."""
    try:
        relative_path = os.path.relpath(config_file_abs_path, base_dir)
        parts = relative_path.split(os.sep)
        if len(parts) < 3:
            return None, None, None
        dataset_short_name = parts[-1].replace(".py", "")
        embedding_type = parts[-2]
        model_name = parts[-3]
        return model_name, embedding_type, dataset_short_name
    except (ValueError, IndexError):
        return None, None, None

def _get_base_parameters(config_module, abs_config_path: str, master_args: argparse.Namespace) -> dict:
    """Aggregates parameters from all sources to form a base configuration for a file."""
    base_params = {}
    file_base_config = getattr(config_module, 'base_config', {})
    
    # Correct base directory for path parsing is the directory containing master_runner.py
    master_runner_dir = os.path.dirname(os.path.abspath(__file__))
    model_from_path, embed_from_path, dataset_from_path = parse_config_path(
        abs_config_path, master_runner_dir
    )
    dataset_name = file_base_config.get('dataset_short_name') or (dataset_from_path if master_args.parse_path else None)

    # 1. Load dataset defaults first, if applicable
    if dataset_name:
        dataset_defaults = DATASET_DEFAULTS.get(dataset_name, {}).copy()
        if not dataset_defaults:
            print(f"  Warning: No defaults found for '{dataset_name}' in dataset_defaults.py.", file=sys.stderr)
        base_params.update(dataset_defaults)

    # 2. Layer file's base_config on top
    base_params.update(file_base_config)

    # 3. Layer path-derived parameters if enabled
    if master_args.parse_path:
        if model_from_path: base_params['model'] = model_from_path
        if embed_from_path: base_params['model_type'] = embed_from_path

    # 4. Inject global CLI arguments from master_runner
    for arg in ['force_refresh_cache', 'smoke_test', 'use_data_cache', 'data_cache_path']:
        if hasattr(master_args, arg):
            base_params[arg] = getattr(master_args, arg)
    
    # 5. Ensure critical required parameters have some value
    # These are required by run_longExp.py, so we need to ensure they exist
    if 'model' not in base_params:
        print(f"  Warning: 'model' not found in config for {abs_config_path}. Using default 'UnknownModel'.", file=sys.stderr)
        base_params['model'] = 'UnknownModel'
    
    if 'data' not in base_params:
        print(f"  Warning: 'data' not found in config for {abs_config_path}. Using default 'custom'.", file=sys.stderr)
        base_params['data'] = 'custom'
    
    # Add some other commonly required parameters if missing
    if 'is_training' not in base_params:
        base_params['is_training'] = 1  # Default to training mode
            
    return base_params

def _finalize_task_parameters(base_params: dict, iter_conf: dict) -> dict:
    """
    Merges base and iteration-specific parameters, then classifies them for different uses.
    Returns a structured dictionary containing parameters for the script, for logging, and for debugging.
    """
    merged_params = base_params.copy()
    merged_params.update(iter_conf)

    # Determine the base name for the ID (e.g., dataset name)
    id_base_name = (
        iter_conf.get('model_id_prefix') or
        base_params.get('model_id_prefix') or
        base_params.get('data_short_name') or
        merged_params.get('dataset_short_name') or
        "experiment"
    )

    # Get components for IDs, ensuring they exist.
    seq_len = merged_params.get('seq_len', 'SL')
    pred_len = merged_params.get('pred_len', 'PL')
    model = merged_params.get('model', 'UnknownModel')
    embed = merged_params.get('model_type', 'UnknownEmbed')

    # --- Parameter Classification ---

    # 1. Metadata for logging and internal use
    meta_params = {
        'log_display_id': f"{model}_{embed}_{id_base_name}_{seq_len}_{pred_len}",
    }

    # 2. Parameters to be passed to run_longExp.py
    # Start with all merged params
    script_params = merged_params.copy()
    # Add the script-facing model_id
    script_params['model_id'] = f"{id_base_name}_{seq_len}_{pred_len}"

    # In theory, unrecognized parameters can be ignored, so this part may not be needed
    # # Remove keys that are for master_runner internal use only, i.e., in the original config files, some parameters are actually internal parameters of master_runner and don't need to be passed to run_longExp.py
    # internal_keys = {'model_id_prefix', 'dataset_short_name', 'data_short_name'}
    # for key in internal_keys:
    #     script_params.pop(key, None)

    # Return a structured dictionary
    return {
        'script_params': script_params,  # Clean params for the subprocess
        'meta_params': meta_params,      # Metadata for logging
        'all_params': merged_params     # All params for full debugging record
    }

def generate_experiment_tasks_from_file(config_file_path: str, master_args: argparse.Namespace) -> list[dict]:
    """Loads a config file and generates a list of all experiment task parameter sets."""
    abs_config_path = os.path.abspath(config_file_path)
    try:
        spec = importlib.util.spec_from_file_location("exp_config_module", abs_config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for config file {abs_config_path}")
        exp_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp_config_module) # type: ignore
    except Exception as e:
        print(f"Error loading config file {abs_config_path}: {e}", file=sys.stderr)
        return []

    specific_iterations = getattr(exp_config_module, 'specific_iterations', [])
    if not specific_iterations:
        print(f"Warning: No 'specific_iterations' found in {abs_config_path}.", file=sys.stderr)
        return []

    base_parameters = _get_base_parameters(exp_config_module, abs_config_path, master_args)
    
    generated_tasks = [_finalize_task_parameters(base_parameters, iter_conf) for iter_conf in specific_iterations]
    return generated_tasks

# --- Subprocess and Worker Logic ---

def _execute_single_task_logic(task_data: dict, gpu_id_for_env=None, max_threads_for_numpy_libs=0, task_timeout=0):
    """
    Core logic to execute a single task via subprocess call.
    Returns a dictionary with execution results.
    """
    command = ['python', '-u', RUN_LONG_EXP_SCRIPT_PATH]
    
    # Use only the script_params, which are pre-cleaned for the subprocess.
    script_params = task_data.get('script_params', {})
    if not script_params:
        return {'success': False, 'status': 'FAILURE', 'stdout': "", 'stderr': "Error: task_data['script_params'] was empty."}

    for key, value in script_params.items():
        if isinstance(value, bool):
            if value: command.append(f"--{key}")
        else:
            command.extend([f"--{key}", str(value)])

    env = os.environ.copy()
    if gpu_id_for_env is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id_for_env)
    if max_threads_for_numpy_libs > 0:
        limit = str(max_threads_for_numpy_libs)
        env.update({'OMP_NUM_THREADS': limit, 'MKL_NUM_THREADS': limit, 'OPENBLAS_NUM_THREADS': limit, 'NUMEXPR_NUM_THREADS': limit})

    try:
        effective_timeout = task_timeout if task_timeout > 0 else None
        result = subprocess.run(
            command, check=True, cwd=PROJECT_ROOT, text=True, 
            capture_output=True, env=env, timeout=effective_timeout
        )
        return {'success': True, 'status': 'SUCCESS', 'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}
    except subprocess.TimeoutExpired as e:
        # Handle subprocess timeout specifically
        stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode(errors='ignore') if e.stdout else "")
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode(errors='ignore') if e.stderr else "")
        timeout_msg = f"Task timed out after {task_timeout} seconds."
        return {'success': False, 'status': 'TIMEOUT', 'stdout': stdout.strip(), 'stderr': f"{timeout_msg}\n{stderr.strip()}".strip()}
    except subprocess.CalledProcessError as e:
        return {'success': False, 'status': 'FAILURE', 'stdout': e.stdout.strip(), 'stderr': e.stderr.strip()}
    except Exception as e:
        return {'success': False, 'status': 'CRASH', 'stdout': "", 'stderr': f"A non-subprocess error occurred: {e}"}

def gpu_worker(task_queue, results_queue, gpu_id, smoke_test_flag, max_threads_numpy, task_timeout):
    """Worker function to be run in a separate process."""
    for task_data in iter(task_queue.get, None):
        exec_result = _execute_single_task_logic(
            task_data, gpu_id, max_threads_numpy, task_timeout
        )
        
        # Merge the execution result with other metadata and put it in the queue
        final_result = {
            **exec_result,
            'task_data': task_data, # Pass the complete structured task data
            'gpu_id': gpu_id,
            'is_smoke_test': smoke_test_flag,
        }
        results_queue.put(final_result)

# --- Main Execution Flow ---

def validate_paths_and_confirm_execution(config_file_args: list[str], auto_confirm: bool = False) -> list[str] | None:
    """Validates config files, confirms with user, and returns valid absolute paths."""
    if not os.path.exists(RUN_LONG_EXP_SCRIPT_PATH):
        print(f"Critical Error: run_longExp.py not found at: {RUN_LONG_EXP_SCRIPT_PATH}", file=sys.stderr)
        sys.exit(1)

    valid_paths, invalid_paths = [], []
    for arg in config_file_args:
        abs_path = os.path.abspath(arg)
        if os.path.exists(abs_path):
            valid_paths.append(abs_path)
        else:
            invalid_paths.append({'original': arg, 'resolved': abs_path})

    print(f"\n--- Configuration File Summary ---\nFound {len(valid_paths)} of {len(config_file_args)} provided files.")
    if invalid_paths:
        print("The following files were NOT found and will be skipped:")
        for detail in invalid_paths:
            print(f"  - '{detail['original']}' (resolved to: '{detail['resolved']}')")
    print("----------------------------------")

    if not valid_paths:
        print("\nNo valid configuration files found. Exiting.")
        return None

    if auto_confirm:
        print(f"\nAuto-confirming execution with {len(valid_paths)} valid file(s).")
        return valid_paths
    
    proceed = input(f"\nProceed with {len(valid_paths)} valid file(s)? (y/n): ").strip().lower()
    if proceed == 'y':
        return valid_paths
    else:
        print("Aborted by user.")
        return None

def generate_all_tasks(valid_config_paths: list[str], master_args: argparse.Namespace) -> list[dict]:
    """Generates a list of all experiment tasks from all valid config files."""
    all_tasks = []
    print("\nGenerating all experiment tasks...")
    for path in valid_config_paths:
        print(f"Processing configuration file: {path}")
        tasks_from_file = generate_experiment_tasks_from_file(path, master_args)
        all_tasks.extend(tasks_from_file)
        print(f"  Generated {len(tasks_from_file)} tasks.")

    # Add a unique ID to each task for precise tracking
    for task in all_tasks:
        if 'meta_params' not in task:
            task['meta_params'] = {}
        task['meta_params']['unique_task_id'] = str(uuid.uuid4())
        
    return all_tasks

def populate_task_queue(queue: multiprocessing.Queue, tasks: list[dict], num_workers: int):
    """Puts all tasks into the queue, followed by a sentinel for each worker."""
    for task in tasks:
        queue.put(task)
    for _ in range(num_workers):
        queue.put(None)

def start_gpu_workers(task_q, results_q, num_gpus, tasks_per_gpu, smoke_flag, max_threads_arg, task_timeout):
    """Creates, starts, and returns a list of worker processes."""
    worker_processes = []
    total_workers = num_gpus * tasks_per_gpu
    print(f"\nStarting {total_workers} worker processes ({tasks_per_gpu} per GPU for {num_gpus} GPU(s))...")
    
    for worker_idx in range(total_workers):
        gpu_id = worker_idx // tasks_per_gpu
        process = multiprocessing.Process(
            target=gpu_worker, 
            args=(task_q, results_q, gpu_id, smoke_flag, max_threads_arg, task_timeout)
        )
        process.start()
        worker_processes.append(process)
        print(f"  Worker {worker_idx} (PID: {process.pid}) started, assigned to physical GPU {gpu_id}.")
    return worker_processes

def collect_results_and_update_progress(all_tasks: list[dict], results_q: multiprocessing.Queue, total_tasks: int, is_smoke_test: bool, log_dir: str, task_timeout: int) -> tuple[int, int]:
    """Collects results, logs them using RunLogger, and updates progress."""
    logger = RunLogger(log_dir)
    success_count, failure_count = 0, 0
    desc = "Smoke Test" if is_smoke_test else "Executing Experiments"
    
    # Use a map of unique IDs to track which tasks have not yet returned a result.
    pending_tasks_map = {task['meta_params']['unique_task_id']: task for task in all_tasks}

    print(f"\nWaiting for tasks to complete ({desc})...")
    print(f"Logs will be saved in: {os.path.abspath(log_dir)}")

    with tqdm.tqdm(total=total_tasks, desc=desc, unit="task", file=sys.stdout) as pbar:
        for _ in range(total_tasks):
            try:
                # The watchdog timeout should be longer than the task timeout to give workers time to report a timeout.
                watchdog_timeout = task_timeout + 120 if task_timeout > 0 else None
                result = results_q.get(timeout=watchdog_timeout)
            except multiprocessing.queues.Empty:
                # This block now handles hung workers, not simple task timeouts.
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                tqdm.tqdm.write(
                    f"\n[{timestamp}] Master watchdog timed out waiting for a result from any worker. "
                    f"Logging {len(pending_tasks_map)} pending tasks as HUNG.", file=sys.stderr
                )
                
                # Log all remaining tasks in the map as HUNG
                for task_id, task_data in pending_tasks_map.items():
                    hung_result = {
                        'success': False, 'status': 'HUNG', 'task_data': task_data, 'gpu_id': 'N/A',
                        'stdout': '',
                        'stderr': 'Task result was not received. Assumed to be HUNG due to a dead worker or master timeout.'
                    }
                    logger.log_task_result(hung_result)
                    failure_count += 1
                    pbar.update(1) # Manually update progress bar for each hung task
                
                break # Exit the collection loop, as we can't expect more results.
            
            # Normal result processing
            logger.log_task_result(result)
            if result['success']:
                success_count += 1
            else:
                failure_count += 1

            # Remove the completed task from the pending map
            task_id = result.get('task_data', {}).get('meta_params', {}).get('unique_task_id')
            if task_id and task_id in pending_tasks_map:
                del pending_tasks_map[task_id]
            
            pbar.set_postfix_str(f"Success: {success_count}, Failed: {failure_count}")
            pbar.update(1)

    logger.close()
    return success_count, failure_count

def join_worker_processes(workers: list[multiprocessing.Process]):
    """Waits for all worker processes to complete."""
    print("\nEnsuring all worker processes have finished...")
    for i, process in enumerate(workers):
        process.join(timeout=60)
        if process.is_alive():
            print(f"Warning: Worker {i} (PID: {process.pid}) did not terminate. Forcing.", file=sys.stderr)
            process.terminate()
            process.join()

def print_execution_summary(total_tasks: int, successes: int, failures: int, is_smoke_test: bool, log_dir: str):
    """Prints the final summary of the execution."""
    title = "Smoke Test Execution Summary" if is_smoke_test else "Task Execution Summary"
    print(f"\n--- {title} ---")
    print(f"Total tasks attempted: {total_tasks}")
    print(f"  Successful: {successes}")
    print(f"  Failed:     {failures}")
    print(f"A detailed log of this run is available in: {os.path.abspath(log_dir)}")

def main():
    """Main entry point of the script."""
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Run experiments based on hierarchical configuration files.")
    parser.add_argument('config_files', nargs='+', help="Paths to experiment configuration files.")
    parser.add_argument('--parse_path', action='store_true', help="Enable parsing model/embedding/dataset from config file path.")
    parser.add_argument('--num_gpus', type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument('--tasks_per_gpu', type=int, default=1, help="Number of parallel tasks per GPU.")
    parser.add_argument('--smoke_test', action='store_true', help="Run in smoke test mode.")
    parser.add_argument('--force_refresh_cache', action='store_true', help="Force refresh data cache.")
    parser.add_argument('--use_data_cache', action='store_true', help="Enable caching of preprocessed data.")
    parser.add_argument('--data_cache_path', type=str, default='./datacache/', help="Path for data cache.")
    parser.add_argument('--max_numpy_threads_per_task', type=int, default=0, help="Max threads for NumPy/etc. per task (0=default).")
    parser.add_argument('--log_dir_base', type=str, default='logs', help="Base directory for run logs.")
    parser.add_argument('--task_timeout', type=int, default=3600, help="Timeout in seconds for each individual task (default: 3600).")
    parser.add_argument('-y', '--yes', action='store_true', help="Automatically confirm execution without prompting for user input.")
    args = parser.parse_args()

    # Setup unique log directory for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir_base, f"master_run_{timestamp}")

    valid_config_paths = validate_paths_and_confirm_execution(args.config_files, args.yes)
    if not valid_config_paths:
        sys.exit(1)

    all_tasks = generate_all_tasks(valid_config_paths, args)
    task_desc = "smoke test tasks" if args.smoke_test else "experiment tasks"
    if not all_tasks:
        print(f"\nNo {task_desc} were generated. Exiting.")
        sys.exit(0)
    
    random.shuffle(all_tasks)
    print(f"\nGenerated and shuffled {len(all_tasks)} total {task_desc}.")

    task_queue = multiprocessing.Queue()
    results_queue = multiprocessing.Queue()
    
    total_workers = args.num_gpus * args.tasks_per_gpu
    populate_task_queue(task_queue, all_tasks, total_workers)
    
    workers = start_gpu_workers(task_queue, results_queue, args.num_gpus, args.tasks_per_gpu, 
                                args.smoke_test, args.max_numpy_threads_per_task, args.task_timeout)

    success, failed = collect_results_and_update_progress(
        all_tasks, results_queue, len(all_tasks), args.smoke_test, log_dir, args.task_timeout
    )

    join_worker_processes(workers)

    print_execution_summary(len(all_tasks), success, failed, args.smoke_test, log_dir)
    print("\nAll specified experiment tasks processed.")

if __name__ == "__main__":
    main() 