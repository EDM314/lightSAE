import itertools

def generate_parameter_combinations(
    tuning_branch_definitions: list[dict],
    current_specific_params_dict: dict
) -> list[dict]:
    """
    Generate all configuration combinations based on tuning parameter branch definition list and specific parameter dictionary.

    Args:
        tuning_branch_definitions (list[dict]):
            A list of dictionaries, each dictionary represents an independent tuning parameter branch.
            Each branch dictionary should contain a 'params' key, whose value is the tuning parameter dictionary for that branch
            (e.g.: {'learning_rate': [0.001], 'optimizer': ['adam']}).
            Branch dictionaries can also contain an optional 'name' key for identification.
            If this list is empty or all branches have empty 'params', combinations will be generated based on specific_params.

        current_specific_params_dict (dict):
            Dictionary containing specific parameters and their candidate values.
            Keys are parameter names, values are candidate lists for the parameters.
            These parameter value lists must have the same length or all be empty, they will be zip combined.
            If this dictionary is empty or all its value lists are empty, the specific parameter part contributes an empty mapping.

    Returns:
        list[dict]: List containing all generated configurations, each configuration is a dictionary.
    """
    all_generated_configs = []

    # 1. Handle specific parameter combinations (this part can be pre-computed once)
    s_keys = list(current_specific_params_dict.keys())
    specific_combinations_as_list = []

    if not current_specific_params_dict or all(not v for v in current_specific_params_dict.values()):
        specific_combinations_as_list = [()] # Ensure one empty tuple if no specific params
    else:
        # Validate consistency of list lengths in specific_params_dict
        s_value_lists_for_check = list(current_specific_params_dict.values())
        # Ensure all lists have items or all are empty for consistent zip behavior
        # (Validation logic is maintained because it's important for specific_params itself)
        all_lists_empty_specific = all(not lst for lst in s_value_lists_for_check)
        all_lists_non_empty_specific = all(lst for lst in s_value_lists_for_check)

        if not (all_lists_empty_specific or all_lists_non_empty_specific):
            raise ValueError(
                "All value lists in current_specific_params_dict must either be all empty or all non-empty."
            )

        if all_lists_non_empty_specific:
            first_list_len = len(s_value_lists_for_check[0])
            if not all(len(lst) == first_list_len for lst in s_value_lists_for_check):
                raise ValueError(
                    "All non-empty value lists in current_specific_params_dict must have the "
                    "same length for one-to-one mapping (zip behavior)."
                )
            specific_combinations_as_list = list(zip(*s_value_lists_for_check))
        else: # all_lists_empty_specific is True
            specific_combinations_as_list = [()] 
        
        if not specific_combinations_as_list and any(s_value_lists_for_check):
             # This case should ideally be caught by previous checks, but as a safeguard for zip behavior on e.g. {'a':[], 'b':[]}
            specific_combinations_as_list = [()]

    # 2. Iterate through each tuning branch definition
    if not tuning_branch_definitions: # If there are no tuning branch definitions
        # Only use specific_params to generate combinations
        # effectively, precomputed_tuning_combinations = [()] and t_keys = []
        for s_value_tuple in specific_combinations_as_list:
            config = {**dict(zip(s_keys, s_value_tuple))}
            all_generated_configs.append(config)
    else:
        for branch_def in tuning_branch_definitions:
            current_tuning_params = branch_def.get('params', {})
            # branch_name = branch_def.get('name', 'unnamed_branch') # Can be used for logging

            t_keys_current_branch = list(current_tuning_params.keys())
            precomputed_current_branch = []

            if not current_tuning_params: # If current branch's params dictionary is empty
                precomputed_current_branch = [()]
                if not t_keys_current_branch:
                    t_keys_current_branch = []
            else:
                # Ensure all values are lists so that itertools.product works
                param_values = [val if isinstance(val, list) else [val] for val in current_tuning_params.values()]
                precomputed_current_branch = list(itertools.product(*param_values))
            
            if not precomputed_current_branch and current_tuning_params:
                # If there are params definitions but product result is empty (e.g., an empty list as value), skip combinations for this branch
                # Or let it generate empty configurations, depending on expected behavior. Currently skipping.
                continue 

            # Generate configurations for current tuning branch and specific parameter combinations
            for t_value_tuple in precomputed_current_branch:
                for s_value_tuple in specific_combinations_as_list:
                    config = {
                        **dict(zip(t_keys_current_branch, t_value_tuple)),
                        **dict(zip(s_keys, s_value_tuple)),
                    }
                    all_generated_configs.append(config)
        
        if not all_generated_configs and tuning_branch_definitions and any(b.get('params') for b in tuning_branch_definitions):
            # If there are branch definitions and at least one branch has parameters, but final combinations are empty
            # (e.g., all branch product results are empty lists), this means no valid combinations were generated.
            # Depending on requirements, an empty list can be returned here, or if at least one branch should produce something, an error can be thrown.
            # Current behavior is to return empty list.
            pass
        elif not all_generated_configs and (not tuning_branch_definitions or all(not b.get('params') for b in tuning_branch_definitions)) and specific_combinations_as_list == [()]:
            # Case: No tuning params, no specific params. Should produce [{}] for one empty config.
            # Only if specific_combinations_as_list was truly from no specific_params or empty specific_params
            if not current_specific_params_dict or all(not v for v in current_specific_params_dict.values()):
                 if not tuning_branch_definitions or all(not b.get('params') for b in tuning_branch_definitions):
                    all_generated_configs = [{}] # One empty configuration


    return all_generated_configs

# --- Use new function to generate configurations ---
if __name__ == "__main__":
    # Common specific parameter definitions
    specific_params = {
        'pred_len': [96, 192, 336, 720],
        'hidden_dim': [128, 256, 512, 1024]
    }

    # Define tuning parameter dictionary lists for different condition branches
    tuning_definitions = [
        {
            'name': 'learned_router', # Optional branch name for logging or debugging
            'params': {
                'learning_rate': [0.005, 0.001, 0.0005, 0.0001],
                'moe_router_type': ['learned']
            }
        },
        {
            'name': 'mlp_seq_router',
            'params': {
                'learning_rate': [0.005, 0.001],
                'moe_router_type': ['mlp_seq']
            }
        }
        # Can add more branches, e.g., a branch without tuning parameters that only relies on specific parameters
        # {
        #     'name': 'specific_only_test',
        #     'params': {} # Empty tuning parameters
        # }
        # Or completely not provide tuning_branch_definitions (pass empty list or None)
        # to only generate combinations of specific parameters
    ]

    # Generate all configurations in one call
    all_configs = generate_parameter_combinations(tuning_definitions, specific_params)
    print(all_configs)
    # Print results for verification
    # for i, config in enumerate(all_configs):
    #     print(f"Config {i+1}: {config}")

    # print(f"\nTotal configurations generated: {len(all_configs)}")
    # # Expected output (same as before):
    # # Branch 1 ('learned'): 4 (LRs) * 1 (moe_type) = 4 tuning combos
    # # Branch 2 ('mlp_seq'): 2 (LRs) * 1 (moe_type) = 2 tuning combos
    # # Total tuning_combinations = 4 + 2 = 6
    # # Total configurations = 6 (tuning) * 4 (specific_params combinations) = 24.

    # print("\n--- Test: No tuning params, only specific ---")
    # configs_no_tuning = generate_parameter_combinations([], specific_params)
    # for i, config in enumerate(configs_no_tuning):
    #     print(f"Config {i+1}: {config}")
    # print(f"Total: {len(configs_no_tuning)}") # Expected: 4

    # print("\n--- Test: No specific params, only tuning ---")
    # configs_no_specific = generate_parameter_combinations(tuning_definitions, {})
    # for i, config in enumerate(configs_no_specific):
    #     print(f"Config {i+1}: {config}")
    # print(f"Total: {len(configs_no_specific)}") # Expected: 6 (4+2)

    # print("\n--- Test: No tuning, No specific ---")
    # configs_none = generate_parameter_combinations([], {})
    # for i, config in enumerate(configs_none):
    #     print(f"Config {i+1}: {config}")
    # print(f"Total: {len(configs_none)}") # Expected: 1 (containing an empty dict {})

    # print("\n--- Test: Tuning with one branch having empty params ---")
    # tuning_with_empty_branch = [
    #     {'name': 'actual_branch', 'params': {'lr': [0.1, 0.2]}},
    #     {'name': 'empty_params_branch', 'params': {}} # This branch will contribute one empty tuning set
    # ]
    # configs_empty_branch = generate_parameter_combinations(tuning_with_empty_branch, specific_params)
    # for i, config in enumerate(configs_empty_branch):
    #     print(f"Config {i+1}: {config}")
    # # Expected: (2 from actual_branch + 1 from empty_params_branch) * 4 specific = 3 * 4 = 12
    # print(f"Total: {len(configs_empty_branch)}") 