from configs.utils import generate_parameter_combinations

base_config = {
    "is_training": 1,
    "patience": 3,
    "itr": 1,
    "seq_len": 96,
    "num_experts": 10,
    "clip_grad_norm": 1.0
}

current_tuning_params = {
    'learning_rate': [0.01, 0.005, 0.001, 0.0005],
    'moe_router_type': ['learned', 'mlp_seq'],
    'rank': [25]
}

current_specific_params = {
    'pred_len': [96, 192, 336, 720]
}


tuning_definitions = [
    {'name': 'main_tuning', 'params': current_tuning_params}
]

specific_iterations = generate_parameter_combinations(tuning_definitions, current_specific_params)
