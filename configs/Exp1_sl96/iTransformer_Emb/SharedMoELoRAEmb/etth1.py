from configs.utils import generate_parameter_combinations

base_config = {
    "is_training": 1,
    "patience": 3,
    "itr": 1,
    "seq_len": 96,
    "clip_grad_norm": 1.0
}

current_tuning_params = {
    'learning_rate': [0.005, 0.001, 0.0005, 0.0001],
    'moe_router_type': ['learned', 'mlp_seq'],
    'rank': [25],
    'num_experts': [3]
}

current_specific_params = {
    'pred_len': [96, 192, 336, 720],
    'e_layers': [2, 2, 2, 2],
    'd_model': [256, 256, 512, 512],
    'd_ff': [256, 256, 512, 512],
    'batch_size': [32, 32, 32, 32]
}


tuning_definitions = [
    {'name': 'main_tuning', 'params': current_tuning_params}
]

specific_iterations = generate_parameter_combinations(tuning_definitions, current_specific_params)
