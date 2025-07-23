from configs.utils import generate_parameter_combinations

base_config = {
    "is_training": 1,
    "patience": 3,
    "itr": 1,
    "seq_len": 96,
    "clip_grad_norm": 1.0,
    "revin": 0
}

current_tuning_params = {
    'learning_rate': [0.0005],
}

current_specific_params = {
    'pred_len': [96],
    'e_layers': [3],
    'd_model': [512],
    'd_ff': [512],
    'batch_size': [32]
}


tuning_definitions = [
    {'name': 'main_tuning', 'params': current_tuning_params}
]

specific_iterations = generate_parameter_combinations(tuning_definitions, current_specific_params)
