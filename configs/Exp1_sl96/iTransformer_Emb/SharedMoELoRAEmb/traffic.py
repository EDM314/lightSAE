from configs.utils import generate_parameter_combinations

base_config = {
    "is_training": 1,
    "patience": 3,
    "itr": 1,
    "seq_len": 96,
    "clip_grad_norm": 1.0,
    "use_softmax": 1,
    "grouped_bias": 1,
    "revin": 1
}

current_tuning_params = {
    'learning_rate': [0.001],
    'moe_router_type': ['learned'],
    'rank': [25],
    'num_experts': [10],
    'pred_len': [96,192,336,720]
}

current_specific_params = {

    'e_layers': [4],
    'd_model': [512],
    'd_ff': [512],
    'batch_size': [16]
}


tuning_definitions = [
    {'name': 'main_tuning', 'params': current_tuning_params}
]

specific_iterations = generate_parameter_combinations(tuning_definitions, current_specific_params)
