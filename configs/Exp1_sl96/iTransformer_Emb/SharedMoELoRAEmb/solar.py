from configs.utils import generate_parameter_combinations

base_config = {
    "is_training": 1,
    "patience": 3,
    "itr": 1,
    "seq_len": 96,
    "clip_grad_norm": 1.0,
    "use_softmax": 1,
    "grouped_bias": 0,
    "revin": 0,
    "inout_type":"in"
}

current_tuning_params = {
    'learning_rate': [0.0001,0.0005],
    'moe_router_type': ['learned'],
    'rank': [25],
    'num_experts': [10],
    'pred_len': [96,192,336,720],
}

current_specific_params = {
    'e_layers': [2],
    'd_model': [512],
    'd_ff': [512],
    'batch_size': [32]
}


tuning_definitions = [
    {'name': 'main_tuning', 'params': current_tuning_params}
]

specific_iterations = generate_parameter_combinations(tuning_definitions, current_specific_params)
