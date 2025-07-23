# configs/Exp1_sl96/dataset_defaults.py

DATASET_DEFAULTS = {
    "etth12":{
        "root_path": "./dataset/",  # Relative to project root where 
        "data_path": "ETTh12.csv", # dataset name
        "data_short_name": "etth12", # dataset short name, used for key in DATASET_DEFAULTS and for building model_id_name
        "data": "ETTh12", # data provider class name
        "features": "M",
        "enc_in": 14,
    },
    "etth1": {
        "root_path": "./dataset/",  # Relative to project root where 
        "data_path": "ETTh1.csv", # dataset name
        "data_short_name": "etth1", # dataset short name, used for key in DATASET_DEFAULTS and for building model_id_name
        "data": "ETTh1", # data provider class name
        "features": "M",
        "enc_in": 7,
        # Add other ETTh1 specific defaults if any
    },
    "etth2": {
        "root_path": "./dataset/",
        "data_path": "ETTh2.csv",
        "data_short_name": "etth2",
        "data": "ETTh2",
        "features": "M",
        "enc_in": 7,
    },
    "ettm1": {
        "root_path": "./dataset/",
        "data_path": "ETTm1.csv",
        "data_short_name": "ettm1",
        "data": "ETTm1",
        "features": "M",
        "enc_in": 7,
    },
    "ettm2": {
        "root_path": "./dataset/",
        "data_path": "ETTm2.csv",
        "data_short_name": "ettm2",
        "data": "ETTm2",
        "features": "M",
        "enc_in": 7,
    },
    "weather": {
        "root_path": "./dataset/",
        "data_path": "weather.csv",
        "data_short_name": "weather",
        "data": "custom",
        "features": "M",
        "enc_in": 21,
    },
    "traffic": {
        "root_path": "./dataset/",
        "data_path": "traffic.csv",
        "data_short_name": "traffic",
        "data": "custom",
        "features": "M",
        "enc_in": 862,
    },
    "electricity": {
        "root_path": "./dataset/",
        "data_path": "electricity.csv",
        "data_short_name": "electricity",
        "data": "custom",
        "features": "M",
        "enc_in": 321,
    },
    "solar": {
        "root_path": "./dataset/",
        "data_path": "solar_AL.txt",
        "data_short_name": "solar",
        "data": "Solar",
        "features": "M",
        "enc_in": 137
    },
    "pems04": {
        "root_path": "./dataset/PEMSnpz/",
        "data_path": "pems04.npz",
        "data_short_name": "pems04",
        "data": "PEMS",
        "features": "M",
        "enc_in": 307,
    },
    # "pems08": {
    #     "root_path": "./dataset/PEMSnpz/",
    #     "data_path": "pems08.npz",
    #     "data_short_name": "pems08",
    #     "data": "PEMS",
    #     "features": "M",
    #     "enc_in": 170,
    # },
    "pems07": {
        "root_path": "./dataset/PEMSnpz/",
        "data_path": "pems07.npz",
        "data_short_name": "pems07",
        "data": "PEMS",
        "features": "M",
        "enc_in": 883,
    },
    # "pems04": {
    #     "root_path": "./dataset/PEMScsv/",
    #     "data_path": "pems04.csv",
    #     "data_short_name": "pems04",
    #     "data": "custom",
    #     "features": "M",
    #     "enc_in": 307,
    # },
    # "pems08": {
    #     "root_path": "./dataset/PEMScsv/",
    #     "data_path": "pems08.csv",
    #     "data_short_name": "pems08",
    #     "data": "custom",
    #     "features": "M",
    #     "enc_in": 170,
    # },
    # Add other datasets as needed
} 