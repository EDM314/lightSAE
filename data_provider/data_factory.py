import os
import pickle
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Solar, Dataset_PEMS
from torch.utils.data import DataLoader

data_dict = {
    'ETTh12': Dataset_ETT_hour,
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'PEMS': Dataset_PEMS,
    'Solar': Dataset_Solar
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    freq = args.freq

    # Construct cache file path
    cache_filename = f"{args.data}_{args.data_path.split('/')[-1].split('.')[0]}_{flag}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_ft{args.features}_tar{args.target}_enc{timeenc}_freq{freq}.pkl"
    cache_file_path = os.path.join(args.data_cache_path, cache_filename)

    data_set = None # Initialize data_set to None

    if args.use_data_cache:
        if args.force_refresh_cache: # Directly access, assuming it always exists when use_data_cache is True
            print(f"INFO: force_refresh_cache is True for '{flag}'. Skipping cache load. Data will be regenerated and cache updated.")
            # data_set remains None, which will trigger dataset creation and subsequent cache saving.
        else:
            # use_data_cache is True and force_refresh_cache is False. Attempt to load from cache.
            if os.path.exists(cache_file_path):
                try:
                    with open(cache_file_path, 'rb') as f:
                        loaded_data_from_cache = pickle.load(f)
                    print(f"INFO: Loaded dataset for '{flag}' from cache: {cache_file_path}")
                    data_set = loaded_data_from_cache
                except Exception as e:
                    print(f"WARNING: Failed to load dataset from cache {cache_file_path}. Error: {e}. Rebuilding...")
                    # data_set remains None, will trigger dataset creation and subsequent cache saving.
            else:
                print(f"INFO: Cache file not found for '{flag}': {cache_file_path}. Building dataset...")
                # data_set remains None, will trigger dataset creation and subsequent cache saving.
    # If args.use_data_cache is False, data_set remains None (no cache operations).
    # All scenarios where data_set remains None will correctly lead to the 'if data_set is None:' block.

    if data_set is None: # If not loaded from cache, or cache is disabled, or refresh is forced, or cache load failed.
        original_data_set_creation = True # Flag to indicate dataset was newly created
        if flag == 'pred':
            data_set = Dataset_Pred(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                # Make sure to pass all necessary args for Dataset_Pred
                inverse=args.inverse if hasattr(args, 'inverse') else False, # Example for optional args
                cols=args.cols if hasattr(args, 'cols') else None
            )
        else:
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq
            )

        if args.use_data_cache and original_data_set_creation:
            try:
                if not os.path.exists(args.data_cache_path):
                    os.makedirs(args.data_cache_path)
                with open(cache_file_path, 'wb') as f:
                    pickle.dump(data_set, f)
                print(f"INFO: Saved dataset for '{flag}' to cache: {cache_file_path}")
            except Exception as e:
                print(f"WARNING: Failed to save dataset to cache {cache_file_path}. Error: {e}")
    
    # Common logic for setting up DataLoader
    if flag == 'test':
        shuffle_flag = False
        drop_last = False  # fix bug
        batch_size = args.batch_size
        
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        # Data = Dataset_Pred # This line is not needed here as Data / data_set is already handled
    elif flag == 'val':
        shuffle_flag = True # For validation set, shuffling theoretically doesn't matter, but in iTransformer's experimental code it seems to be set to True
        drop_last = True 
        batch_size = args.batch_size
    elif flag == 'train':
        shuffle_flag = True
        drop_last = True # In TFB, drop_last is set to True here, in ThuTS, drop_last is set to False here (they sometimes set it inconsistently themselves, in iTransformer it's set to True)
        batch_size = args.batch_size
    else:
        raise ValueError(f"Invalid flag: {flag}")

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
