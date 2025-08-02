"""
Save your trained quantile model for production use
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import qlib
from qlib.constant import REG_US
from qlib_custom.custom_multi_quantile import MultiQuantileModel
from qlib_custom.custom_ndl import CustomNestedDataLoader
from qlib_custom.crypto_loader_optimized import crypto_dataloader_optimized
from qlib_custom.gdelt_loader_optimized import gdelt_dataloader_optimized
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import DatasetH
from qlib.utils import init_instance_by_config

def save_trained_model():
    """
    Save your trained model from ppo_sweep_optuna_tuned.py for production use
    """
    
    # Initialize qlib (same as in your training script)
    provider_uri = "/Projects/qlib_trading_v2/qlib_data/CRYPTO_DATA"
    qlib.init(provider_uri=provider_uri, region=REG_US)
    
    # Model configuration (from your training script)
    SEED = 42
    train_start_time = "2018-08-02"
    train_end_time = "2023-12-31"
    valid_start_time = "2024-01-01"
    valid_end_time = "2024-09-30"
    test_start_time = "2024-10-01"
    test_end_time = "2025-04-01"
    
    # Your optimized parameters
    """ lgb_params = {
        0.1: {'learning_rate': 0.03915802868187673, 'colsample_bytree': 0.6224232548522113, 
              'subsample': 0.7322459139253197, 'lambda_l1': 6.957072141326349, 
              'lambda_l2': 0.004366116342801104, 'max_depth': 10, 'seed': SEED},
        0.5: {'learning_rate': 0.08751145729062904, 'colsample_bytree': 0.5897687601362188, 
              'subsample': 0.754061620932527, 'lambda_l1': 1.9808527398597983e-06, 
              'lambda_l2': 2.91987558633637e-05, 'max_depth': 10, 'seed': SEED},
        0.9: {'learning_rate': 0.028047164919345058, 'colsample_bytree': 0.841009708338563, 
              'subsample': 0.6210307287531586, 'lambda_l1': 2.9139063969227813e-08, 
              'lambda_l2': 6.363456739796053, 'max_depth': 10, 'seed': SEED}
    } """

    GENERIC_LGBM_PARAMS = {
        # Core quantile settings
        "objective": "quantile",
        "metric": ["l1", "l2"], # , "l2", "l1" # "rmse"
        "boosting_type": "gbdt",
        "device": "cpu",
        "verbose": -1,
        "random_state": 42,
        
        # Conservative learning settings for feature exploration
        # "learning_rate": 0.05,           # Moderate learning rate
        # "num_leaves": 64,                # Balanced complexity
        # "max_depth": 8,                  # Reasonable depth for GDELT features
        
        # Regularization (moderate to prevent overfitting)
        # "lambda_l1": 0.1,
        # "lambda_l2": 0.1,
        # "min_data_in_leaf": 20,
        # "feature_fraction": 0.8,         # Use 80% of features per tree
        # "bagging_fraction": 0.8,         # Use 80% of data per iteration
        # "bagging_freq": 5,
        
        # Early stopping
        "early_stopping_rounds": 50,
        "num_boost_round": 2250,         # Let early stopping decide

        # Set seed for reproducibility
        "seed": SEED
    }

    multi_quantile_params = {
        # 0.1: {'learning_rate': 0.060555113429817814, 'colsample_bytree': 0.7214813020361056, 'subsample': 0.7849919729082881, 'lambda_l1': 8.722794281828277e-05, 'lambda_l2': 3.220667556916701e-05, 'max_depth': 10, 'num_leaves': 224, **GENERIC_LGBM_PARAMS},
        # 0.5: {'learning_rate': 0.02753370821225369, 'max_depth': -1, 'lambda_l1': 0.1, 'lambda_l2': 0.1, **GENERIC_LGBM_PARAMS},
        # 0.9: {'learning_rate': 0.09355380738420341, 'max_depth': 10, 'num_leaves': 249, 'lambda_l1': 0.1, 'lambda_l2': 0.1, **GENERIC_LGBM_PARAMS}

        0.1: {'learning_rate': 0.026, 'max_depth': 7, **GENERIC_LGBM_PARAMS},
        0.5: {'learning_rate': 0.027, 'max_depth': 7, **GENERIC_LGBM_PARAMS},                
        0.9: {'learning_rate': 0.028, 'max_depth': 7, **GENERIC_LGBM_PARAMS} 
    }
    
    # Create model
    model = MultiQuantileModel(
        quantiles=[0.1, 0.5, 0.9],
        lgb_params=multi_quantile_params
    )
    
    # Setup data pipeline (same as training)
    freq_config = {"feature": "60min", "label": "day"}
    
    inst_processors = [{
        "class": "TimeRangeFlt",
        "module_path": "qlib.data.dataset.processor",
        "kwargs": {
            "start_time": train_start_time,
            "end_time": test_end_time,
            "freq": freq_config["feature"]
        }
    }]
    
    crypto_data_loader = {
        "class": "crypto_dataloader_optimized",
        "module_path": "qlib_custom.crypto_loader_optimized",
        "kwargs": {
            "config": {
                "feature": crypto_dataloader_optimized.get_feature_config(),
                "label": crypto_dataloader_optimized.get_label_config(),
            },
            "freq": freq_config["feature"],
            "inst_processors": inst_processors
        }
    }
    
    gdelt_data_loader = {
        "class": "gdelt_dataloader_optimized", 
        "module_path": "qlib_custom.gdelt_loader_optimized",
        "kwargs": {
            "config": {
                "feature": gdelt_dataloader_optimized.get_feature_config()
            },
            "freq": freq_config["label"],
            "inst_processors": inst_processors
        }
    }
    
    nested_dl = CustomNestedDataLoader(dataloader_l=[crypto_data_loader, gdelt_data_loader], join="left")
    
    handler_config = {
        "instruments": ["BTCUSDT", "GDELT_BTC_FEAT"],
        "start_time": train_start_time,
        "end_time": test_end_time,
        "data_loader": nested_dl,
        "infer_processors": [],
        "learn_processors": [{"class": "DropnaLabel"}],
        "shared_processors": [],
        "process_type": DataHandlerLP.PTYPE_A,
        "drop_raw": False
    }
    
    dataset_handler_config = {
        "class": "DataHandlerLP",
        "module_path": "qlib.data.dataset.handler",
        "kwargs": handler_config,
    }
    
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": dataset_handler_config,
            "segments": {
                "train": (train_start_time, train_end_time),
                "valid": (valid_start_time, valid_end_time),
                "test": (test_start_time, test_end_time),
            },
        }
    }
    
    # Initialize dataset
    dataset = init_instance_by_config(dataset_config)
    
    # Train model
    print("Training model...")
    model.fit(dataset=dataset)
    
    # Save model
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "trained_quantile_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")
    
    # Save model metadata
    metadata = {
        "quantiles": [0.1, 0.5, 0.9],
        "training_period": f"{train_start_time} to {train_end_time}",
        "validation_period": f"{valid_start_time} to {valid_end_time}",
        "features": crypto_dataloader_optimized.get_feature_config()[1],  # feature names
        "lgb_params": multi_quantile_params,
        "seed": SEED
    }
    
    metadata_path = models_dir / "model_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model metadata saved to {metadata_path}")
    
    # Test prediction
    print("Testing prediction...")
    test_pred = model.predict(dataset, "valid")
    print("Sample predictions:")
    print(test_pred.head())
    
    return model_path, metadata_path

def create_feature_pipeline_config():
    """
    Create a configuration file for the feature pipeline
    """
    config = {
        "data_source": {
            "provider_uri": "/Projects/qlib_trading_v2/qlib_data/CRYPTO_DATA",
            "instruments": ["BTCUSDT", "GDELT_BTC_FEAT"],
            "freq_config": {
                "feature": "60min",
                "label": "day"
            }
        },
        "features": {
            "crypto": crypto_dataloader_optimized.get_feature_config()[1],
            "gdelt": gdelt_dataloader_optimized.get_feature_config()[1]
        },
        "thresholds": {
            "signal_thresh_percentile": 0.85,
            "spread_thresh_percentile": 0.85,
            "rolling_window": 30,
            "min_window": 10
        }
    }
    
    config_path = Path("./config/feature_pipeline.json")
    config_path.parent.mkdir(exist_ok=True)
    
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Feature pipeline config saved to {config_path}")
    return config_path

if __name__ == "__main__":
    print("=== Saving Model for Production ===")
    
    try:
        model_path, metadata_path = save_trained_model()
        config_path = create_feature_pipeline_config()
        
        print("\n=== Success ===")
        print(f"Model: {model_path}")
        print(f"Metadata: {metadata_path}")
        print(f"Config: {config_path}")
        print("\nYou can now use these files with realtime_predictor.py")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        import traceback
        traceback.print_exc()