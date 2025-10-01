"""
Optuna Hyperparameter Optimizer - Refactored from training_pipeline_optuna.py
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import qlib
from qlib.constant import REG_US
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error as MSE

import optuna
import lightgbm as lgbm

# Import your existing modules
from src.data.nested_data_loader import CustomNestedDataLoader
from src.data.crypto_loader import crypto_dataloader_optimized as crypto_dataloader
from src.data.gdelt_loader import gdelt_dataloader_optimized as gdelt_dataloader

# Import utility functions
from training_pipeline_optuna import check_transform_proc, cross_validation_fcn


class OptunaOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(self,
                 train_start: str = "2018-08-02",
                 train_end: str = "2023-12-31",
                 valid_start: str = "2024-01-01", 
                 valid_end: str = "2024-09-30",
                 provider_uri: str = "/Projects/qlib_trading_v2/qlib_data/CRYPTO_DATA",
                 quantile: float = 0.1,
                 seed: int = 42):
        
        self.train_start = train_start
        self.train_end = train_end
        self.valid_start = valid_start
        self.valid_end = valid_end
        self.provider_uri = provider_uri
        self.quantile = quantile
        self.seed = seed
        
        # Initialize qlib
        qlib.init(provider_uri=provider_uri, region=REG_US)
        
        # Setup data
        self.df_train = None
        self._setup_data()
    
    def _setup_data(self):
        """Setup data loaders and prepare training data."""
        
        # Processor configuration
        _learn_processors = [{"class": "DropnaLabel"}]
        _infer_processors = []
        
        learn_processors = check_transform_proc(_learn_processors, None, None)
        infer_processors = check_transform_proc(_infer_processors, None, None)
        
        freq_config = {
            "feature": "60min",
            "label": "day"
        }
        
        inst_processors = [
            {
                "class": "TimeRangeFlt",
                "module_path": "qlib.data.dataset.processor",
                "kwargs": {
                    "start_time": self.train_start,
                    "end_time": self.valid_end,  # Include validation for optimization
                    "freq": freq_config["feature"]
                }
            }
        ]
        
        # Data loaders
        crypto_data_loader = {
            "class": "crypto_dataloader_optimized",
            "module_path": "src.data.crypto_loader",
            "kwargs": {
                "config": {
                    "feature": crypto_dataloader.get_feature_config(),
                    "label": crypto_dataloader.get_label_config(),
                },
                "freq": freq_config["feature"],
                "inst_processors": inst_processors
            }
        }
        
        gdelt_data_loader = {
            "class": "gdelt_dataloader_optimized",
            "module_path": "src.data.gdelt_loader",
            "kwargs": {
                "config": {
                    "feature": gdelt_dataloader.get_feature_config()
                },
                "freq": freq_config["label"],
                "inst_processors": inst_processors
            }
        }
        
        nested_dl = CustomNestedDataLoader(
            dataloader_l=[crypto_data_loader, gdelt_data_loader], 
            join="left"
        )
        
        # Handler configuration
        handler_config = {
            "instruments": ["BTCUSDT", "GDELT_BTC_FEAT"],
            "start_time": self.train_start,
            "end_time": self.valid_end,
            "data_loader": nested_dl,
            "infer_processors": infer_processors,
            "learn_processors": learn_processors,
            "shared_processors": [],
            "process_type": DataHandlerLP.PTYPE_A,
            "drop_raw": False
        }
        
        dataset_handler_config = {
            "class": "DataHandlerLP",
            "module_path": "qlib.data.dataset.handler",
            "kwargs": handler_config,
        }
        
        # Dataset configuration
        dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": dataset_handler_config,
                "segments": {
                    "train": (self.train_start, self.train_end),
                    "valid": (self.valid_start, self.valid_end),
                },
            }
        }
        
        # Initialize dataset
        dataset = init_instance_by_config(dataset_config)
        
        # Prepare training data
        df_train, df_valid = dataset.prepare(
            segments=["train", "valid"], 
            col_set=["feature", "label"], 
            data_key=DataHandlerLP.DK_L
        )
        
        # Combine for cross-validation
        self.df_train = {
            "feature": pd.concat([df_train["feature"], df_valid["feature"]], axis=0),
            "label": pd.concat([df_train["label"], df_valid["label"]], axis=0)
        }
        
        print(f"📊 Optimization data prepared: {len(self.df_train['feature'])} samples")
    
    def objective(self, trial):
        """Optuna objective function."""
        
        # Suggest hyperparameters
        params = {
            "objective": "quantile",
            "metric": ["l2", "l1"],
            "boosting_type": "gbdt",
            "device": "cpu",
            "verbose": -1,
            "alpha": self.quantile,
            
            # Hyperparameters to optimize
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            "max_depth": trial.suggest_int("max_depth", 4, 25),
            "num_leaves": trial.suggest_int("num_leaves", 20, 512),
            
            # Fixed parameters
            "min_data_in_leaf": 20,
            "early_stopping_rounds": 100,
            "num_boost_round": 1000,
            "seed": self.seed
        }
        
        # Create model
        model = lgbm.LGBMRegressor(**params)
        
        # Perform cross-validation
        lgbm_model, mean_score = cross_validation_fcn(
            self.df_train, 
            model, 
            early_stopping_flag=True
        )
        
        # Store best iteration
        if hasattr(lgbm_model, 'best_iteration_'):
            trial.set_user_attr('best_iteration', lgbm_model.best_iteration_)
        
        return mean_score
    
    def optimize(self, 
                 n_trials: int = 100,
                 timeout: Optional[int] = None,
                 study_name: Optional[str] = None,
                 storage: Optional[str] = None) -> optuna.Study:
        """Run hyperparameter optimization."""
        
        print(f"🔧 Starting Optuna optimization for quantile {self.quantile}")
        print(f"   Trials: {n_trials}")
        print(f"   Timeout: {timeout}s" if timeout else "   No timeout")
        
        # Create study
        study_name = study_name or f"quantile_{self.quantile}_optimization"
        
        if storage:
            study = optuna.create_study(
                direction="minimize",
                study_name=study_name,
                storage=storage,
                load_if_exists=True
            )
        else:
            study = optuna.create_study(direction="minimize")
        
        # Add pruner for early stopping of unpromising trials
        study.sampler = optuna.samplers.TPESampler(seed=self.seed)
        study.pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
        
        # Optimize
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[self._log_callback]
        )
        
        # Print results
        print(f"\n✅ Optimization completed!")
        print(f"   Best score: {study.best_value:.6f}")
        print(f"   Best parameters:")
        for key, value in study.best_params.items():
            print(f"     {key}: {value}")
        
        return study
    
    def _log_callback(self, study, trial):
        """Callback to log optimization progress."""
        if trial.number % 10 == 0:
            print(f"   Trial {trial.number}: {trial.value:.6f}")
    
    def get_best_params(self, study: optuna.Study) -> Dict:
        """Get best parameters from study."""
        return study.best_params
    
    def save_study(self, study: optuna.Study, path: Path):
        """Save study results."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save best parameters
        import json
        with open(path, 'w') as f:
            json.dump({
                'best_params': study.best_params,
                'best_value': study.best_value,
                'n_trials': len(study.trials),
                'quantile': self.quantile
            }, f, indent=2)
        
        print(f"💾 Study results saved to {path}")
    
    def create_optimized_model_config(self, study: optuna.Study) -> Dict:
        """Create model configuration with optimized parameters."""
        
        best_params = study.best_params.copy()
        
        # Add fixed parameters
        best_params.update({
            "objective": "quantile",
            "metric": ["l1", "l2"],
            "boosting_type": "gbdt",
            "device": "cpu",
            "verbose": -1,
            "alpha": self.quantile,
            "min_data_in_leaf": 20,
            "early_stopping_rounds": 50,
            "num_boost_round": 2250,
            "seed": self.seed
        })
        
        return best_params