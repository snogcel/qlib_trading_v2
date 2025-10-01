"""
Pipeline Manager - Refactored training pipeline logic
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import qlib
from qlib.constant import REG_US
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP

# Import your existing modules
from src.data.nested_data_loader import CustomNestedDataLoader
from src.models.multi_quantile import MultiQuantileModel
from src.data.crypto_loader import crypto_dataloader_optimized as crypto_dataloader
from src.data.gdelt_loader import gdelt_dataloader_optimized as gdelt_dataloader

# Import your feature engineering functions
from training_pipeline import (
    check_transform_proc, prob_up_piecewise, kelly_sizing, 
    identify_market_regimes, q50_regime_aware_signals, 
    signal_classification, quantile_loss
)


class PipelineManager:
    """Manages the complete training pipeline."""
    
    def __init__(self, 
                 train_start: str = "2018-08-02",
                 train_end: str = "2023-12-31", 
                 valid_start: str = "2024-01-01",
                 valid_end: str = "2024-09-30",
                 test_start: str = "2024-10-01", 
                 test_end: str = "2025-04-01",
                 provider_uri: str = "/Projects/qlib_trading_v2/qlib_data/CRYPTO_DATA",
                 quantiles: List[float] = None,
                 seed: int = 42):
        
        self.train_start = train_start
        self.train_end = train_end
        self.valid_start = valid_start
        self.valid_end = valid_end
        self.test_start = test_start
        self.test_end = test_end
        self.provider_uri = provider_uri
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.seed = seed
        
        # Initialize qlib
        qlib.init(provider_uri=provider_uri, region=REG_US)
        
        # Model configuration
        self.core_lgbm_params = {
            "objective": "quantile",
            "metric": ["l1", "l2"],
            "boosting_type": "gbdt",
            "device": "cpu",
            "verbose": -1,
            "random_state": 43,
            "early_stopping_rounds": 50,
            "num_boost_round": 2250,
            "seed": seed
        }
        
        self.multi_quantile_params = {
            q: {"max_depth": 25, **self.core_lgbm_params} for q in self.quantiles
        }
    
    def setup_data_loaders(self) -> Dict:
        """Set up data loaders and handlers."""
        
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
                    "end_time": self.test_end,
                    "freq": freq_config["feature"]
                }
            }
        ]
        
        # Crypto data loader
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
        
        # GDELT data loader
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
        
        # Nested data loader
        nested_dl = CustomNestedDataLoader(
            dataloader_l=[crypto_data_loader, gdelt_data_loader], 
            join="left"
        )
        
        # Handler configuration
        handler_config = {
            "instruments": ["BTCUSDT", "GDELT_BTC_FEAT"],
            "start_time": self.train_start,
            "end_time": self.test_end,
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
        
        return dataset_handler_config
    
    def create_task_config(self, dataset_handler_config: Dict) -> Dict:
        """Create task configuration."""
        
        task_config = {
            "model": {
                "class": "MultiQuantileModel",
                "module_path": "src.models.multi_quantile",
                "kwargs": {
                    "quantiles": self.quantiles,
                    "lgb_params": self.multi_quantile_params
                }
            },
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": dataset_handler_config,
                    "segments": {
                        "train": (self.train_start, self.train_end),
                        "valid": (self.valid_start, self.valid_end),
                        "test": (self.test_start, self.test_end),
                    },
                }
            },
            "task": {
                "model": "LGBModel",
                "dataset": "DatasetH",
                "record": [
                    {
                        "class": "SignalRecord",
                        "module_path": "qlib.workflow.record_temp"
                    },
                    {
                        "class": "PortAnaRecord",
                        "module_path": "qlib.workflow.record_temp"
                    }
                ]
            }
        }
        
        return task_config
    
    def train_model(self) -> Tuple[MultiQuantileModel, object, Dict]:
        """Train the multi-quantile model."""
        
        print("🔧 Setting up data loaders...")
        dataset_handler_config = self.setup_data_loaders()
        
        print("📊 Creating task configuration...")
        task_config = self.create_task_config(dataset_handler_config)
        
        print("🏗️  Initializing model and dataset...")
        model = init_instance_by_config(task_config["model"])
        dataset = init_instance_by_config(task_config["dataset"])
        
        print("📈 Preparing data segments...")
        df_train, df_valid, df_test = dataset.prepare(
            segments=["train", "valid", "test"], 
            col_set=["feature", "label"], 
            data_key=DataHandlerLP.DK_L
        )
        
        # Split data
        X_train, y_train = df_train["feature"], df_train["label"]
        X_val, y_val = df_valid["feature"], df_valid["label"]
        
        print("🚀 Training model...")
        model.fit(dataset=dataset)
        
        print("🔮 Generating predictions...")
        preds_train = model.predict(dataset, "train")
        preds_valid = model.predict(dataset, "valid")
        
        # Calculate and display quantile losses
        print("\n📊 Model Performance:")
        for quantile in self.quantiles:
            loss, coverage = quantile_loss(y_val, preds_valid[f"quantile_{quantile}"], quantile)
            print(f"  Q{int(quantile*100):02d} - Loss: {loss:.6f}, Coverage: {coverage:.2%}")
        
        # Generate feature importance
        print("📋 Calculating feature importance...")
        feature_importance = {}
        for quantile in self.quantiles:
            importance = model.models[quantile].get_feature_importance(importance_type='gain')
            feature_importance[f"q{int(quantile*100)}"] = importance
        
        # Combine all data for feature engineering
        print("🔧 Engineering features...")
        y_all = pd.concat([y_train, y_val], axis=0, join='outer', ignore_index=False)
        X_all = pd.concat([X_train, X_val], axis=0, join='outer', ignore_index=False)
        preds_all = pd.concat([preds_train, preds_valid], axis=0, join='outer', ignore_index=False)
        
        # Create comprehensive feature set
        df_all_components = [
            preds_all[f"quantile_{self.quantiles[0]}"].rename("q10"),
            preds_all[f"quantile_{self.quantiles[1]}"].rename("q50"),
            preds_all[f"quantile_{self.quantiles[2]}"].rename("q90"),
            y_all["LABEL0"].rename("truth"),
        ]
        
        # Add all features from X_all
        for col in X_all.columns:
            df_all_components.append(X_all[col])
        
        df_all = pd.concat(df_all_components, axis=1).dropna()
        
        print("⚙️  Applying feature engineering...")
        # Apply your feature engineering pipeline
        df_all["prob_up"] = df_all.apply(prob_up_piecewise, axis=1)
        df_all = df_all.dropna(subset=['prob_up'])
        
        df_all = q50_regime_aware_signals(df_all)
        df_all["signal_tier"] = df_all.apply(signal_classification, axis=1)
        df_all["kelly_position_size"] = df_all.apply(kelly_sizing, axis=1)
        
        # Generate trading signals
        self._generate_trading_signals(df_all)
        
        results = {
            'model': model,
            'dataset': dataset,
            'predictions': {
                'train': preds_train,
                'valid': preds_valid
            },
            'features': df_all,
            'feature_importance': feature_importance,
            'performance': {
                'quantile_losses': {
                    f"q{int(q*100)}": quantile_loss(y_val, preds_valid[f"quantile_{q}"], q)
                    for q in self.quantiles
                }
            }
        }
        
        print("✅ Model training completed!")
        return model, dataset, results
    
    def _generate_trading_signals(self, df_all: pd.DataFrame):
        """Generate trading signals using Q50-centric approach."""
        
        q50 = df_all["q50"]
        economically_significant = df_all['economically_significant']
        
        # Pure Q50 directional logic
        buy_mask = economically_significant & (q50 > 0)
        sell_mask = economically_significant & (q50 < 0)
        
        # Assign side using Q50-centric approach
        df_all["side"] = -1  # default to HOLD
        df_all.loc[buy_mask, "side"] = 1   # LONG when q50 > 0 and tradeable
        df_all.loc[sell_mask, "side"] = 0  # SHORT when q50 < 0 and tradeable
        
        # VARIANCE-ENHANCED signal strength
        tradeable = economically_significant
        df_all['signal_strength'] = np.where(
            tradeable,
            df_all['abs_q50'] * np.minimum(
                df_all['enhanced_info_ratio'] / df_all['effective_info_ratio_threshold'], 2.0
            ),
            0.0
        )
        
        # Position sizing suggestions
        base_position_size = 0.1 / np.maximum(df_all['vol_risk'] * 1000, 0.1)
        df_all['position_size_suggestion'] = np.where(
            tradeable,
            base_position_size.clip(0.01, 0.5),
            0.0
        )
        
        # Print signal summary
        signal_counts = df_all['side'].value_counts()
        total_signals = len(df_all)
        print(f"\n📊 Q50-centric signals generated:")
        for side, count in signal_counts.items():
            side_name = {1: 'LONG', 0: 'SHORT', -1: 'HOLD'}[side]
            print(f"   {side_name}: {count:,} ({count/total_signals*100:.1f}%)")
    
    def save_model(self, model: MultiQuantileModel, path: Path):
        """Save trained model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"💾 Model saved to {path}")
    
    def load_model(self, path: Path) -> MultiQuantileModel:
        """Load trained model."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"📂 Model loaded from {path}")
        return model
    
    def save_features(self, df: pd.DataFrame, path: Path):
        """Save engineered features."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(path)
        print(f"💾 Features saved to {path}")
    
    def export_results(self, results: Dict, output_dir: Path):
        """Export training results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save feature importance
        for quantile_name, importance in results['feature_importance'].items():
            importance.to_csv(output_dir / f"feature_importance_{quantile_name}.csv")
        
        # Save features
        results['features'].to_csv(output_dir / "engineered_features.csv")
        
        # Save performance metrics
        with open(output_dir / "performance_metrics.json", 'w') as f:
            import json
            json.dump(results['performance'], f, indent=2, default=str)
        
        print(f"📁 Results exported to {output_dir}")