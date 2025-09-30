"""
Comprehensive Model Evaluation Suite for Quantile Models
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class QuantileModelEvaluator:
    """
    Comprehensive evaluation framework for multi-quantile models
    """
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        self.quantiles = quantiles
        self.results = {}
        
    def quantile_loss(self, y_true: pd.Series, y_pred: pd.Series, quantile: float) -> Tuple[float, float]:
        """
        Calculate quantile loss and empirical coverage
        
        Returns:
            loss: Quantile loss value
            coverage: Empirical coverage percentage
        """
        # Ensure both are Series with matching structure
        if isinstance(y_pred, pd.DataFrame) and y_pred.shape[1] == 1:
            y_pred = y_pred.iloc[:, 0]
        
        if isinstance(y_true, pd.DataFrame) and y_true.shape[1] == 1:
            y_true = y_true.iloc[:, 0]
        
        # Align index names and values
        y_pred.index.names = y_true.index.names
        y_true_aligned, y_pred_aligned = y_true.align(y_pred, join='inner')
        
        # Calculate quantile loss
        errors = y_true_aligned - y_pred_aligned
        loss = np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))
        
        # Calculate empirical coverage
        coverage = (y_true_aligned < y_pred_aligned).mean()
        
        return loss, coverage
    
    def evaluate_quantile_performance(self, y_true: pd.Series, predictions: pd.DataFrame) -> Dict:
        """
        Evaluate performance across all quantiles
        """
        results = {}
        
        for q in self.quantiles:
            col_name = f"quantile_{q:.2f}"
            if col_name in predictions.columns:
                loss, coverage = self.quantile_loss(y_true, predictions[col_name], q)
                
                results[q] = {
                    'loss': loss,
                    'coverage': coverage,
                    'target_coverage': q if q <= 0.5 else 1 - q,
                    'coverage_error': abs(coverage - (q if q <= 0.5 else 1 - q))
                }
        
        return results
    
    def prediction_interval_analysis(self, y_true: pd.Series, predictions: pd.DataFrame, 
                                   instrument: str = "BTCUSDT") -> Dict:
        """
        Analyze prediction intervals (e.g., Q10-Q90 band)
        """
        # Filter for specific instrument if multi-index
        if isinstance(y_true.index, pd.MultiIndex):
            y_true_inst = y_true.xs(instrument, level="instrument")
            pred_inst = predictions.xs(instrument, level="instrument")
        else:
            y_true_inst = y_true
            pred_inst = predictions
        
        q10 = pred_inst["quantile_0.10"]
        q50 = pred_inst["quantile_0.50"] 
        q90 = pred_inst["quantile_0.90"]
        
        # Interval coverage (should be ~80% for Q10-Q90)
        in_interval = (y_true_inst >= q10) & (y_true_inst <= q90)
        interval_coverage = in_interval.mean()
        
        # Interval width statistics
        interval_width = q90 - q10
        
        # Asymmetry analysis
        upper_width = q90 - q50
        lower_width = q50 - q10
        asymmetry = (upper_width - lower_width) / (upper_width + lower_width + 1e-12)
        
        # Sharpness (narrower intervals are better, all else equal)
        mean_width = interval_width.mean()
        median_width = interval_width.median()
        
        return {
            'interval_coverage': interval_coverage,
            'target_coverage': 0.8,  # Q10-Q90 should cover 80%
            'coverage_error': abs(interval_coverage - 0.8),
            'mean_width': mean_width,
            'median_width': median_width,
            'width_std': interval_width.std(),
            'asymmetry_mean': asymmetry.mean(),
            'asymmetry_std': asymmetry.std()
        }
    
    def feature_importance_analysis(self, model, save_path: Optional[Path] = None) -> Dict:
        """
        Analyze and compare feature importance across quantiles
        """
        importance_data = {}
        
        for q in self.quantiles:
            if q in model.models:
                lgb_model = model.models[q]
                importance = lgb_model.get_feature_importance(importance_type='gain')
                feature_names = lgb_model.model.feature_name()
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                importance_data[q] = importance_df
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame()
        for q, df in importance_data.items():
            comparison_df[f'Q{int(q*100)}'] = df.set_index('feature')['importance']
        
        comparison_df = comparison_df.fillna(0)
        
        # Calculate feature stability across quantiles
        feature_stability = comparison_df.std(axis=1) / (comparison_df.mean(axis=1) + 1e-12)
        feature_stability = feature_stability.sort_values()
        
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(exist_ok=True)
            
            # Save individual importance files
            for q, df in importance_data.items():
                df.to_csv(save_path / f"feature_importance_Q{int(q*100)}.csv", index=False)
            
            # Save comparison
            comparison_df.to_csv(save_path / "feature_importance_comparison.csv")
            feature_stability.to_csv(save_path / "feature_stability.csv")
        
        return {
            'individual': importance_data,
            'comparison': comparison_df,
            'stability': feature_stability
        }
    
    def directional_accuracy(self, y_true: pd.Series, predictions: pd.DataFrame) -> Dict:
        """
        Evaluate directional prediction accuracy
        """
        q50 = predictions["quantile_0.50"]
        
        # Align data
        y_true_aligned, q50_aligned = y_true.align(q50, join='inner')
        
        # Calculate directional accuracy
        true_direction = np.sign(y_true_aligned)
        pred_direction = np.sign(q50_aligned)
        
        directional_accuracy = (true_direction == pred_direction).mean()
        
        # Accuracy by direction
        up_mask = true_direction > 0
        down_mask = true_direction < 0
        
        up_accuracy = (true_direction[up_mask] == pred_direction[up_mask]).mean() if up_mask.sum() > 0 else 0
        down_accuracy = (true_direction[down_mask] == pred_direction[down_mask]).mean() if down_mask.sum() > 0 else 0
        
        return {
            'overall_accuracy': directional_accuracy,
            'up_accuracy': up_accuracy,
            'down_accuracy': down_accuracy,
            'up_samples': up_mask.sum(),
            'down_samples': down_mask.sum()
        }
    
    def regime_analysis(self, y_true: pd.Series, predictions: pd.DataFrame, 
                       volatility: pd.Series) -> Dict:
        """
        Analyze model performance across volatility regimes
        """
        # Define volatility regimes
        vol_quantiles = volatility.quantile([0.33, 0.67])
        
        low_vol_mask = volatility <= vol_quantiles.iloc[0]
        med_vol_mask = (volatility > vol_quantiles.iloc[0]) & (volatility <= vol_quantiles.iloc[1])
        high_vol_mask = volatility > vol_quantiles.iloc[1]
        
        regimes = {
            'low_vol': low_vol_mask,
            'med_vol': med_vol_mask,
            'high_vol': high_vol_mask
        }
        
        regime_results = {}
        
        for regime_name, mask in regimes.items():
            if mask.sum() > 10:  # Minimum samples
                regime_true = y_true[mask]
                regime_pred = predictions[mask]
                
                # Quantile performance in this regime
                quantile_perf = self.evaluate_quantile_performance(regime_true, regime_pred)
                
                # Directional accuracy in this regime
                dir_acc = self.directional_accuracy(regime_true, regime_pred)
                
                regime_results[regime_name] = {
                    'samples': mask.sum(),
                    'quantile_performance': quantile_perf,
                    'directional_accuracy': dir_acc
                }
        
        return regime_results
    
    def create_evaluation_report(self, y_true: pd.Series, predictions: pd.DataFrame,
                               model, volatility: pd.Series = None, 
                               save_path: str = "./evaluation_results") -> Dict:
        """
        Generate comprehensive evaluation report
        """
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        print("🔍 Starting comprehensive model evaluation...")
        
        # 1. Quantile Performance
        print("Evaluating quantile performance...")
        quantile_results = self.evaluate_quantile_performance(y_true, predictions)
        
        # 2. Prediction Intervals
        print("📏 Analyzing prediction intervals...")
        interval_results = self.prediction_interval_analysis(y_true, predictions)
        
        # 3. Feature Importance
        print("🎯 Analyzing feature importance...")
        feature_results = self.feature_importance_analysis(model, save_path / "feature_importance")
        
        # 4. Directional Accuracy
        print("🧭 Evaluating directional accuracy...")
        directional_results = self.directional_accuracy(y_true, predictions)
        
        # 5. Regime Analysis (if volatility provided)
        regime_results = None
        if volatility is not None:
            print("🌊 Analyzing performance across volatility regimes...")
            regime_results = self.regime_analysis(y_true, predictions, volatility)
        
        # Compile results
        full_results = {
            'quantile_performance': quantile_results,
            'prediction_intervals': interval_results,
            'feature_importance': feature_results,
            'directional_accuracy': directional_results,
            'regime_analysis': regime_results
        }
        
        # Save results
        import json
        with open(save_path / "evaluation_summary.json", 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = self._convert_for_json(full_results)
            json.dump(json_results, f, indent=2)
        
        # Generate plots
        self._create_evaluation_plots(y_true, predictions, full_results, save_path)
        
        print(f"Evaluation complete! Results saved to {save_path}")
        return full_results
    
    def _convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types"""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        else:
            return obj
    
    def _create_evaluation_plots(self, y_true: pd.Series, predictions: pd.DataFrame,
                               results: Dict, save_path: Path):
        """Create visualization plots"""
        
        # 1. Quantile Loss and Coverage Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        quantiles = list(results['quantile_performance'].keys())
        losses = [results['quantile_performance'][q]['loss'] for q in quantiles]
        coverages = [results['quantile_performance'][q]['coverage'] for q in quantiles]
        targets = [results['quantile_performance'][q]['target_coverage'] for q in quantiles]
        
        ax1.bar(range(len(quantiles)), losses)
        ax1.set_xlabel('Quantile')
        ax1.set_ylabel('Quantile Loss')
        ax1.set_title('Quantile Loss by Quantile')
        ax1.set_xticks(range(len(quantiles)))
        ax1.set_xticklabels([f'Q{int(q*100)}' for q in quantiles])
        
        ax2.plot(range(len(quantiles)), coverages, 'o-', label='Empirical Coverage')
        ax2.plot(range(len(quantiles)), targets, 's--', label='Target Coverage')
        ax2.set_xlabel('Quantile')
        ax2.set_ylabel('Coverage')
        ax2.set_title('Coverage Analysis')
        ax2.set_xticks(range(len(quantiles)))
        ax2.set_xticklabels([f'Q{int(q*100)}' for q in quantiles])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / "quantile_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature Importance Comparison
        if 'comparison' in results['feature_importance']:
            importance_df = results['feature_importance']['comparison']
            
            plt.figure(figsize=(12, 8))
            top_features = importance_df.sum(axis=1).nlargest(20)
            importance_subset = importance_df.loc[top_features.index]
            
            sns.heatmap(importance_subset.T, annot=True, fmt='.0f', cmap='viridis')
            plt.title('Feature Importance Across Quantiles (Top 20 Features)')
            plt.xlabel('Features')
            plt.ylabel('Quantiles')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(save_path / "feature_importance_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Prediction Interval Visualization (sample)
        if isinstance(y_true.index, pd.MultiIndex):
            instrument = "BTCUSDT"
            y_sample = y_true.xs(instrument, level="instrument").tail(100)
            pred_sample = predictions.xs(instrument, level="instrument").tail(100)
        else:
            y_sample = y_true.tail(100)
            pred_sample = predictions.tail(100)
        
        plt.figure(figsize=(15, 8))
        
        x = range(len(y_sample))
        plt.plot(x, y_sample.values, 'k-', label='True Values', linewidth=1.5)
        plt.plot(x, pred_sample["quantile_0.50"].values, 'b--', label='Q50 (Median)', linewidth=1)
        
        plt.fill_between(x, 
                        pred_sample["quantile_0.10"].values,
                        pred_sample["quantile_0.90"].values,
                        alpha=0.3, color='skyblue', label='Q10-Q90 Interval')
        
        plt.xlabel('Time Steps')
        plt.ylabel('Returns')
        plt.title('Prediction Intervals vs True Values (Last 100 Observations)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / "prediction_intervals.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Plots saved to {save_path}")


def run_comprehensive_evaluation():
    """
    Run evaluation on your existing model and data
    """
    # This assumes you have the data from your training script
    try:
        # Load your data (adjust paths as needed)
        df_all = pd.read_csv("df_all_macro_analysis.csv", index_col=[0, 1])
        
        # Split into validation set for evaluation
        df_val = df_all.loc[("BTCUSDT", "2024-01-01"):("BTCUSDT", "2024-09-30")]
        
        # Extract true values and predictions
        y_true = df_val["truth"]
        predictions = df_val[["q10", "q50", "q90"]].rename(columns={
            "q10": "quantile_0.10",
            "q50": "quantile_0.50", 
            "q90": "quantile_0.90"
        })
        
        volatility = df_val["$realized_vol_10"]
        
        # Load your trained model (you'll need to save it first)
        import pickle
        with open("./models/trained_quantile_model.pkl", 'rb') as f:
            model = pickle.load(f)
        
        # Run evaluation
        evaluator = QuantileModelEvaluator()
        results = evaluator.create_evaluation_report(
            y_true=y_true,
            predictions=predictions,
            model=model,
            volatility=volatility,
            save_path="./evaluation_results"
        )
        
        # Print summary
        print("\n" + "="*50)
        print("📋 EVALUATION SUMMARY")
        print("="*50)
        
        print("\n🎯 Quantile Performance:")
        for q, perf in results['quantile_performance'].items():
            print(f"  Q{int(q*100):2d}: Loss={perf['loss']:.4f}, Coverage={perf['coverage']:.1%} (target: {perf['target_coverage']:.1%})")
        
        print(f"\n📏 Prediction Interval (Q10-Q90):")
        interval = results['prediction_intervals']
        print(f"  Coverage: {interval['interval_coverage']:.1%} (target: 80%)")
        print(f"  Mean Width: {interval['mean_width']:.4f}")
        print(f"  Asymmetry: {interval['asymmetry_mean']:.3f}")
        
        print(f"\n🧭 Directional Accuracy:")
        dir_acc = results['directional_accuracy']
        print(f"  Overall: {dir_acc['overall_accuracy']:.1%}")
        print(f"  Up moves: {dir_acc['up_accuracy']:.1%} ({dir_acc['up_samples']} samples)")
        print(f"  Down moves: {dir_acc['down_accuracy']:.1%} ({dir_acc['down_samples']} samples)")
        
        return results
        
    except Exception as e:
        print(f"Error running evaluation: {e}")
        print("Make sure you have:")
        print("1. df_all_macro_analysis.csv from your training script")
        print("2. Trained model saved at ./models/trained_quantile_model.pkl")
        return None


if __name__ == "__main__":
    results = run_comprehensive_evaluation()