"""
Master Feature Optimization Runner
Coordinates feature experiments and hyperparameter optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import argparse
import sys

from feature_experiment_suite import FeatureExperimentSuite, run_feature_experiments
from hyperparameter_optimizer import HyperparameterOptimizer, run_hyperparameter_optimization
from quantile_backtester import QuantileBacktester, BacktestConfig

def run_quick_feature_test():
    """Run a quick test of the most promising feature combinations"""
    
    print("="*80)
    print("QUICK FEATURE TEST - Top 3 Most Promising Configurations")
    print("="*80)
    
    suite = FeatureExperimentSuite()
    
    # Define top 3 most promising experiments
    quick_experiments = [
        "enhanced_volatility",
        "multi_timeframe_momentum", 
        "feature_interactions"
    ]
    
    all_experiments = suite.define_feature_experiments()
    selected_experiments = [exp for exp in all_experiments if exp.name in quick_experiments]
    
    results = {}
    for config in selected_experiments:
        print(f"\nTesting {config.name}...")
        result = suite.run_experiment(config)
        results[config.name] = result
    
    # Quick comparison
    print(f"\n{'='*60}")
    print("QUICK TEST RESULTS COMPARISON")
    print(f"{'='*60}")
    
    for exp_name, result in results.items():
        if 'error' not in result:
            aggressive_result = result['backtest_results']['aggressive']
            print(f"{exp_name:25} | Return: {aggressive_result['total_return']:6.2%} | "
                  f"Sharpe: {aggressive_result['sharpe_ratio']:5.2f} | "
                  f"Trades: {aggressive_result['total_trades']:5d}")
    
    return results

def run_targeted_optimization(best_feature_config: str):
    """Run targeted optimization on the best performing feature configuration"""
    
    print(f"\n{'='*80}")
    print(f"TARGETED OPTIMIZATION - {best_feature_config}")
    print(f"{'='*80}")
    
    # Load the best feature configuration
    suite = FeatureExperimentSuite()
    all_experiments = suite.define_feature_experiments()
    target_config = next((exp for exp in all_experiments if exp.name == best_feature_config), None)
    
    if not target_config:
        print(f"Configuration {best_feature_config} not found!")
        return None
    
    # Engineer features
    df_features = suite.engineer_features(target_config)
    
    # Run targeted hyperparameter optimization
    optimizer = HyperparameterOptimizer()
    
    # Optimize backtest configuration
    print("Optimizing backtest configuration...")
    best_config, optimization_results = optimizer.optimize_backtest_config(
        df_features, n_trials=50
    )
    
    # Optimize signal thresholds
    print("Optimizing signal thresholds...")
    threshold_results = optimizer.optimize_signal_thresholds(
        df_features, best_config, n_trials=30
    )
    
    # Test final optimized configuration
    print("Testing final optimized configuration...")
    
    # Apply optimized thresholds
    df_optimized = df_features.copy()
    best_thresh_params = threshold_results['best_params']
    
    # Recalculate thresholds with optimized parameters
    span = best_thresh_params.get('rolling_window', 30)
    min_span = max(5, span // 3)
    
    df_optimized["signal_thresh"] = (
        df_optimized["abs_q50"]
        .rolling(window=span, min_periods=min_span)
        .quantile(best_thresh_params.get('signal_thresh_percentile', 0.85))
        .fillna(method="bfill")
    )
    
    df_optimized["spread_thresh"] = (
        df_optimized["spread"]
        .rolling(window=span, min_periods=min_span)
        .quantile(best_thresh_params.get('spread_thresh_percentile', 0.85))
        .fillna(method="bfill")
    )
    
    # Recalculate derived features
    df_optimized = optimizer._recalculate_derived_features(df_optimized)
    
    # Final backtest
    total_len = len(df_optimized)
    test_start = int(total_len * 0.8)
    df_test = df_optimized.iloc[test_start:]
    
    final_backtester = QuantileBacktester(best_config)
    final_results = final_backtester.run_backtest(df_test, price_col='truth')
    
    # Generate final report
    print(f"\n{'='*60}")
    print("FINAL OPTIMIZED RESULTS")
    print(f"{'='*60}")
    print(final_backtester.generate_report(final_results))
    
    # Save optimized configuration
    optimized_config = {
        'feature_config': target_config.name,
        'backtest_config': best_config.__dict__,
        'threshold_params': best_thresh_params,
        'final_metrics': final_backtester.metrics,
        'optimization_timestamp': datetime.now().isoformat()
    }
    
    output_file = Path("./optimized_configurations") / f"optimized_{best_feature_config}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(optimized_config, f, indent=2, default=str)
    
    print(f"\nOptimized configuration saved to {output_file}")
    
    return optimized_config

def analyze_feature_importance():
    """Analyze feature importance across different configurations"""
    
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*80}")
    
    suite = FeatureExperimentSuite()
    
    # Test key feature groups
    feature_groups_to_test = [
        "baseline",
        "enhanced_volatility", 
        "multi_timeframe_momentum",
        "feature_interactions"
    ]
    
    all_experiments = suite.define_feature_experiments()
    selected_experiments = [exp for exp in all_experiments if exp.name in feature_groups_to_test]
    
    importance_summary = {}
    
    for config in selected_experiments:
        print(f"\nAnalyzing {config.name}...")
        
        # Engineer features
        df_features = suite.engineer_features(config)
        
        # Calculate feature importance
        importance = suite._calculate_feature_importance(df_features, config)
        importance_summary[config.name] = importance
        
        # Show top 10 features
        print(f"Top 10 features for {config.name}:")
        for i, (feature, score) in enumerate(list(importance.items())[:10]):
            print(f"  {i+1:2d}. {feature:25} | {score:.4f}")
    
    # Find most consistently important features
    all_features = set()
    for importance in importance_summary.values():
        all_features.update(importance.keys())
    
    feature_consistency = {}
    for feature in all_features:
        scores = []
        for config_name, importance in importance_summary.items():
            if feature in importance:
                scores.append(importance[feature])
        
        if len(scores) > 1:  # Feature appears in multiple configurations
            feature_consistency[feature] = {
                'mean_importance': np.mean(scores),
                'std_importance': np.std(scores),
                'consistency_score': np.mean(scores) / (np.std(scores) + 1e-6),
                'appears_in': len(scores)
            }
    
    # Sort by consistency score
    consistent_features = sorted(
        feature_consistency.items(), 
        key=lambda x: x[1]['consistency_score'], 
        reverse=True
    )
    
    print(f"\n{'='*60}")
    print("MOST CONSISTENTLY IMPORTANT FEATURES")
    print(f"{'='*60}")
    
    for i, (feature, stats) in enumerate(consistent_features[:15]):
        print(f"{i+1:2d}. {feature:25} | "
              f"Mean: {stats['mean_importance']:.4f} | "
              f"Consistency: {stats['consistency_score']:.2f} | "
              f"Appears in: {stats['appears_in']} configs")
    
    # Save feature importance analysis
    output_file = Path("./feature_analysis") / f"feature_importance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)
    
    analysis_results = {
        'importance_by_config': importance_summary,
        'feature_consistency': feature_consistency,
        'most_consistent_features': dict(consistent_features[:20])
    }
    
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"\nFeature importance analysis saved to {output_file}")
    
    return analysis_results

def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description="Feature Optimization Suite")
    parser.add_argument("--mode", choices=["quick", "full", "optimize", "analyze"], 
                       default="quick", help="Optimization mode")
    parser.add_argument("--config", type=str, help="Specific configuration to optimize")
    parser.add_argument("--trials", type=int, default=100, help="Number of optimization trials")
    
    args = parser.parse_args()
    
    print(f"Starting Feature Optimization Suite - Mode: {args.mode}")
    print(f"Timestamp: {datetime.now()}")
    
    if args.mode == "quick":
        print("\nRunning quick feature test...")
        results = run_quick_feature_test()
        
        # Find best performing configuration
        best_config = None
        best_score = -999
        
        for config_name, result in results.items():
            if 'error' not in result:
                score = result['backtest_results']['aggressive']['sharpe_ratio']
                if score > best_score:
                    best_score = score
                    best_config = config_name
        
        if best_config:
            print(f"\n🏆 Best performing configuration: {best_config} (Sharpe: {best_score:.3f})")
            print(f"💡 Recommendation: Run 'python run_feature_optimization.py --mode optimize --config {best_config}' for detailed optimization")
    
    elif args.mode == "full":
        print("\n🔬 Running full feature experiment suite...")
        results = run_feature_experiments()
        
        print("\n⚙️ Running comprehensive hyperparameter optimization...")
        optimization_results = run_hyperparameter_optimization()
    
    elif args.mode == "optimize":
        if not args.config:
            print("Error: --config required for optimize mode")
            sys.exit(1)
        
        print(f"\n🎯 Running targeted optimization for {args.config}...")
        results = run_targeted_optimization(args.config)
    
    elif args.mode == "analyze":
        print("\nRunning feature importance analysis...")
        results = analyze_feature_importance()
    
    print(f"\nFeature optimization completed at {datetime.now()}")

if __name__ == "__main__":
    main()