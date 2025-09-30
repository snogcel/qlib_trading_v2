#!/usr/bin/env python3
"""
Pipeline Test Coverage Validator
Comprehensive testing of data flow through the complete quantile trading pipeline
"""

# Add project root to Python path for src imports
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class PipelineCoverageValidator:
    """Validate complete pipeline flow and test coverage"""
    
    def __init__(self):
        self.pipeline_stages = {}
        self.coverage_results = {}
        self.critical_paths = {}
        
    def test_data_ingestion_stage(self):
        """Test Stage 1: Data Ingestion"""
        print("\n[MICROSCOPE] TESTING DATA INGESTION STAGE")
        print("-" * 50)
        
        results = {}
        
        # Test 1: Check if data files exist and are accessible
        expected_data_files = [
            'data3/macro_features.pkl',
            'df_all_macro_analysis.csv'
        ]
        
        data_availability = {}
        for file_path in expected_data_files:
            exists = os.path.exists(file_path)
            data_availability[file_path] = exists
            
            if exists:
                try:
                    if file_path.endswith('.pkl'):
                        df = pd.read_pickle(file_path)
                    else:
                        df = pd.read_csv(file_path)
                    
                    data_availability[f"{file_path}_shape"] = df.shape
                    data_availability[f"{file_path}_columns"] = len(df.columns)
                    
                    print(f"{file_path}: {df.shape}")
                    
                except Exception as e:
                    data_availability[f"{file_path}_error"] = str(e)
                    print(f"{file_path}: ERROR - {e}")
            else:
                print(f"{file_path}: NOT FOUND")
        
        # Test 2: Data quality checks
        try:
            df = pd.read_pickle('data3/macro_features.pkl')
            
            # Check for critical columns
            critical_columns = ['q10', 'q50', 'q90', 'vol_risk', 'vol_raw']
            missing_columns = [col for col in critical_columns if col not in df.columns]
            
            data_quality = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_critical_columns': missing_columns,
                'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'date_range': f"{df.index.get_level_values(1).min()} to {df.index.get_level_values(1).max()}"
            }
            
            quality_score = 1.0 - (len(missing_columns) / len(critical_columns)) - (data_quality['null_percentage'] / 100)
            
            results['data_quality'] = {
                'status': 'PASS' if quality_score > 0.8 else 'FAIL',
                'score': quality_score,
                'details': data_quality
            }
            
            print(f"Data Quality Score: {quality_score:.2f}")
            
        except Exception as e:
            results['data_quality'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"Data Quality: ERROR - {e}")
        
        results['data_availability'] = data_availability
        self.pipeline_stages['data_ingestion'] = results
        return results
    
    def test_feature_engineering_stage(self):
        """Test Stage 2: Feature Engineering"""
        print("\n[GEAR] TESTING FEATURE ENGINEERING STAGE")
        print("-" * 50)
        
        results = {}
        
        try:
            # Load data and test feature engineering
            df = pd.read_pickle('data3/macro_features.pkl')
            
            # Test regime feature availability
            regime_features = [col for col in df.columns if 'regime' in col.lower()]
            volatility_features = [col for col in df.columns if 'vol' in col.lower()]
            sentiment_features = [col for col in df.columns if any(x in col.lower() for x in ['fg', 'btc_dom', 'fear', 'greed'])]
            
            feature_categories = {
                'regime_features': len(regime_features),
                'volatility_features': len(volatility_features), 
                'sentiment_features': len(sentiment_features),
                'total_features': len(df.columns)
            }
            
            # Test feature quality
            feature_quality_checks = {}
            
            # Check volatility features have reasonable ranges
            if 'vol_risk' in df.columns:
                vol_risk_stats = df['vol_risk'].describe()
                feature_quality_checks['vol_risk_range'] = {
                    'min': vol_risk_stats['min'],
                    'max': vol_risk_stats['max'],
                    'reasonable': 0 <= vol_risk_stats['min'] and vol_risk_stats['max'] <= 1
                }
            
            # Check quantile features exist and have reasonable relationships
            if all(col in df.columns for col in ['q10', 'q50', 'q90']):
                quantile_relationships = {
                    'q10_lt_q50': (df['q10'] <= df['q50']).mean(),
                    'q50_lt_q90': (df['q50'] <= df['q90']).mean(),
                    'logical_ordering': (df['q10'] <= df['q50']).mean() > 0.8 and (df['q50'] <= df['q90']).mean() > 0.8
                }
                feature_quality_checks['quantile_relationships'] = quantile_relationships
            
            overall_quality = sum(1 for check in feature_quality_checks.values() 
                                if isinstance(check, dict) and check.get('reasonable', check.get('logical_ordering', False))) / len(feature_quality_checks)
            
            results['feature_engineering'] = {
                'status': 'PASS' if overall_quality > 0.8 else 'FAIL',
                'categories': feature_categories,
                'quality_checks': feature_quality_checks,
                'quality_score': overall_quality
            }
            
            print(f"Feature Categories: {feature_categories}")
            print(f"Feature Quality Score: {overall_quality:.2f}")
            
        except Exception as e:
            results['feature_engineering'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"Feature Engineering: ERROR - {e}")
        
        self.pipeline_stages['feature_engineering'] = results
        return results
    
    def test_signal_generation_stage(self):
        """Test Stage 3: Signal Generation"""
        print("\n[TARGET] TESTING SIGNAL GENERATION STAGE")
        print("-" * 50)
        
        results = {}
        
        try:
            df = pd.read_pickle('data3/macro_features.pkl')
            
            # Test signal-related columns exist
            signal_columns = ['q10', 'q50', 'q90', 'prob_up', 'side', 'signal_tier']
            available_signal_columns = [col for col in signal_columns if col in df.columns]
            
            signal_availability = {
                'expected_columns': len(signal_columns),
                'available_columns': len(available_signal_columns),
                'missing_columns': [col for col in signal_columns if col not in df.columns],
                'coverage': len(available_signal_columns) / len(signal_columns)
            }
            
            # Test signal quality
            signal_quality = {}
            
            if 'side' in df.columns:
                side_distribution = df['side'].value_counts()
                signal_quality['side_distribution'] = {
                    'long_signals': side_distribution.get(1, 0),
                    'short_signals': side_distribution.get(0, 0), 
                    'hold_signals': side_distribution.get(-1, 0),
                    'total_signals': len(df[df['side'] != -1]) if -1 in side_distribution else 0
                }
            
            if 'prob_up' in df.columns:
                prob_up_stats = df['prob_up'].describe()
                signal_quality['prob_up_distribution'] = {
                    'min': prob_up_stats['min'],
                    'max': prob_up_stats['max'],
                    'mean': prob_up_stats['mean'],
                    'valid_range': 0 <= prob_up_stats['min'] and prob_up_stats['max'] <= 1
                }
            
            overall_signal_quality = signal_availability['coverage']
            
            results['signal_generation'] = {
                'status': 'PASS' if overall_signal_quality > 0.7 else 'FAIL',
                'availability': signal_availability,
                'quality': signal_quality,
                'quality_score': overall_signal_quality
            }
            
            print(f"Signal Availability: {signal_availability['coverage']:.2f}")
            if 'side_distribution' in signal_quality:
                print(f"Trading Signals: {signal_quality['side_distribution']['total_signals']:,}")
            
        except Exception as e:
            results['signal_generation'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"Signal Generation: ERROR - {e}")
        
        self.pipeline_stages['signal_generation'] = results
        return results
    
    def test_performance_validation_stage(self):
        """Test Stage 4: Performance Validation"""
        print("\n[CHART_UP] TESTING PERFORMANCE VALIDATION STAGE")
        print("-" * 50)
        
        results = {}
        
        # Test 1: Check if backtest results exist
        backtest_paths = [
            'results/backtests/hummingbot',
            'results/backtests/validated'
        ]
        
        backtest_availability = {}
        for path in backtest_paths:
            exists = os.path.exists(path)
            backtest_availability[path] = exists
            
            if exists:
                files = list(Path(path).glob('**/*'))
                backtest_availability[f"{path}_file_count"] = len(files)
                print(f"{path}: {len(files)} files")
            else:
                print(f"{path}: NOT FOUND")
        
        # Test 2: Performance metrics validation
        try:
            # Look for metrics files
            metrics_files = list(Path('.').glob('**/metrics.json'))
            
            if metrics_files:
                with open(metrics_files[0], 'r') as f:
                    metrics = json.load(f)
                
                # Extract key performance metrics
                performance_metrics = {
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'total_return': metrics.get('total_return', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'total_trades': metrics.get('total_trades', 0)
                }
                
                # Validate performance meets standards
                performance_valid = (
                    performance_metrics['sharpe_ratio'] > 1.0 and
                    performance_metrics['total_return'] > 0.1 and
                    abs(performance_metrics['max_drawdown']) < 0.2
                )
                
                results['performance_validation'] = {
                    'status': 'PASS' if performance_valid else 'FAIL',
                    'metrics': performance_metrics,
                    'meets_standards': performance_valid
                }
                
                print(f"Sharpe Ratio: {performance_metrics['sharpe_ratio']:.3f}")
                print(f"Total Return: {performance_metrics['total_return']:.2%}")
                print(f"Max Drawdown: {performance_metrics['max_drawdown']:.2%}")
                
            else:
                results['performance_validation'] = {
                    'status': 'FAIL',
                    'error': 'No metrics.json files found'
                }
                print("No performance metrics found")
                
        except Exception as e:
            results['performance_validation'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"Performance Validation: ERROR - {e}")
        
        results['backtest_availability'] = backtest_availability
        self.pipeline_stages['performance_validation'] = results
        return results
    
    def test_critical_paths(self):
        """Test critical data flow paths through the pipeline"""
        print("\n[LIGHTNING] TESTING CRITICAL PATHS")
        print("-" * 50)
        
        critical_paths = {
            'data_to_features': {
                'description': 'Raw data → Feature engineering',
                'components': ['data ingestion', 'feature engineering'],
                'test': self._test_data_to_features_path
            },
            'features_to_signals': {
                'description': 'Features → Signal generation', 
                'components': ['feature engineering', 'signal generation'],
                'test': self._test_features_to_signals_path
            },
            'signals_to_performance': {
                'description': 'Signals → Performance metrics',
                'components': ['signal generation', 'performance validation'],
                'test': self._test_signals_to_performance_path
            }
        }
        
        path_results = {}
        
        for path_name, path_info in critical_paths.items():
            try:
                result = path_info['test']()
                path_results[path_name] = {
                    'status': 'PASS' if result else 'FAIL',
                    'description': path_info['description'],
                    'components': path_info['components']
                }
                print(f"{'✅' if result else '❌'} {path_info['description']}: {'PASS' if result else 'FAIL'}")
                
            except Exception as e:
                path_results[path_name] = {
                    'status': 'ERROR',
                    'description': path_info['description'],
                    'error': str(e)
                }
                print(f"{path_info['description']}: ERROR - {e}")
        
        self.critical_paths = path_results
        return path_results
    
    def _test_data_to_features_path(self):
        """Test data ingestion to feature engineering path"""
        try:
            df = pd.read_pickle('data3/macro_features.pkl')
            return len(df) > 1000 and len(df.columns) > 10
        except:
            return False
    
    def _test_features_to_signals_path(self):
        """Test feature engineering to signal generation path"""
        try:
            df = pd.read_pickle('data3/macro_features.pkl')
            has_quantiles = all(col in df.columns for col in ['q10', 'q50', 'q90'])
            has_signals = 'side' in df.columns or 'prob_up' in df.columns
            return has_quantiles and has_signals
        except:
            return False
    
    def _test_signals_to_performance_path(self):
        """Test signal generation to performance validation path"""
        try:
            # Check if performance results exist
            metrics_files = list(Path('.').glob('**/metrics.json'))
            return len(metrics_files) > 0
        except:
            return False
    
    def generate_pipeline_visualization(self):
        """Generate visual representation of pipeline coverage"""
        print("\n[BAR_CHART] GENERATING PIPELINE VISUALIZATION")
        print("-" * 50)
        
        # Calculate coverage scores for each stage
        stage_scores = {}
        
        for stage_name, stage_data in self.pipeline_stages.items():
            if isinstance(stage_data, dict):
                # Calculate overall score for the stage
                passed_tests = sum(1 for test_result in stage_data.values() 
                                 if isinstance(test_result, dict) and test_result.get('status') == 'PASS')
                total_tests = sum(1 for test_result in stage_data.values() 
                                if isinstance(test_result, dict) and 'status' in test_result)
                
                stage_scores[stage_name] = passed_tests / total_tests if total_tests > 0 else 0
        
        # Create visualization data
        visualization_data = {
            'pipeline_stages': stage_scores,
            'critical_paths': {
                path_name: 1.0 if path_data.get('status') == 'PASS' else 0.0
                for path_name, path_data in self.critical_paths.items()
            },
            'overall_health': sum(stage_scores.values()) / len(stage_scores) if stage_scores else 0
        }
        
        print(f"Pipeline Health Score: {visualization_data['overall_health']:.2f}")
        
        return visualization_data
    
    def generate_comprehensive_report(self):
        """Generate comprehensive pipeline coverage report"""
        print("\n" + "="*60)
        print("PIPELINE TEST COVERAGE REPORT")
        print("="*60)
        
        # Run all tests
        self.test_data_ingestion_stage()
        self.test_feature_engineering_stage()
        self.test_signal_generation_stage()
        self.test_performance_validation_stage()
        self.test_critical_paths()
        
        # Generate visualization
        viz_data = self.generate_pipeline_visualization()
        
        # Calculate overall metrics
        total_stages = len(self.pipeline_stages)
        healthy_stages = sum(1 for stage_data in self.pipeline_stages.values()
                           if isinstance(stage_data, dict) and 
                           any(test.get('status') == 'PASS' for test in stage_data.values() 
                               if isinstance(test, dict)))
        
        critical_paths_healthy = sum(1 for path_data in self.critical_paths.values()
                                   if path_data.get('status') == 'PASS')
        total_critical_paths = len(self.critical_paths)
        
        print(f"\n[ROCKET] PIPELINE HEALTH SUMMARY:")
        print(f"  Healthy Stages: {healthy_stages}/{total_stages}")
        print(f"  Critical Paths Working: {critical_paths_healthy}/{total_critical_paths}")
        print(f"  Overall Health Score: {viz_data['overall_health']:.2f}")
        
        # Save detailed results
        os.makedirs('test_results', exist_ok=True)
        
        comprehensive_results = {
            'pipeline_stages': self.pipeline_stages,
            'critical_paths': self.critical_paths,
            'visualization_data': viz_data,
            'summary': {
                'healthy_stages': healthy_stages,
                'total_stages': total_stages,
                'critical_paths_healthy': critical_paths_healthy,
                'total_critical_paths': total_critical_paths,
                'overall_health': viz_data['overall_health']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open('test_results/pipeline_coverage.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print(f"\n[FLOPPY_DISK] Results saved to: test_results/pipeline_coverage.json")
        
        # Recommendations
        if viz_data['overall_health'] >= 0.8:
            print(f"\n[PARTY] EXCELLENT! Pipeline is healthy and well-tested")
            print("  Ready for RD-Agent integration and multi-asset expansion")
        elif viz_data['overall_health'] >= 0.6:
            print(f"\n[THUMBS_UP] GOOD! Pipeline is mostly healthy")
            print("  Focus on improving failed stages before major enhancements")
        else:
            print(f"\n[WARNING] ATTENTION NEEDED! Pipeline has significant gaps")
            print("  Address critical path failures before proceeding")
        
        return comprehensive_results

def main():
    """Run comprehensive pipeline coverage validation"""
    validator = PipelineCoverageValidator()
    results = validator.generate_comprehensive_report()
    return results

if __name__ == "__main__":
    main()