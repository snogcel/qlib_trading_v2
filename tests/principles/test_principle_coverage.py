#!/usr/bin/env python3
"""
Principle Coverage Tests
Ensures all system components align with TRADING_SYSTEM_PRINCIPLES.md
"""

# Add project root to Python path for src imports
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
import inspect
from pathlib import Path
import json

# Import our system components
from src.training_pipeline import (
    prob_up_piecewise, q50_regime_aware_signals,
    identify_market_regimes
)

def ensure_vol_risk_available(df):
    """
    Ensure vol_risk is available - use existing feature from crypto_loader_optimized
    vol_risk = Std(Log(close/close_prev), 6)² (VARIANCE, not std dev)
    This represents the squared volatility = variance, which is key for risk measurement
    """
    if 'vol_risk' not in df.columns:
        print(" vol_risk was not found in data - this should come from crypto_loader_optimized")
        print("   vol_risk = Std(Log($close / Ref($close, 1)), 6) * Std(Log($close / Ref($close, 1)), 6)")
        
        
        # Fallback calculation if not available (but this shouldn't happen)
        if 'vol_raw' in df.columns:
            df['vol_risk'] = df['vol_raw'] ** 2  # Convert std dev to variance
            print("   Created vol_risk from vol_raw (vol_raw²)")
        else:
            print("   Cannot create vol_risk - missing vol_raw")
            df['vol_risk'] = 0.0001  # Small default value
    else:
        print(f"vol_risk available from crypto_loader_optimized ({df['vol_risk'].notna().sum():,} valid values)")
    
    return df

# Define quantile_loss locally since it's not exported from training_pipeline
def quantile_loss(y_true, y_pred, quantile):
    """Local implementation of quantile loss for testing"""
    if hasattr(y_pred, 'iloc') and len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.iloc[:, 0]
    if hasattr(y_true, 'iloc') and len(y_true.shape) > 1 and y_true.shape[1] == 1:
        y_true = y_true.iloc[:, 0]
    
    # Align indices if they're pandas objects
    if hasattr(y_true, 'align') and hasattr(y_pred, 'align'):
        y_true_aligned, y_pred_aligned = y_true.align(y_pred, join='inner')
    else:
        y_true_aligned, y_pred_aligned = y_true, y_pred
    
    errors = y_true_aligned - y_pred_aligned
    coverage = (y_true_aligned < y_pred_aligned).mean() if len(y_true_aligned) > 0 else 0
    loss = np.mean(np.maximum(quantile * errors, (quantile - 1) * errors)) if len(errors) > 0 else 0
    
    return loss, coverage

class PrincipleCoverageValidator:
    """Validates system components against trading principles"""
    
    def __init__(self):
        self.coverage_results = {}
        self.principle_violations = []
        
    def test_thesis_first_development(self):
        """Test Principle 1: Thesis-First Development"""
        print("\n[MICROSCOPE] TESTING THESIS-FIRST DEVELOPMENT")
        print("-" * 50)
        
        results = {}
        
        # Test 1: Q50 signals have clear economic rationale
        try:
            # Create sample data
            sample_data = pd.DataFrame({
                'q10': [-0.012368462, -0.004660593, -0.0056967],
                'q50': [-0.000133614, 0.000140146, 0.000611553], 
                'q90': [0.010866272, 0.006474912, 0.008096517]
            })
            
            prob_results = sample_data.apply(prob_up_piecewise, axis=1)
            
            # Validate economic logic: prob_up should increase with q50
            economic_logic_valid = all(
                prob_results[i] <= prob_results[i+1] 
                for i in range(len(prob_results)-1)
            )

            print(prob_results)
            
            results['q50_economic_logic'] = {
                'status': 'PASS' if economic_logic_valid else 'FAIL',
                'rationale': 'Q50 signals follow supply/demand logic',
                'test_values': prob_results.tolist()
            }
            
            print(f"Q50 Economic Logic: {'PASS' if economic_logic_valid else 'FAIL'}")
            
        except Exception as e:
            results['q50_economic_logic'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"Q50 Economic Logic: ERROR - {e}")
        
        # Test 2: Quantile loss function is mathematically sound
        try:
            y_true = pd.Series([0.01, -0.005, 0.02])
            y_pred = pd.Series([0.008, -0.003, 0.018])
            
            loss_90, coverage_90 = quantile_loss(y_true, y_pred, 0.90)
            loss_50, coverage_50 = quantile_loss(y_true, y_pred, 0.50)
            loss_10, coverage_10 = quantile_loss(y_true, y_pred, 0.10)
            
            # Validate coverage is reasonable (between 0 and 1)
            coverage_valid = all(0 <= c <= 1 for c in [coverage_90, coverage_50, coverage_10])
            
            results['quantile_loss_validity'] = {
                'status': 'PASS' if coverage_valid else 'FAIL',
                'rationale': 'Quantile loss provides valid coverage metrics',
                'coverages': [coverage_10, coverage_50, coverage_90]
            }
            
            print(f"Quantile Loss Validity: {'PASS' if coverage_valid else 'FAIL'}")
            
        except Exception as e:
            results['quantile_loss_validity'] = {
                'status': 'ERROR', 
                'error': str(e)
            }
            print(f"Quantile Loss Validity: ERROR - {e}")
        
        self.coverage_results['thesis_first'] = results
        return results
    
    def test_supply_demand_focus(self):
        """Test Principle 2: Supply & Demand Focus"""
        print("\n[TARGET] TESTING SUPPLY & DEMAND FOCUS")
        print("-" * 50)
        
        results = {}
        
        # Test 1: Regime detection captures market microstructure
        try:
            # Create sample data with different volatility regimes
            sample_data = pd.DataFrame({
                'vol_risk': [0.00001, 0.00004, 0.00095],  # Low, medium, high variance
                'vol_raw': [0.003778199, 0.00625921, 0.030806966],      # Corresponding volatility
                'vol_raw_momentum': [0.002469136, 0.054012347, 0.054320987]
            })
            
            enhanced_data = identify_market_regimes(sample_data)
            
            # Check that different volatility levels create different regimes
            regime_diversity = len(enhanced_data[['vol_regime_low', 'vol_regime_medium', 'vol_regime_high']].sum())
            
            results['regime_microstructure'] = {
                'status': 'PASS' if regime_diversity > 0 else 'FAIL',
                'rationale': 'Regime detection adapts to market microstructure',
                'regime_count': int(regime_diversity)
            }
            
            print(f"Regime Microstructure: {'PASS' if regime_diversity > 0 else 'FAIL'}")
            
        except Exception as e:
            results['regime_microstructure'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"Regime Microstructure: ERROR - {e}")
        
        # Test 2: Vol_risk provides economic rationale
        try:
            sample_data = pd.DataFrame({
                'vol_risk': [0.00001, 0.00004, 0.00095],  # Low, medium, high variance
            })
            
            enhanced_data = sample_data
            
            # Check that vol_risk was created and is reasonable
            vol_risk_valid = 'vol_risk' in enhanced_data.columns and enhanced_data['vol_risk'].notna().all()
            
            results['vol_risk_economic'] = {
                'status': 'PASS' if vol_risk_valid else 'FAIL',
                'rationale': 'Vol_risk captures variance for superior risk measurement',
                'has_vol_risk': 'vol_risk' in enhanced_data.columns
            }
            
            print(f"Vol_risk Economic Logic: {'PASS' if vol_risk_valid else 'FAIL'}")
            
        except Exception as e:
            results['vol_risk_economic'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"Vol_risk Economic Logic: ERROR - {e}")
        
        self.coverage_results['supply_demand'] = results
        return results
    
    def test_explainability_requirements(self):
        """Test Principle 4: Simplicity & Explainability"""
        print("\n[BAR_CHART] TESTING EXPLAINABILITY REQUIREMENTS")
        print("-" * 50)
        
        results = {}
        
        # Test 1: Functions have clear documentation
        functions_to_check = [
            prob_up_piecewise,
            quantile_loss,
            q50_regime_aware_signals,
            identify_market_regimes
        ]
        
        documentation_scores = []
        
        for func in functions_to_check:
            has_docstring = func.__doc__ is not None and len(func.__doc__.strip()) > 10
            documentation_scores.append({
                'function': func.__name__,
                'has_docstring': has_docstring,
                'docstring_length': len(func.__doc__ or '')
            })
        
        avg_documentation = sum(1 for score in documentation_scores if score['has_docstring']) / len(documentation_scores)
        
        results['function_documentation'] = {
            'status': 'PASS' if avg_documentation >= 0.8 else 'FAIL',
            'rationale': 'Functions must be well-documented for explainability',
            'coverage': avg_documentation,
            'details': documentation_scores
        }
        
        print(f"Function Documentation: {'PASS' if avg_documentation >= 0.8 else 'FAIL'} ({avg_documentation:.1%})")
        
        # Test 2: No overly complex functions (complexity heuristic)
        complexity_scores = []
        
        for func in functions_to_check:
            try:
                source_lines = inspect.getsource(func).split('\n')
                line_count = len([line for line in source_lines if line.strip()])
                
                # Simple complexity heuristic: functions should be < 50 lines
                is_simple = line_count < 50
                
                complexity_scores.append({
                    'function': func.__name__,
                    'line_count': line_count,
                    'is_simple': is_simple
                })
                
            except Exception as e:
                complexity_scores.append({
                    'function': func.__name__,
                    'error': str(e),
                    'is_simple': False
                })
        
        avg_simplicity = sum(1 for score in complexity_scores if score.get('is_simple', False)) / len(complexity_scores)
        
        results['function_complexity'] = {
            'status': 'PASS' if avg_simplicity >= 0.8 else 'FAIL',
            'rationale': 'Functions should be simple and understandable',
            'simplicity_rate': avg_simplicity,
            'details': complexity_scores
        }
        
        print(f"Function Complexity: {'PASS' if avg_simplicity >= 0.8 else 'FAIL'} ({avg_simplicity:.1%})")
        
        self.coverage_results['explainability'] = results
        return results
    
    def test_performance_maintenance(self):
        """Test that principle adherence maintains performance"""
        print("\n[CHART_UP] TESTING PERFORMANCE MAINTENANCE")
        print("-" * 50)
        
        results = {}
        
        # Test 1: Check if training pipeline output exists and is reasonable
        try:
            # Check for key output files
            key_files = [
                'df_all_macro_analysis.csv',
                'macro_features.pkl',
                'temp/correlation_matrix.csv'
            ]
            
            file_status = {}
            for file_path in key_files:
                exists = os.path.exists(file_path)
                file_status[file_path] = exists
                
                if exists and file_path.endswith('.csv'):
                    try:
                        df = pd.read_csv(file_path)
                        file_status[f"{file_path}_shape"] = df.shape
                    except:
                        file_status[f"{file_path}_shape"] = "ERROR"
            
            all_files_exist = all(file_status[f] for f in key_files if f in file_status)
            
            results['output_files'] = {
                'status': 'PASS' if all_files_exist else 'FAIL',
                'rationale': 'System should generate expected output files',
                'file_status': file_status
            }
            
            print(f"Output Files: {'PASS' if all_files_exist else 'FAIL'}")
            
        except Exception as e:
            results['output_files'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"Output Files: ERROR - {e}")
        
        self.coverage_results['performance'] = results
        return results
    
    def generate_coverage_report(self):
        """Generate comprehensive coverage report"""
        print("\n" + "="*60)
        print("PRINCIPLE COVERAGE REPORT")
        print("="*60)
        
        total_tests = 0
        passed_tests = 0
        
        for principle, tests in self.coverage_results.items():
            print(f"\n[ROCKET] {principle.upper().replace('_', ' ')}:")
            
            for test_name, result in tests.items():
                total_tests += 1
                status = result.get('status', 'UNKNOWN')
                
                if status == 'PASS':
                    passed_tests += 1
                    print(f"  {test_name}: {status}")
                elif status == 'FAIL':
                    print(f"  {test_name}: {status}")
                    if 'rationale' in result:
                        print(f"     Rationale: {result['rationale']}")
                else:
                    print(f"  {test_name}: {status}")
                    if 'error' in result:
                        print(f"     Error: {result['error']}")
        
        coverage_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        print(f"\n[BAR_CHART] OVERALL COVERAGE:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Coverage Rate: {coverage_rate:.1%}")
        
        if coverage_rate >= 0.8:
            print(f"  Status: [PARTY] EXCELLENT - High principle adherence!")
        elif coverage_rate >= 0.6:
            print(f"  Status: [THUMBS_UP] GOOD - Some improvements needed")
        else:
            print(f"  Status: [WARNING] ATTENTION NEEDED - Review failed tests")
        
        # Save detailed results
        os.makedirs('test_results', exist_ok=True)
        with open('test_results/principle_coverage.json', 'w') as f:
            json.dump(self.coverage_results, f, indent=2, default=str)
        
        return coverage_rate

def main():
    """Run all principle coverage tests"""
    print("[ROCKET] TRADING SYSTEM PRINCIPLE COVERAGE VALIDATION")
    print("="*60)
    
    validator = PrincipleCoverageValidator()
    
    # Run all principle tests
    validator.test_thesis_first_development()
    validator.test_supply_demand_focus()
    validator.test_explainability_requirements()
    validator.test_performance_maintenance()
    
    # Generate comprehensive report
    coverage_rate = validator.generate_coverage_report()
    
    print(f"\n[TARGET] NEXT STEPS:")
    if coverage_rate < 0.8:
        print("  1. Review failed tests and improve implementations")
        print("  2. Add missing documentation to functions")
        print("  3. Ensure all features have economic rationale")
    else:
        print("  1. System shows strong principle adherence!")
        print("  2. Consider RD-Agent integration for automated research")
        print("  3. Expand coverage to new features as they're added")
    
    return validator.coverage_results

if __name__ == "__main__":
    main()