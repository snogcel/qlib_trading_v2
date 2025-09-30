#!/usr/bin/env python3
"""
Demo script for the TestExecutor class.

This script demonstrates the key capabilities of the TestExecutor including:
- Environment setup and cleanup
- Market condition simulation
- Test execution with different test types
- Data loading and caching
- Error handling and recovery
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime

from src.testing.executors.test_executor import TestExecutor
from src.testing.models.test_case import TestCase, TestType, TestPriority
from src.testing.models.test_result import TestStatus
from src.testing.config.data_config import DataConfig


def create_sample_data(n_days: int = 100) -> pd.DataFrame:
    """Create sample market data for testing."""
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    
    # Generate realistic-looking crypto data
    np.random.seed(42)  # For reproducible results
    
    # Price data with some trend and volatility
    returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
    prices = 50000 * (1 + returns).cumprod()  # Starting at $50k
    
    data = pd.DataFrame({
        'close': prices,
        'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        'volume': np.random.lognormal(15, 0.5, n_days),  # Log-normal volume
        
        # Feature data
        'q50': np.random.normal(0, 0.01, n_days),  # Quantile predictions
        'q10': np.random.normal(-0.005, 0.008, n_days),
        'q90': np.random.normal(0.005, 0.008, n_days),
        'vol_risk': np.random.gamma(2, 0.01, n_days),  # Volatility risk
        'spread': np.random.gamma(1.5, 0.005, n_days),  # Bid-ask spread
        
        # Labels (future returns)
        'label': np.random.normal(0, 0.015, n_days)
    }, index=dates)
    
    # Ensure high > low and other realistic constraints
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data


def demo_basic_functionality():
    """Demonstrate basic TestExecutor functionality."""
    print("=" * 60)
    print("DEMO: Basic TestExecutor Functionality")
    print("=" * 60)
    
    # Create data configuration
    data_config = DataConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        test_assets=["BTC", "ETH"],
        max_missing_data_pct=0.05
    )
    
    # Create test executor
    executor = TestExecutor(
        data_config=data_config,
        enable_parallel=False,
        max_execution_time=30.0
    )
    
    print(f"✓ TestExecutor initialized")
    print(f"  - Supported test types: {len(executor.get_supported_test_types())}")
    print(f"  - Max execution time: {executor.max_execution_time}s")
    print(f"  - Data config: {executor.data_config.test_assets}")
    
    # Show executor metadata
    metadata = executor.get_executor_metadata()
    print(f"  - Executor version: {metadata['version']}")
    print(f"  - Capabilities: {list(metadata['capabilities'].keys())}")
    
    return executor


def demo_environment_setup(executor: TestExecutor):
    """Demonstrate test environment setup and cleanup."""
    print("\n" + "=" * 60)
    print("DEMO: Test Environment Setup and Cleanup")
    print("=" * 60)
    
    # Create a test case that requires market data
    test_case = TestCase(
        test_id="demo_env_test",
        feature_name="q50",
        test_type=TestType.ECONOMIC_HYPOTHESIS,
        description="Demo environment setup test",
        test_data_requirements={
            'assets': ['BTC'],
            'start_date': '2024-01-01',
            'end_date': '2024-03-31'
        },
        regime_context="bull",
        market_conditions={'volatility_multiplier': 1.5}
    )
    
    print(f"📋 Test case created: {test_case.test_id}")
    print(f"   - Feature: {test_case.feature_name}")
    print(f"   - Type: {test_case.test_type.value}")
    print(f"   - Regime: {test_case.regime_context}")
    
    # Mock data loading for demo
    sample_data = create_sample_data(90)  # 3 months of data
    
    def mock_load_data_for_case(test_case):
        return sample_data
    
    executor._load_test_data_for_case = mock_load_data_for_case
    
    try:
        # Setup environment
        print("\nSetting up test environment...")
        environment = executor.setup_test_environment(test_case)
        
        print(f"✓ Environment setup completed")
        print(f"   - Test ID: {environment['test_id']}")
        print(f"   - Temp directory: {environment['temp_dir']}")
        print(f"   - Data shape: {environment['data'].shape if environment['data'] is not None else 'None'}")
        print(f"   - Market conditions: {environment['market_conditions']}")
        
        # Verify temp directory exists
        temp_dir = Path(environment['temp_dir'])
        print(f"   - Temp dir exists: {temp_dir.exists()}")
        
        # Show some data statistics
        if environment['data'] is not None:
            data = environment['data']
            print(f"   - Data date range: {data.index.min()} to {data.index.max()}")
            print(f"   - Mean close price: ${data['close'].mean():.2f}")
            print(f"   - Price volatility: {data['close'].pct_change().std():.4f}")
        
    finally:
        # Cleanup environment
        print("\n🧹 Cleaning up test environment...")
        executor.cleanup_test_environment(test_case, environment)
        
        print(f"✓ Environment cleanup completed")
        print(f"   - Temp dir exists: {temp_dir.exists()}")
        print(f"   - Active environments: {len(executor._active_environments)}")


def demo_market_simulation(executor: TestExecutor):
    """Demonstrate market condition simulation."""
    print("\n" + "=" * 60)
    print("DEMO: Market Condition Simulation")
    print("=" * 60)
    
    # Create base data
    base_data = create_sample_data(50)
    print(f"Base data created: {len(base_data)} days")
    print(f"   - Price range: ${base_data['close'].min():.2f} - ${base_data['close'].max():.2f}")
    print(f"   - Base volatility: {base_data['close'].pct_change().std():.4f}")
    
    # Test different market regimes
    regimes = ['bull', 'bear', 'sideways', 'high_volatility', 'low_volatility']
    
    for regime in regimes:
        print(f"\n🏛️ Simulating {regime} market conditions...")
        
        try:
            simulated_data = executor.simulate_market_conditions(regime, base_data)
            
            # Calculate statistics
            returns = simulated_data['close'].pct_change().dropna()
            total_return = (simulated_data['close'].iloc[-1] / simulated_data['close'].iloc[0]) - 1
            volatility = returns.std()
            
            print(f"   ✓ Simulation completed")
            print(f"     - Total return: {total_return:.2%}")
            print(f"     - Volatility: {volatility:.4f}")
            print(f"     - Positive days: {(returns > 0).mean():.1%}")
            
            # Validate regime characteristics
            if regime == 'bull':
                print(f"     - Bull market validated: {total_return > 0}")
            elif regime == 'bear':
                print(f"     - Bear market validated: {total_return < 0}")
            elif regime == 'high_volatility':
                base_vol = base_data['close'].pct_change().std()
                print(f"     - High volatility validated: {volatility > base_vol}")
            
        except Exception as e:
            print(f"   Simulation failed: {str(e)}")


def demo_test_execution(executor: TestExecutor):
    """Demonstrate test case execution."""
    print("\n" + "=" * 60)
    print("DEMO: Test Case Execution")
    print("=" * 60)
    
    # Create sample data for testing
    sample_data = create_sample_data(100)
    
    def mock_load_data_for_case(test_case):
        return sample_data
    
    executor._load_test_data_for_case = mock_load_data_for_case
    
    # Create different types of test cases
    test_cases = []
    
    # Economic hypothesis test
    test_cases.append(TestCase(
        test_id="econ_001",
        feature_name="q50",
        test_type=TestType.ECONOMIC_HYPOTHESIS,
        description="Test Q50 directional bias in bull market",
        validation_criteria={'directional_bias': True},
        regime_context="bull",
        priority=TestPriority.HIGH
    ))
    
    # Performance test
    test_cases.append(TestCase(
        test_id="perf_001",
        feature_name="portfolio",
        test_type=TestType.PERFORMANCE,
        description="Test portfolio performance metrics",
        validation_criteria={'sharpe_ratio': 0.5},
        priority=TestPriority.MEDIUM
    ))
    
    # Failure mode test
    test_cases.append(TestCase(
        test_id="fail_001",
        feature_name="vol_risk",
        test_type=TestType.FAILURE_MODE,
        description="Test vol_risk handling of data gaps",
        test_parameters={'failure_type': 'data_gaps'},
        validation_criteria={'graceful_handling': True},
        priority=TestPriority.HIGH
    ))
    
    # Implementation test
    test_cases.append(TestCase(
        test_id="impl_001",
        feature_name="vol_risk",
        test_type=TestType.IMPLEMENTATION,
        description="Validate vol_risk calculation formula",
        validation_criteria={'implementation_valid': True},
        priority=TestPriority.CRITICAL
    ))
    
    print(f"📋 Created {len(test_cases)} test cases")
    for tc in test_cases:
        print(f"   - {tc.test_id}: {tc.test_type.value} ({tc.priority.value})")
    
    # Execute individual tests
    print(f"\n🔬 Executing individual tests...")
    
    for i, test_case in enumerate(test_cases):
        print(f"\n   Test {i+1}/{len(test_cases)}: {test_case.test_id}")
        
        try:
            result = executor.execute_single_test(test_case)
            
            print(f"     ✓ Status: {result.status.value}")
            print(f"     - Passed: {result.passed}")
            print(f"     - Execution time: {result.execution_time:.3f}s")
            print(f"     - Confidence: {result.confidence.value} ({result.confidence_score:.2f})")
            
            if result.analysis:
                print(f"     - Analysis: {result.analysis[:100]}...")
            
            if result.error_message:
                print(f"     - Error: {result.error_message}")
                
        except Exception as e:
            print(f"     Execution failed: {str(e)}")
    
    # Execute test suite
    print(f"\nExecuting complete test suite...")
    
    try:
        results = executor.execute_test_suite(test_cases)
        
        # Analyze results
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = sum(1 for r in results if not r.passed and r.status != TestStatus.SKIPPED)
        
        print(f"✓ Test suite execution completed")
        print(f"   - Total tests: {total_tests}")
        print(f"   - Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"   - Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        
        # Show execution statistics
        stats = executor.get_execution_statistics()
        print(f"   - Success rate: {stats['success_rate']:.1%}")
        print(f"   - Average execution time: {stats['average_execution_time']:.3f}s")
        
    except Exception as e:
        print(f"Test suite execution failed: {str(e)}")


def demo_data_caching(executor: TestExecutor):
    """Demonstrate data caching functionality."""
    print("\n" + "=" * 60)
    print("DEMO: Data Caching")
    print("=" * 60)
    
    # Mock data loading to track calls
    call_count = 0
    sample_data = create_sample_data(50)
    
    def mock_load_test_data(data_requirements):
        nonlocal call_count
        call_count += 1
        print(f"   📥 Data loading call #{call_count}")
        return {'BTC': sample_data}
    
    # Patch the load_test_data function
    import src.testing.utils.data_utils
    original_load = src.testing.utils.data_utils.load_test_data
    src.testing.utils.data_utils.load_test_data = mock_load_test_data
    
    try:
        data_requirements = {
            'assets': ['BTC'],
            'start_date': '2024-01-01',
            'end_date': '2024-03-31'
        }
        
        print(f"🗄️ Testing data caching with requirements: {data_requirements}")
        
        # First load - should call data loader
        print(f"\n   First load (cache miss expected):")
        data1 = executor.load_test_data(data_requirements)
        print(f"     - Data shape: {data1.shape}")
        print(f"     - Cache hits: {executor._cache_hits}")
        print(f"     - Cache misses: {executor._cache_misses}")
        
        # Second load - should use cache
        print(f"\n   Second load (cache hit expected):")
        data2 = executor.load_test_data(data_requirements)
        print(f"     - Data shape: {data2.shape}")
        print(f"     - Cache hits: {executor._cache_hits}")
        print(f"     - Cache misses: {executor._cache_misses}")
        
        # Third load with different requirements - should call data loader again
        different_requirements = {
            'assets': ['ETH'],
            'start_date': '2024-01-01',
            'end_date': '2024-03-31'
        }
        
        print(f"\n   Third load with different requirements (cache miss expected):")
        data3 = executor.load_test_data(different_requirements)
        print(f"     - Data shape: {data3.shape}")
        print(f"     - Cache hits: {executor._cache_hits}")
        print(f"     - Cache misses: {executor._cache_misses}")
        
        print(f"\n✓ Caching demo completed")
        print(f"   - Total data loader calls: {call_count}")
        print(f"   - Cache efficiency: {executor._cache_hits/(executor._cache_hits + executor._cache_misses)*100:.1f}%")
        
    finally:
        # Restore original function
        src.testing.utils.data_utils.load_test_data = original_load


def demo_error_handling(executor: TestExecutor):
    """Demonstrate error handling and recovery."""
    print("\n" + "=" * 60)
    print("DEMO: Error Handling and Recovery")
    print("=" * 60)
    
    # Test case with invalid test function
    print("🚨 Testing error handling...")
    
    def failing_test_function(data, params):
        raise ValueError("Simulated test failure")
    
    error_test_case = TestCase(
        test_id="error_test",
        feature_name="test_feature",
        test_type=TestType.ECONOMIC_HYPOTHESIS,
        description="Test case designed to fail",
        test_function=failing_test_function,
        test_data_requirements={'assets': ['BTC']}
    )
    
    # Mock data loading
    sample_data = create_sample_data(30)
    executor._load_test_data_for_case = lambda tc: sample_data
    
    print(f"   Executing failing test case...")
    result = executor.execute_single_test(error_test_case)
    
    print(f"✓ Error handling completed")
    print(f"   - Status: {result.status.value}")
    print(f"   - Error message: {result.error_message}")
    print(f"   - Has stack trace: {result.stack_trace is not None}")
    print(f"   - Analysis: {result.analysis}")
    
    # Test timeout handling
    print(f"\n⏱️ Testing timeout handling...")
    
    def slow_test_function(data, params):
        import time
        time.sleep(0.1)  # Sleep for 100ms
        return {'result': 'completed'}
    
    timeout_test_case = TestCase(
        test_id="timeout_test",
        feature_name="test_feature",
        test_type=TestType.PERFORMANCE,
        description="Test case with potential timeout",
        test_function=slow_test_function,
        test_data_requirements={'assets': ['BTC']}
    )
    
    # Set very short timeout for demo
    original_timeout = executor.max_execution_time
    executor.max_execution_time = 0.05  # 50ms timeout
    
    try:
        result = executor.execute_single_test(timeout_test_case)
        print(f"✓ Timeout test completed")
        print(f"   - Status: {result.status.value}")
        print(f"   - Execution time: {result.execution_time:.3f}s")
        
        if result.status == TestStatus.TIMEOUT:
            print(f"   - Timeout detected correctly")
        else:
            print(f"   - Test completed within timeout")
            
    finally:
        # Restore original timeout
        executor.max_execution_time = original_timeout


def main():
    """Run all demos."""
    print("TestExecutor Demo Script")
    print("This script demonstrates the key capabilities of the TestExecutor class.")
    
    try:
        # Basic functionality
        executor = demo_basic_functionality()
        
        # Environment setup and cleanup
        demo_environment_setup(executor)
        
        # Market simulation
        demo_market_simulation(executor)
        
        # Test execution
        demo_test_execution(executor)
        
        # Data caching
        demo_data_caching(executor)
        
        # Error handling
        demo_error_handling(executor)
        
        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)
        
        # Final statistics
        final_stats = executor.get_execution_statistics()
        print(f"Final Execution Statistics:")
        print(f"   - Total tests executed: {final_stats['total_tests_executed']}")
        print(f"   - Success rate: {final_stats['success_rate']:.1%}")
        print(f"   - Average execution time: {final_stats['average_execution_time']:.3f}s")
        print(f"   - Cache hit rate: {final_stats['cache_hit_rate']:.1%}")
        
    except Exception as e:
        print(f"\nDemo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())