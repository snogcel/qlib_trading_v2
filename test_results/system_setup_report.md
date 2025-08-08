# Feature Test Coverage System Report

**Version:** 1.0.0
**Configuration Directory:** config\feature_testing
**Feature Template:** docs\FEATURE_KNOWLEDGE_TEMPLATE.md
**Output Directory:** test_results\feature_coverage
**Components Initialized:** False

## Configuration Summary

### Execution
- **parallel_execution:** True
- **max_workers:** 4
- **test_timeout:** 30.0
- **retry_failed_tests:** True
- **max_retries:** 2

### Validation
- **confidence_threshold:** 0.6
- **performance_tolerance:** 0.05
- **statistical_significance:** 0.05
- **data_quality_threshold:** 0.95

### Data_Loader
- **data_directory:** data
- **crypto_data_path:** qlib_data\CRYPTO
- **backup_sources:** ['data/processed', 'data/raw']
- **required_fields:** ['open', 'high', 'low', 'close', 'volume']
- **quality_threshold:** 0.95
- **cache_enabled:** True
- **cache_directory:** data\cache\feature_testing

### Paths
- **template:** docs\FEATURE_KNOWLEDGE_TEMPLATE.md
- **output:** test_results\feature_coverage
- **logs:** logs\feature_testing

## System Validation
**Status:** PASSED

All system validation checks passed successfully.