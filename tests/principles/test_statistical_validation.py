#!/usr/bin/env python3
"""
Statistical Validation Test Suite
Implements requirements from docs/SYSTEM_VALIDATION_SPEC.md for all features in docs/FEATURE_DOCUMENTATION.md

This test suite ensures:
1. Time-series aware cross-validation (no look-ahead bias)
2. Out-of-sample testing (performance on unseen data)
3. Regime robustness (works across bull/bear/sideways markets)
4. Feature stability (predictive power persists over time)
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy import stats
import warnings

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import actual implementation functions
from src.training_pipeline import (
    q50_regime_aware_signals,
    prob_up_piecewise,
    kelly_with_vol_raw_deciles,
    get_vol_raw_decile,
    identify_market_regimes,
    ensure_vol_risk_available
)

class TestStatisticalValidation:
    """Statistical validation tests for all documented features"""
    
    @pytest.fixture
    def time_se