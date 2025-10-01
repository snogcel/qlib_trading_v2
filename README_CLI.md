# Trading Pipeline CLI

A unified command-line interface for quantile trading model training, hyperparameter optimization, and backtesting.

## Features

- **Model Training**: Train multi-quantile models with your crypto and GDELT data
- **Hyperparameter Optimization**: Use Optuna for automated hyperparameter tuning
- **Backtesting**: Run comprehensive backtests with multiple position sizing strategies
- **Full Pipeline**: Execute complete workflows from training to backtesting
- **Configuration Management**: Save and manage configuration settings

## Installation

```bash
# Install in development mode
pip install -e .

# Or install directly
python setup.py install
```

## Quick Start

```bash
# Train a model with default parameters
trading-cli train

# Optimize hyperparameters
trading-cli optimize --trials 100

# Run backtest with price data
trading-cli backtest --price-data ./data/btc_60min.csv

# Full pipeline: train -> optimize -> backtest
trading-cli pipeline --trials 50 --price-data ./data/btc_60min.csv
```

## Commands

### `train` - Train Models

Train multi-quantile models with your data.

```bash
trading-cli train [OPTIONS]

Options:
  --train-start TEXT     Training start date (default: 2018-08-02)
  --train-end TEXT       Training end date (default: 2023-12-31)
  --valid-start TEXT     Validation start date (default: 2024-01-01)
  --valid-end TEXT       Validation end date (default: 2024-09-30)
  --provider-uri TEXT    Qlib data provider URI
  --output-dir TEXT      Model output directory (default: ./models)
  --save-features        Save engineered features
  --quantiles FLOAT...   Quantiles to predict (default: 0.1 0.5 0.9)
```

**Example:**
```bash
trading-cli train --save-features --output-dir ./my_models
```

### `optimize` - Hyperparameter Optimization

Use Optuna to find optimal hyperparameters.

```bash
trading-cli optimize [OPTIONS]

Options:
  --trials INTEGER       Number of optimization trials (default: 100)
  --timeout INTEGER      Optimization timeout in seconds
  --study-name TEXT      Optuna study name
  --storage TEXT         Optuna storage URL
  --optimize-quantile    Which quantile to optimize (default: 0.5)
```

**Example:**
```bash
trading-cli optimize --trials 200 --study-name my_study --timeout 3600
```

### `backtest` - Run Backtesting

Execute comprehensive backtests with real price data.

```bash
trading-cli backtest [OPTIONS]

Options:
  --price-data TEXT         Price data CSV file path
  --initial-balance FLOAT   Initial balance (default: 100000.0)
  --max-position FLOAT      Max position percentage (default: 0.3)
  --fee-rate FLOAT          Trading fee rate (default: 0.001)
  --sizing-method TEXT      Position sizing method (default: enhanced)
  --backtest-output TEXT    Backtest output directory
```

**Sizing Methods:**
- `simple`: Basic position sizing
- `kelly`: Kelly Criterion
- `volatility`: Volatility-adjusted sizing
- `enhanced`: Ensemble approach (recommended)

**Example:**
```bash
trading-cli backtest --price-data ./data/btc_hourly.csv --sizing-method kelly --max-position 0.2
```

### `pipeline` - Full Pipeline

Execute the complete workflow: training, optimization, and backtesting.

```bash
trading-cli pipeline [OPTIONS]

# Combines all options from train, optimize, and backtest commands
```

**Example:**
```bash
trading-cli pipeline \
  --trials 100 \
  --price-data ./data/btc_hourly.csv \
  --save-features \
  --sizing-method enhanced \
  --output-dir ./pipeline_results
```

### `predict` - Generate Predictions

Generate predictions from a trained model.

```bash
trading-cli predict [OPTIONS]

Options:
  --model-path TEXT     Path to trained model (required)
  --input-data TEXT     Input data CSV file
  --output TEXT         Output predictions file (default: ./predictions.csv)
  --start-date TEXT     Prediction start date
  --end-date TEXT       Prediction end date
```

### `config` - Configuration Management

Manage configuration settings.

```bash
trading-cli config [OPTIONS]

Options:
  --show              Show current configuration
  --reset             Reset to default configuration
  --set KEY VALUE     Set configuration value
```

**Examples:**
```bash
# Show current config
trading-cli config --show

# Set default initial balance
trading-cli config --set initial_balance 50000

# Reset to defaults
trading-cli config --reset
```

## Data Requirements

### Price Data Format

For backtesting, provide a CSV file with the following columns:

```csv
datetime,open,high,low,close,volume
2024-01-01 00:00:00,42000.0,42100.0,41900.0,42050.0,1234.56
2024-01-01 01:00:00,42050.0,42200.0,42000.0,42150.0,2345.67
...
```

### Prediction Data Format

The system expects your trained models to generate predictions with these columns:
- `q10`, `q50`, `q90`: Quantile predictions
- `side`: Trading side (1=BUY, 0=SELL, -1=HOLD)
- `signal_tier`: Signal confidence tier
- Various technical and sentiment features

## Output Files

### Training Output
- `model_YYYYMMDD_HHMMSS.pkl`: Trained model
- `feature_importance_*.csv`: Feature importance for each quantile
- `engineered_features.csv`: Complete feature set
- `performance_metrics.json`: Model performance metrics

### Optimization Output
- `best_params_*.json`: Best hyperparameters found
- Optuna study database (if using persistent storage)

### Backtesting Output
- `portfolio_history.csv`: Complete portfolio evolution
- `trades.csv`: Individual trade records
- `holds_analysis.csv`: Hold state analysis
- `signal_analysis_pivot.csv`: Signal analysis in pivot-ready format
- `metrics.json`: Performance metrics
- `report.txt`: Summary report

## Configuration File

The CLI uses a configuration file at `./config/trading_config.json`:

```json
{
  "train_start": "2018-08-02",
  "train_end": "2023-12-31",
  "valid_start": "2024-01-01",
  "valid_end": "2024-09-30",
  "provider_uri": "/Projects/qlib_trading_v2/qlib_data/CRYPTO_DATA",
  "quantiles": [0.1, 0.5, 0.9],
  "initial_balance": 100000.0,
  "max_position": 0.3,
  "fee_rate": 0.001,
  "sizing_method": "enhanced"
}
```

## Advanced Usage

### Custom Hyperparameter Ranges

Modify the `OptunaOptimizer` class to adjust hyperparameter search spaces:

```python
# In src/training/optuna_optimizer.py
"learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
"max_depth": trial.suggest_int("max_depth", 3, 30),
```

### Custom Position Sizing

Add new sizing methods in `HummingbotQuantileBacktester`:

```python
def _my_custom_sizing(self, q10, q50, q90, confidence, **kwargs):
    # Your custom logic here
    return position_size
```

### Integration with External Systems

The CLI outputs are designed to integrate with:
- Hummingbot trading bot
- Portfolio management systems
- Risk management tools
- Reporting dashboards

## Troubleshooting

### Common Issues

1. **Data Path Errors**: Ensure your `provider_uri` points to valid qlib data
2. **Memory Issues**: Reduce data range or use smaller datasets for testing
3. **Model Loading**: Check that model files exist and are compatible
4. **Price Data Alignment**: Ensure price data timestamps match prediction data

### Debug Mode

Set the `DEBUG` environment variable for detailed error traces:

```bash
DEBUG=1 trading-cli train
```

### Logging

The CLI provides detailed logging. Key indicators:
- 🚀 Starting operations
- 🔧 Setup and configuration
- 📊 Data processing
- ✅ Successful completion
- ❌ Errors

## Performance Tips

1. **Use SSD storage** for qlib data provider
2. **Optimize data ranges** - start with smaller date ranges for testing
3. **Parallel processing** - the system uses multiple cores automatically
4. **Memory management** - monitor RAM usage with large datasets
5. **Early stopping** - use reasonable trial counts for optimization

## Contributing

To extend the CLI:

1. Add new commands in `src/cli/trading_cli.py`
2. Implement functionality in appropriate modules
3. Update argument parsing and help text
4. Add tests for new features

## License

MIT License - see LICENSE file for details.