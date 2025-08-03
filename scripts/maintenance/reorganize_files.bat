@echo off
echo Starting file reorganization...

REM Create additional required directories
mkdir src\rl_execution\components 2>nul
mkdir config\rl_execution 2>nul
mkdir docs\rl_execution 2>nul
mkdir scripts\data_management 2>nul

echo Created additional directories

REM Core system files
echo Moving core system files...
move ppo_sweep_optuna_tuned_v2.py src\training_pipeline.py
move hummingbot_backtester.py src\backtesting\backtester.py
move save_model_for_production.py src\production\model_persistence.py
move run_hummingbot_backtest.py src\backtesting\run_backtest.py

REM Feature engineering
echo Moving feature engineering files...
move qlib_custom\regime_features.py src\features\regime_features.py
move advanced_position_sizing.py src\features\position_sizing.py

REM Data processing
echo Moving data processing files...
move qlib_custom\crypto_loader_optimized.py src\data\crypto_loader.py
move qlib_custom\gdelt_loader_optimized.py src\data\gdelt_loader.py
move qlib_custom\custom_ndl.py src\data\nested_data_loader.py

REM Models
echo Moving model files...
move qlib_custom\custom_multi_quantile.py src\models\multi_quantile.py
move qlib_custom\custom_signal_env.py src\models\signal_environment.py

REM Production
echo Moving production files...
move realtime_predictor.py src\production\realtime_predictor.py
move hummingbot_bridge.py src\production\hummingbot_bridge.py
move setup_mqtt.py src\production\mqtt_setup.py

REM RL Execution System (check if files exist first)
echo Moving RL execution files...
if exist rl_order_execution\train_meta_wrapper.py (
    move rl_order_execution\train_meta_wrapper.py src\rl_execution\meta_training.py
)
if exist rl_order_execution\troubleshoot_dataloader.py (
    move rl_order_execution\troubleshoot_dataloader.py src\data\dataloader_diagnostics.py
)
if exist rl_order_execution\custom_logger_callback.py (
    move rl_order_execution\custom_logger_callback.py src\logging\tensorboard_logger.py
)

REM RL Components (move all custom_*.py files)
echo Moving RL component files...
if exist rl_order_execution\custom_action_interpreter.py (
    move rl_order_execution\custom_action_interpreter.py src\rl_execution\components\
)
if exist rl_order_execution\custom_data_handler.py (
    move rl_order_execution\custom_data_handler.py src\rl_execution\components\
)
if exist rl_order_execution\custom_data_provider.py (
    move rl_order_execution\custom_data_provider.py src\rl_execution\components\
)
if exist rl_order_execution\custom_order.py (
    move rl_order_execution\custom_order.py src\rl_execution\components\
)
if exist rl_order_execution\custom_reward.py (
    move rl_order_execution\custom_reward.py src\rl_execution\components\
)
if exist rl_order_execution\custom_simulator.py (
    move rl_order_execution\custom_simulator.py src\rl_execution\components\
)
if exist rl_order_execution\custom_state_interpreter.py (
    move rl_order_execution\custom_state_interpreter.py src\rl_execution\components\
)
if exist rl_order_execution\custom_train.py (
    move rl_order_execution\custom_train.py src\rl_execution\components\
)
if exist rl_order_execution\custom_training_vessel.py (
    move rl_order_execution\custom_training_vessel.py src\rl_execution\components\
)

REM Move remaining RL files
if exist rl_order_execution\order_execution_env.py (
    move rl_order_execution\order_execution_env.py src\rl_execution\
)
if exist rl_order_execution\workflow.py (
    move rl_order_execution\workflow.py src\rl_execution\
)

REM Configuration files
echo Moving configuration files...
if exist rl_order_execution\exp_configs\ (
    xcopy rl_order_execution\exp_configs\*.* config\rl_execution\ /E /I /Y
    rmdir rl_order_execution\exp_configs /S /Q
)

REM Scripts (preserve the rl_order_execution/scripts folder as you mentioned)
echo Preserving RL scripts folder...
REM (keeping rl_order_execution\scripts\ as is per your note)

REM Documentation
echo Moving documentation...
if exist rl_order_execution\README.md (
    move rl_order_execution\README.md docs\rl_execution\README.md
)

REM Data management
echo Moving data management files...
move qlib_data_import.txt scripts\data_management\import_commands.txt

REM Move tier logging if it exists in rl_order_execution (might already be in qlib_custom)
if exist rl_order_execution\custom_tier_logging.py (
    move rl_order_execution\custom_tier_logging.py src\rl_execution\components\
)

echo File reorganization complete!
echo.
echo Summary of moves:
echo - Core system files moved to src/
echo - RL execution system organized in src/rl_execution/
echo - Data loaders moved to src/data/
echo - Models moved to src/models/
echo - Production files moved to src/production/
echo - Configuration files moved to config/
echo - Data import commands moved to scripts/data_management/
echo.
echo Note: rl_order_execution/scripts/ folder preserved as requested
echo.
pause