# Integration Proof of Concept Framework

## 🎯 Overview
A systematic approach to validate both NautilusTrader and Hummingbot integration paths through minimal viable implementations, allowing data-driven platform selection.

---

## POC Testing Strategy

### **Parallel POC Approach**
```
Week 1-2: Build Both POCs in Parallel
├── NautilusTrader POC (Primary)
│   ├── Basic strategy implementation
│   ├── Q50 signal integration
│   └── Paper trading validation
└── Hummingbot POC (Fallback)
    ├── External signal provider
    ├── MQTT integration
    └── Strategy modification

Week 3: Compare Results & Make Decision
├── Performance comparison
├── Implementation complexity assessment
├── Future scalability evaluation
└── Final platform selection
```

---

## ⚡ NautilusTrader POC

### **Minimal Viable Strategy**
```python
# poc/nautilus_q50_strategy.py
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data.tick import QuoteTick
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.objects import Quantity
import pandas as pd
import pickle

class Q50MinimalStrategy(Strategy):
    """
    Minimal Q50 strategy for NautilusTrader POC
    Uses pre-generated signals from our existing system
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Load our pre-generated signals
        self.signals_df = None
        self.current_position = 0
        self.signal_threshold = 0.6  # Probability threshold
        
        # Performance tracking
        self.trades_executed = 0
        self.last_signal_time = None
        
    def on_start(self):
        """Initialize strategy"""
        self.log.info("Starting Q50 Minimal Strategy")
        
        # Load our Q50 signals
        try:
            self.signals_df = pd.read_pickle('data3/macro_features.pkl')
            self.log.info(f"Loaded {len(self.signals_df)} Q50 signals")
        except Exception as e:
            self.log.error(f"Failed to load Q50 signals: {e}")
            return
        
        # Subscribe to market data
        self.subscribe_quote_ticks(self.instrument_id)
        
    def on_quote_tick(self, tick: QuoteTick):
        """Process market data and execute Q50 signals"""
        
        # Get current timestamp
        current_time = tick.ts_event
        
        # Find matching signal (simplified - use latest available)
        signal = self.get_current_signal(current_time)
        
        if signal is not None:
            self.process_q50_signal(signal, tick)
    
    def get_current_signal(self, timestamp):
        """Get Q50 signal for current timestamp"""
        
        if self.signals_df is None or len(self.signals_df) == 0:
            return None
        
        # Simplified: use most recent signal
        # In production, would match timestamp exactly
        try:
            latest_signal = self.signals_df.iloc[-1]
            
            # Check if we have required columns
            required_cols = ['q10', 'q50', 'q90', 'prob_up', 'side']
            if not all(col in self.signals_df.columns for col in required_cols):
                self.log.warning("Missing required signal columns")
                return None
                
            return latest_signal
            
        except Exception as e:
            self.log.error(f"Error getting signal: {e}")
            return None
    
    def process_q50_signal(self, signal, tick):
        """Process Q50 signal and execute trades"""
        
        try:
            # Extract signal components
            prob_up = signal.get('prob_up', 0.5)
            side = signal.get('side', -1)  # -1=hold, 0=sell, 1=buy
            q50 = signal.get('q50', 0)
            
            # Determine action based on our Q50 logic
            should_buy = (side == 1 and prob_up > self.signal_threshold)
            should_sell = (side == 0 and (1 - prob_up) > self.signal_threshold)
            should_hold = (side == -1)
            
            # Execute trades
            if should_buy and self.current_position <= 0:
                self.execute_buy_order(signal, tick)
                
            elif should_sell and self.current_position >= 0:
                self.execute_sell_order(signal, tick)
                
            elif should_hold and self.current_position != 0:
                self.close_position(tick)
                
        except Exception as e:
            self.log.error(f"Error processing signal: {e}")
    
    def execute_buy_order(self, signal, tick):
        """Execute buy order based on Q50 signal"""
        
        # Calculate position size (simplified Kelly sizing)
        position_size = self.calculate_position_size(signal)
        
        if position_size > 0:
            order = MarketOrder(
                trader_id=self.trader_id,
                strategy_id=self.strategy_id,
                instrument_id=self.instrument_id,
                order_side=OrderSide.BUY,
                quantity=Quantity(position_size, precision=4),
                time_in_force=TimeInForce.IOC,
            )
            
            self.submit_order(order)
            self.trades_executed += 1
            self.log.info(f"Executed BUY order: size={position_size}, signal_strength={signal.get('q50', 0):.4f}")
    
    def execute_sell_order(self, signal, tick):
        """Execute sell order based on Q50 signal"""
        
        position_size = self.calculate_position_size(signal)
        
        if position_size > 0:
            order = MarketOrder(
                trader_id=self.trader_id,
                strategy_id=self.strategy_id,
                instrument_id=self.instrument_id,
                order_side=OrderSide.SELL,
                quantity=Quantity(position_size, precision=4),
                time_in_force=TimeInForce.IOC,
            )
            
            self.submit_order(order)
            self.trades_executed += 1
            self.log.info(f"Executed SELL order: size={position_size}, signal_strength={signal.get('q50', 0):.4f}")
    
    def calculate_position_size(self, signal):
        """Calculate position size using simplified Kelly sizing"""
        
        try:
            # Get signal strength
            q50 = abs(signal.get('q50', 0))
            vol_risk = signal.get('vol_risk', 0.01)
            
            # Simplified Kelly calculation
            base_size = 0.1  # 10% of capital
            signal_multiplier = min(q50 * 100, 2.0)  # Cap at 2x
            vol_adjustment = 1.0 / (1.0 + vol_risk * 10)  # Reduce size in high vol
            
            position_size = base_size * signal_multiplier * vol_adjustment
            
            # Cap position size
            return min(position_size, 0.5)  # Max 50% of capital
            
        except Exception as e:
            self.log.error(f"Error calculating position size: {e}")
            return 0.1  # Default size
    
    def close_position(self, tick):
        """Close current position"""
        
        if self.current_position != 0:
            side = OrderSide.SELL if self.current_position > 0 else OrderSide.BUY
            
            order = MarketOrder(
                trader_id=self.trader_id,
                strategy_id=self.strategy_id,
                instrument_id=self.instrument_id,
                order_side=side,
                quantity=Quantity(abs(self.current_position), precision=4),
                time_in_force=TimeInForce.IOC,
            )
            
            self.submit_order(order)
            self.log.info(f"Closed position: {self.current_position}")
    
    def on_order_filled(self, event):
        """Handle order fill events"""
        
        # Update position tracking
        if event.order_side == OrderSide.BUY:
            self.current_position += float(event.last_qty)
        else:
            self.current_position -= float(event.last_qty)
            
        self.log.info(f"Order filled: {event.order_side}, new position: {self.current_position}")
    
    def on_stop(self):
        """Strategy shutdown"""
        self.log.info(f"Q50 Strategy stopped. Trades executed: {self.trades_executed}")
```

### **NautilusTrader Configuration**
```python
# poc/nautilus_config.py
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.config import StrategyConfig

# Strategy configuration
class Q50StrategyConfig(StrategyConfig):
    """Configuration for Q50 strategy"""
    
    instrument_id: str = "BTCUSDT.BINANCE"
    signal_threshold: float = 0.6
    max_position_size: float = 0.5
    base_position_size: float = 0.1

# Trading node configuration
config = TradingNodeConfig(
    trader_id="Q50_TRADER",
    strategies=[
        {
            "strategy_path": "poc.nautilus_q50_strategy:Q50MinimalStrategy",
            "config_path": "poc.nautilus_config:Q50StrategyConfig",
        }
    ],
    data_engine={
        "qsize": 100000,
        "time_bars_build_with_no_updates": False,
        "time_bars_timestamp_on_close": True,
        "validate_data_sequence": True,
    },
    risk_engine={
        "bypass": False,
        "max_order_submit_rate": "100/00:00:01",
        "max_order_modify_rate": "100/00:00:01",
        "max_notional_per_order": {"USD": 1_000_000},
    },
    exec_engine={
        "reconciliation": True,
        "reconciliation_lookback_mins": 1440,
        "snapshot_orders": True,
        "snapshot_positions": True,
    },
    adapters=[
        {
            "adapter": "BinanceSpotDataClient",
            "config": {
                "api_key": None,  # Paper trading
                "api_secret": None,
                "testnet": True,
            }
        }
    ]
)
```

---

## 🤖 Hummingbot POC

### **External Signal Provider Approach**
```python
# poc/hummingbot_signal_provider.py
import asyncio
import json
import pandas as pd
from datetime import datetime
import websockets
import logging

class Q50HummingbotSignalProvider:
    """
    External service that generates Q50 signals and sends them to Hummingbot
    Uses WebSocket or file-based communication
    """
    
    def __init__(self):
        self.signals_df = None
        self.last_signal_time = None
        self.signal_interval = 60  # 1 minute
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Load Q50 signals and initialize"""
        
        try:
            self.signals_df = pd.read_pickle('data3/macro_features.pkl')
            self.logger.info(f"Loaded {len(self.signals_df)} Q50 signals")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load signals: {e}")
            return False
    
    async def run_signal_generation(self):
        """Main signal generation loop"""
        
        if not await self.initialize():
            return
        
        self.logger.info("Starting Q50 signal generation for Hummingbot")
        
        while True:
            try:
                # Generate current signal
                signal = self.generate_current_signal()
                
                if signal:
                    # Send to Hummingbot via file (simple approach)
                    await self.send_signal_to_hummingbot(signal)
                    
                    self.logger.info(f"Sent signal: {signal['action']} - strength: {signal['strength']:.3f}")
                
                # Wait for next interval
                await asyncio.sleep(self.signal_interval)
                
            except Exception as e:
                self.logger.error(f"Error in signal generation: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    def generate_current_signal(self):
        """Generate Q50 signal for current time"""
        
        if self.signals_df is None or len(self.signals_df) == 0:
            return None
        
        try:
            # Get latest signal (simplified)
            latest_signal = self.signals_df.iloc[-1]
            
            # Extract components
            q10 = latest_signal.get('q10', 0)
            q50 = latest_signal.get('q50', 0)
            q90 = latest_signal.get('q90', 0)
            prob_up = latest_signal.get('prob_up', 0.5)
            side = latest_signal.get('side', -1)
            
            # Convert to Hummingbot format
            signal = {
                'timestamp': datetime.now().isoformat(),
                'symbol': 'BTCUSDT',
                'action': self.convert_side_to_action(side, prob_up),
                'strength': abs(q50),
                'confidence': max(prob_up, 1 - prob_up),
                'position_size': self.calculate_position_size(latest_signal),
                'metadata': {
                    'q10': q10,
                    'q50': q50,
                    'q90': q90,
                    'prob_up': prob_up,
                    'regime': latest_signal.get('regime_volatility', 'unknown')
                }
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return None
    
    def convert_side_to_action(self, side, prob_up):
        """Convert our side signal to Hummingbot action"""
        
        if side == 1 and prob_up > 0.6:
            return 'BUY'
        elif side == 0 and prob_up < 0.4:
            return 'SELL'
        else:
            return 'HOLD'
    
    def calculate_position_size(self, signal):
        """Calculate position size for Hummingbot"""
        
        try:
            q50 = abs(signal.get('q50', 0))
            vol_risk = signal.get('vol_risk', 0.01)
            
            # Simplified Kelly sizing
            base_size = 0.1
            signal_multiplier = min(q50 * 100, 2.0)
            vol_adjustment = 1.0 / (1.0 + vol_risk * 10)
            
            return min(base_size * signal_multiplier * vol_adjustment, 0.5)
            
        except:
            return 0.1
    
    async def send_signal_to_hummingbot(self, signal):
        """Send signal to Hummingbot via file communication"""
        
        # Simple file-based communication
        signal_file = 'hummingbot_signals/latest_signal.json'
        
        try:
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(signal_file), exist_ok=True)
            
            # Write signal to file
            with open(signal_file, 'w') as f:
                json.dump(signal, f, indent=2)
                
            # Also maintain signal history
            history_file = 'hummingbot_signals/signal_history.jsonl'
            with open(history_file, 'a') as f:
                f.write(json.dumps(signal) + '\n')
                
        except Exception as e:
            self.logger.error(f"Error sending signal: {e}")

# Run the signal provider
async def main():
    provider = Q50HummingbotSignalProvider()
    await provider.run_signal_generation()

if __name__ == "__main__":
    asyncio.run(main())
```

### **Hummingbot Strategy Modification**
```python
# hummingbot/strategy/q50_external_signal/q50_external_signal.py
import json
import os
from decimal import Decimal
from typing import List, Dict
from hummingbot.strategy.strategy_py_base import StrategyPyBase
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.limit_order import LimitOrder

class Q50ExternalSignalStrategy(StrategyPyBase):
    """
    Hummingbot strategy that consumes external Q50 signals
    """
    
    def __init__(self, config_map):
        super().__init__(config_map)
        
        self.signal_file = 'hummingbot_signals/latest_signal.json'
        self.last_signal_time = None
        self.current_signal = None
        
        # Configuration
        self.min_signal_strength = 0.001
        self.max_position_size = 0.5
        
    def tick(self, timestamp: float):
        """Main strategy tick - check for new signals and execute"""
        
        # Load latest signal
        new_signal = self.load_latest_signal()
        
        if new_signal and self.is_signal_fresh(new_signal):
            self.current_signal = new_signal
            self.execute_signal(new_signal)
    
    def load_latest_signal(self):
        """Load latest Q50 signal from file"""
        
        try:
            if os.path.exists(self.signal_file):
                with open(self.signal_file, 'r') as f:
                    signal = json.load(f)
                return signal
        except Exception as e:
            self.logger().error(f"Error loading signal: {e}")
        
        return None
    
    def is_signal_fresh(self, signal):
        """Check if signal is fresh and different from last"""
        
        signal_time = signal.get('timestamp')
        
        if signal_time != self.last_signal_time:
            self.last_signal_time = signal_time
            return True
        
        return False
    
    def execute_signal(self, signal):
        """Execute trading action based on Q50 signal"""
        
        action = signal.get('action', 'HOLD')
        strength = signal.get('strength', 0)
        position_size = signal.get('position_size', 0.1)
        
        # Check minimum signal strength
        if strength < self.min_signal_strength:
            return
        
        # Execute based on action
        if action == 'BUY':
            self.execute_buy_signal(signal)
        elif action == 'SELL':
            self.execute_sell_signal(signal)
        elif action == 'HOLD':
            self.close_positions()
    
    def execute_buy_signal(self, signal):
        """Execute buy order"""
        
        trading_pair = self.trading_pair
        position_size = min(signal.get('position_size', 0.1), self.max_position_size)
        
        # Get current price
        current_price = self.get_mid_price()
        
        if current_price and position_size > 0:
            # Calculate order amount
            order_amount = Decimal(str(position_size))
            
            # Create buy order
            buy_order = LimitOrder(
                client_order_id="",
                trading_pair=trading_pair,
                is_buy=True,
                base_currency=trading_pair.split('-')[0],
                quote_currency=trading_pair.split('-')[1],
                amount=order_amount,
                price=current_price * Decimal("1.001")  # Slight premium
            )
            
            self.buy_with_specific_market(
                trading_pair=trading_pair,
                amount=order_amount,
                order_type=OrderType.LIMIT,
                price=buy_order.price
            )
            
            self.logger().info(f"Executed Q50 BUY: {order_amount} at {buy_order.price}")
    
    def execute_sell_signal(self, signal):
        """Execute sell order"""
        
        trading_pair = self.trading_pair
        position_size = min(signal.get('position_size', 0.1), self.max_position_size)
        
        current_price = self.get_mid_price()
        
        if current_price and position_size > 0:
            order_amount = Decimal(str(position_size))
            
            self.sell_with_specific_market(
                trading_pair=trading_pair,
                amount=order_amount,
                order_type=OrderType.LIMIT,
                price=current_price * Decimal("0.999")  # Slight discount
            )
            
            self.logger().info(f"Executed Q50 SELL: {order_amount} at {current_price * Decimal('0.999')}")
    
    def close_positions(self):
        """Close all open positions"""
        
        # Get current positions
        active_orders = self.get_active_orders(self.trading_pair)
        
        # Cancel all active orders
        for order in active_orders:
            self.cancel_order(self.trading_pair, order.client_order_id)
```

---

## POC Evaluation Framework

### **Performance Metrics**
```python
# poc/evaluation_framework.py
class POCEvaluator:
    """Evaluate and compare both POC implementations"""
    
    def __init__(self):
        self.metrics = {
            'nautilus': {},
            'hummingbot': {}
        }
    
    def evaluate_implementation_complexity(self, platform):
        """Rate implementation complexity (1-10, lower is better)"""
        
        complexity_factors = {
            'setup_difficulty': 0,      # How hard to get running
            'code_complexity': 0,       # Lines of code, complexity
            'integration_points': 0,    # Number of integration points
            'debugging_difficulty': 0,  # How easy to debug issues
            'documentation_quality': 0  # Quality of available docs
        }
        
        return complexity_factors
    
    def evaluate_performance(self, platform):
        """Evaluate trading performance"""
        
        performance_metrics = {
            'signal_latency': 0,        # Time from signal to execution
            'execution_accuracy': 0,    # How well signals are executed
            'system_stability': 0,      # Crashes, errors, reliability
            'resource_usage': 0,        # CPU, memory usage
            'scalability': 0           # Can it handle multiple assets?
        }
        
        return performance_metrics
    
    def evaluate_future_potential(self, platform):
        """Assess long-term strategic value"""
        
        future_factors = {
            'rd_agent_compatibility': 0,    # Can integrate with RD-Agent
            'multi_asset_support': 0,       # Multiple trading pairs
            'advanced_features': 0,         # Room for sophisticated features
            'community_ecosystem': 0,       # Community and plugin support
            'professional_adoption': 0      # Used by professional traders
        }
        
        return future_factors
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        
        report = {
            'implementation_winner': None,
            'performance_winner': None,
            'future_potential_winner': None,
            'overall_recommendation': None,
            'detailed_analysis': self.metrics
        }
        
        return report
```

---

## 🎯 Success Criteria

### **POC Must Demonstrate:**

1. **Basic Functionality** ✅
   - Load our Q50 signals successfully
   - Execute trades based on signals
   - Handle position management

2. **Performance Validation** ✅
   - Signal-to-execution latency < 5 seconds
   - Accurate signal interpretation
   - Stable operation for 24+ hours

3. **Integration Quality** ✅
   - Clean code architecture
   - Error handling and logging
   - Configuration management

4. **Scalability Indicators** ✅
   - Can handle multiple signals per minute
   - Memory usage remains stable
   - Easy to extend with new features

---

## Implementation Timeline

### **Week 1: Setup & Basic Implementation**
- [ ] **Day 1-2**: NautilusTrader installation and basic strategy
- [ ] **Day 3-4**: Hummingbot external signal provider
- [ ] **Day 5-7**: Basic signal integration for both platforms

### **Week 2: Testing & Validation**
- [ ] **Day 8-10**: Paper trading validation
- [ ] **Day 11-12**: Performance testing and optimization
- [ ] **Day 13-14**: Documentation and comparison analysis

### **Week 3: Decision & Next Steps**
- [ ] **Day 15-16**: Comprehensive evaluation and comparison
- [ ] **Day 17-18**: Platform selection and roadmap creation
- [ ] **Day 19-21**: Begin full implementation on selected platform

---

## 🎯 Expected Outcome

**By end of Week 3, we'll have:**

1. **Data-driven platform selection** based on actual implementation experience
2. **Working prototype** of our Q50 system integrated with chosen platform
3. **Clear roadmap** for full production implementation
4. **Risk mitigation** through validated fallback option

This approach gives us the confidence to make the right long-term decision while maintaining momentum on integration! 🚀