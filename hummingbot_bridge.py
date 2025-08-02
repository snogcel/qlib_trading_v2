import pandas as pd
import numpy as np
import json
import time
from typing import Dict, Tuple
import paho.mqtt.client as mqtt
from datetime import datetime

class QlibToHummingbotBridge:
    """
    Bridge between your qlib quantile predictions and Hummingbot AI livestream format
    """
    
    def __init__(self, mqtt_broker="localhost", mqtt_port=1883, topic_prefix="hbot/predictions"):
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.topic_prefix = topic_prefix
        self.client = None
        self._setup_mqtt()
    
    def _setup_mqtt(self):
        """Initialize MQTT client"""
        self.client = mqtt.Client()
        self.client.connect(self.mqtt_broker, self.mqtt_port, 60)
        self.client.loop_start()
    
    def quantiles_to_probabilities(self, q10: float, q50: float, q90: float) -> Tuple[float, float, float]:
        """
        Convert your quantile predictions to [short_prob, neutral_prob, long_prob]
        
        Based on your prob_up_piecewise logic from ppo_sweep_optuna_tuned.py
        """
        # Calculate probability of upside movement (price > 0)
        if q90 <= 0:
            prob_up = 0.0
        elif q10 >= 0:
            prob_up = 1.0
        elif q10 < 0 <= q50:
            # 0 lies between q10 and q50
            cdf0 = 0.10 + 0.40 * (0 - q10) / (q50 - q10)
            prob_up = 1 - cdf0
        else:
            # 0 lies between q50 and q90
            cdf0 = 0.50 + 0.40 * (0 - q50) / (q90 - q50)
            prob_up = 1 - cdf0
        
        prob_down = 1 - prob_up
        
        # Use spread and confidence to determine neutral probability
        spread = q90 - q10
        spread_normalized = min(spread / 0.02, 1.0)  # normalize by typical spread
        
        # Higher spread = more uncertainty = higher neutral probability
        neutral_weight = spread_normalized * 0.3  # max 30% neutral
        
        # Redistribute probabilities
        prob_up_adj = prob_up * (1 - neutral_weight)
        prob_down_adj = prob_down * (1 - neutral_weight)
        prob_neutral = neutral_weight
        
        return prob_down_adj, prob_neutral, prob_up_adj
    
    def calculate_target_pct(self, q10: float, q50: float, q90: float, tier_confidence: float) -> float:
        """
        Calculate expected price movement percentage for position sizing
        """
        # Use median prediction as base target
        base_target = abs(q50)
        
        # Adjust by confidence - higher confidence = larger position
        confidence_multiplier = 0.5 + (tier_confidence / 10.0) * 1.5  # 0.5x to 2x
        
        # Consider spread for risk adjustment
        spread = q90 - q10
        risk_adjustment = min(spread / 0.02, 2.0)  # cap at 2x
        
        target_pct = base_target * confidence_multiplier / risk_adjustment
        
        return float(np.clip(target_pct, 0.001, 0.05))  # 0.1% to 5%
    
    def publish_signal(self, trading_pair: str, q10: float, q50: float, q90: float, 
                      tier_confidence: float, additional_features: Dict = None):
        """
        Publish ML signal in Hummingbot format
        """
        # Convert to probabilities
        short_prob, neutral_prob, long_prob = self.quantiles_to_probabilities(q10, q50, q90)
        
        # Calculate target percentage
        target_pct = self.calculate_target_pct(q10, q50, q90, tier_confidence)
        
        # Create signal payload
        signal = {
            "timestamp": datetime.now().isoformat(),
            "probabilities": [short_prob, neutral_prob, long_prob],
            "target_pct": target_pct,
            "confidence": tier_confidence,
            "raw_quantiles": {
                "q10": q10,
                "q50": q50,
                "q90": q90
            }
        }
        
        # Add additional features if provided
        if additional_features:
            signal["features"] = additional_features
        
        # Publish to MQTT
        normalized_pair = trading_pair.replace("-", "_").lower()
        topic = f"{self.topic_prefix}/{normalized_pair}/ML_SIGNALS"
        
        self.client.publish(topic, json.dumps(signal))
        print(f"Published signal for {trading_pair}: {signal}")
    
    def close(self):
        """Clean up MQTT connection"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()

# Example usage with your existing pipeline
def integrate_with_existing_pipeline():
    """
    Example of how to integrate with your existing prediction pipeline
    """
    bridge = QlibToHummingbotBridge()
    
    # Load your latest predictions (from your existing pipeline)
    # This would be called in real-time as new predictions are generated
    df_latest = pd.read_csv("df_all_macro_analysis.csv", index_col=[0, 1])
    
    # Get latest prediction for BTCUSDT
    latest_row = df_latest.loc[("BTCUSDT", df_latest.index.get_level_values(1).max())]
    
    # Extract features
    additional_features = {
        "fg_index": latest_row["$fg_index"],
        "btc_dom": latest_row["$btc_dom"],
        "momentum_5": latest_row["$momentum_5"],
        "momentum_10": latest_row["$momentum_10"],
        "vol_scaled": latest_row["vol_scaled"],
        "signal_score": latest_row["signal_score"],
        "spread_score": latest_row["spread_score"]
    }
    
    # Publish signal
    bridge.publish_signal(
        trading_pair="BTCUSDT",
        q10=latest_row["q10"],
        q50=latest_row["q50"],
        q90=latest_row["q90"],
        tier_confidence=latest_row["tier_confidence"],
        additional_features=additional_features
    )
    
    bridge.close()

if __name__ == "__main__":
    integrate_with_existing_pipeline()