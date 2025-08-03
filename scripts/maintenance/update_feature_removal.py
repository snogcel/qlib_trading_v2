#!/usr/bin/env python3
"""
Updated feature removal script that keeps essential columns for backtesting
"""

# Remove these - they're either redundant or harmful:
features_to_remove = [
    'signal_rel_clipped',   # Identical to signal_rel (1.0000 correlation)
    'signal_sigmoid',       # Redundant with signal_rel (0.9993 correlation)
    'signal_score',         # Worse than components
    'spread_rel',           # Weak and worse than raw spread
    'spread_rel_clipped',   # Redundant
    'spread_tanh',          # Weak
    'spread_sigmoid',       # Weak  
    'spread_score',         # HARMFUL - negative correlation!
    'spread_tier',          # Very weak (0.0010)
    # 'tier_confidence',    # KEEP - needed for position sizing
]

# Essential columns to ALWAYS keep for backtesting:
essential_columns = [
    'instrument',
    'datetime', 
    'q10', 'q50', 'q90',           # Core predictions
    'truth',                       # Actual returns - ESSENTIAL for backtesting
    'signal_tier',                 # Used for confidence mapping
    'signal_thresh_adaptive',      # Used for signal thresholding  
    'spread_thresh',               # Used for adaptive thresholding
    'side',                        # PPO side encoding
    'prob_up',                     # Probability calculations
    'spread',                      # Raw spread (better than derived versions)
]

print("Features to remove:")
for feature in features_to_remove:
    print(f"  - {feature}")

print(f"\nEssential columns to keep:")
for col in essential_columns:
    print(f"  - {col}")

print(f"\nMake sure your data processing pipeline keeps these essential columns!")
print(f"The 'truth' column is particularly important - it contains actual returns for backtesting.")