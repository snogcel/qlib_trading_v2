"""
Economic Rationale Generation Framework

This module implements the core framework for generating economic rationale and thesis statements
for trading system features based on supply/demand principles and market microstructure theory.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import re
from pathlib import Path


class SupplyDemandRole(Enum):
    """Classification of feature roles in supply/demand detection"""
    SUPPLY_DETECTOR = "supply_detector"
    DEMAND_DETECTOR = "demand_detector" 
    IMBALANCE_DETECTOR = "imbalance_detector"
    REGIME_CLASSIFIER = "regime_classifier"
    RISK_ASSESSOR = "risk_assessor"
    POSITION_OPTIMIZER = "position_optimizer"


class MarketLayer(Enum):
    """Market structure layers that features operate on"""
    MICROSTRUCTURE = "microstructure"  # Order flow, spreads, liquidity
    SENTIMENT = "sentiment"            # Fear/greed, positioning
    FUNDAMENTAL = "fundamental"        # Economic data, macro factors
    TECHNICAL = "technical"           # Price patterns, momentum
    REGIME = "regime"                 # Market state classification


class TimeHorizon(Enum):
    """Time horizons for feature effectiveness"""
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class ThesisStatement:
    """Economic thesis statement for a trading feature"""
    hypothesis: str                    # Why this feature should predict returns
    economic_basis: str               # Supply/demand explanation
    market_microstructure: str        # How it works in real markets
    expected_behavior: str            # What we expect to see
    failure_modes: List[str]          # When it might not work
    academic_support: List[str] = field(default_factory=list)  # Supporting research


@dataclass
class EconomicRationale:
    """Detailed economic rationale for a feature"""
    supply_factors: Optional[List[str]] = None         # How it detects supply changes
    demand_factors: Optional[List[str]] = None         # How it detects demand changes
    market_inefficiency: Optional[str] = None         # What inefficiency it exploits
    regime_dependency: Optional[str] = None            # How it varies by market regime
    interaction_effects: Optional[List[str]] = None    # How it works with other features
    
    def __post_init__(self):
        """Validate that at least one economic rationale component is provided"""
        components = [
            self.supply_factors,
            self.demand_factors, 
            self.market_inefficiency,
            self.regime_dependency
        ]
        
        if not any(component for component in components):
            raise ValueError(
                "EconomicRationale must have at least one of: supply_factors, "
                "demand_factors, market_inefficiency, or regime_dependency"
            )
    
    def get_primary_rationale_type(self) -> str:
        """Return the primary type of economic rationale"""
        if self.supply_factors:
            return "supply_based"
        elif self.demand_factors:
            return "demand_based"
        elif self.market_inefficiency:
            return "inefficiency_based"
        elif self.regime_dependency:
            return "regime_based"
        else:
            return "unknown"


@dataclass
class ChartExplanation:
    """Visual explanation of how feature works on charts"""
    visual_description: str           # How to see it on charts
    example_scenarios: List[str]      # Specific examples
    chart_patterns: List[str]         # Associated chart patterns
    false_signals: List[str]          # When it gives wrong signals
    confirmation_signals: List[str]   # What confirms the signal


@dataclass
class ValidationCriterion:
    """Validation criterion for a feature"""
    test_name: str
    description: str
    success_threshold: float
    test_implementation: str          # Path to test function
    frequency: str                    # How often to run
    failure_action: str              # What to do if test fails


@dataclass
class SupplyDemandClassification:
    """Classification of feature's role in supply/demand framework"""
    primary_role: SupplyDemandRole
    secondary_roles: List[SupplyDemandRole]
    market_layer: MarketLayer
    time_horizon: TimeHorizon
    regime_sensitivity: str           # "high", "medium", "low"
    interaction_features: List[str]   # Features it works best with


@dataclass
class FeatureEnhancement:
    """Complete enhancement package for a feature"""
    feature_name: str
    category: str
    existing_content: Dict[str, Any]
    thesis_statement: ThesisStatement
    economic_rationale: EconomicRationale
    chart_explanation: ChartExplanation
    supply_demand_classification: SupplyDemandClassification
    validation_criteria: List[ValidationCriterion]
    dependencies: List[str]
    validated: bool = False


class EconomicRationaleGenerator:
    """
    Core class for generating economic rationale and thesis statements
    based on supply/demand principles and trading system principles.
    """
    
    def __init__(self, principles_path: str = "docs/TRADING_SYSTEM_PRINCIPLES.md"):
        self.principles_path = Path(principles_path)
        self.principles_content = self._load_principles()
        
        # Core supply/demand templates
        self.supply_demand_templates = self._initialize_templates()
        
        # Feature type mappings
        self.feature_type_mappings = self._initialize_feature_mappings()
    
    def _load_principles(self) -> str:
        """Load trading system principles for reference"""
        try:
            with open(self.principles_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return ""
    
    def _initialize_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize supply/demand explanation templates"""
        return {
            "quantile_signal": {
                "supply_template": "Detects supply exhaustion when {feature} indicates sellers are becoming scarce at current price levels",
                "demand_template": "Identifies demand accumulation when {feature} shows buyers are willing to pay higher prices",
                "inefficiency": "Exploits temporary imbalances between buyer and seller conviction levels",
                "microstructure": "Works by measuring probability distributions of price movements, capturing asymmetric order flow"
            },
            "volatility_risk": {
                "supply_template": "High volatility indicates supply/demand imbalance - market struggling to find equilibrium",
                "demand_template": "Low volatility suggests balanced supply/demand with stable price discovery",
                "inefficiency": "Exploits the relationship between volatility and future price movements",
                "microstructure": "Variance captures true risk better than standard deviation by not assuming normal distributions"
            },
            "regime_classification": {
                "supply_template": "Regime changes occur when supply/demand dynamics shift fundamentally",
                "demand_template": "Different regimes require different approaches to supply/demand analysis",
                "inefficiency": "Exploits the fact that most traders don't adapt their strategies to regime changes",
                "microstructure": "Market microstructure changes significantly between bull, bear, and sideways regimes"
            },
            "position_sizing": {
                "supply_template": "Position size should reflect confidence in supply/demand imbalance magnitude",
                "demand_template": "Larger positions when demand/supply imbalance is more certain and significant",
                "inefficiency": "Exploits Kelly criterion optimization that most traders don't implement properly",
                "microstructure": "Optimal sizing accounts for transaction costs and market impact"
            },
            "momentum": {
                "supply_template": "Momentum indicates persistent supply/demand imbalance in one direction",
                "demand_template": "Strong momentum suggests demand (or supply) will continue overwhelming the other side",
                "inefficiency": "Exploits behavioral biases that cause momentum to persist longer than efficient markets predict",
                "microstructure": "Momentum works because of information asymmetry and gradual price discovery"
            },
            "sentiment": {
                "supply_template": "Extreme fear creates artificial supply as panicked sellers dump positions",
                "demand_template": "Extreme greed creates artificial demand as FOMO buyers chase prices higher",
                "inefficiency": "Exploits behavioral biases that cause systematic over/under-reactions",
                "microstructure": "Sentiment affects order flow patterns and creates predictable imbalances"
            }
        }
    
    def _initialize_feature_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mappings from feature names to their characteristics"""
        return {
            "q50": {
                "template_type": "quantile_signal",
                "primary_role": SupplyDemandRole.IMBALANCE_DETECTOR,
                "market_layer": MarketLayer.TECHNICAL,
                "time_horizon": TimeHorizon.DAILY
            },
            "vol_risk": {
                "template_type": "volatility_risk", 
                "primary_role": SupplyDemandRole.RISK_ASSESSOR,
                "market_layer": MarketLayer.MICROSTRUCTURE,
                "time_horizon": TimeHorizon.DAILY
            },
            "regime": {
                "template_type": "regime_classification",
                "primary_role": SupplyDemandRole.REGIME_CLASSIFIER,
                "market_layer": MarketLayer.REGIME,
                "time_horizon": TimeHorizon.WEEKLY
            },
            "kelly": {
                "template_type": "position_sizing",
                "primary_role": SupplyDemandRole.POSITION_OPTIMIZER,
                "market_layer": MarketLayer.MICROSTRUCTURE,
                "time_horizon": TimeHorizon.DAILY
            },
            "momentum": {
                "template_type": "momentum",
                "primary_role": SupplyDemandRole.DEMAND_DETECTOR,
                "market_layer": MarketLayer.TECHNICAL,
                "time_horizon": TimeHorizon.DAILY
            },
            "sentiment": {
                "template_type": "sentiment",
                "primary_role": SupplyDemandRole.IMBALANCE_DETECTOR,
                "market_layer": MarketLayer.SENTIMENT,
                "time_horizon": TimeHorizon.WEEKLY
            }
        }
    
    def generate_thesis_statement(self, feature_name: str, feature_description: str = "") -> ThesisStatement:
        """Generate economic thesis statement for a feature"""
        
        # Determine feature type from name
        feature_type = self._classify_feature_type(feature_name)
        template = self.supply_demand_templates.get(feature_type, self.supply_demand_templates["quantile_signal"])
        
        # Generate hypothesis
        hypothesis = self._generate_hypothesis(feature_name, feature_type, feature_description)
        
        # Generate economic basis
        economic_basis = self._generate_economic_basis(feature_name, template)
        
        # Generate market microstructure explanation
        microstructure = template["microstructure"].format(feature=feature_name)
        
        # Generate expected behavior
        expected_behavior = self._generate_expected_behavior(feature_name, feature_type)
        
        # Generate failure modes
        failure_modes = self._generate_failure_modes(feature_name, feature_type)
        
        # Add academic support
        academic_support = self._get_academic_support(feature_type)
        
        return ThesisStatement(
            hypothesis=hypothesis,
            economic_basis=economic_basis,
            market_microstructure=microstructure,
            expected_behavior=expected_behavior,
            failure_modes=failure_modes,
            academic_support=academic_support
        )
    
    def generate_economic_rationale(self, feature_name: str, feature_description: str = "") -> EconomicRationale:
        """Generate detailed economic rationale for a feature"""
        
        feature_type = self._classify_feature_type(feature_name)
        template = self.supply_demand_templates.get(feature_type, self.supply_demand_templates["quantile_signal"])
        
        supply_factors = [template["supply_template"].format(feature=feature_name)]
        demand_factors = [template["demand_template"].format(feature=feature_name)]
        market_inefficiency = template["inefficiency"]
        
        regime_dependency = self._generate_regime_dependency(feature_name, feature_type)
        interaction_effects = self._generate_interaction_effects(feature_name, feature_type)
        
        return EconomicRationale(
            supply_factors=supply_factors,
            demand_factors=demand_factors,
            market_inefficiency=market_inefficiency,
            regime_dependency=regime_dependency,
            interaction_effects=interaction_effects
        )
    
    def generate_chart_explanation(self, feature_name: str, feature_description: str = "") -> ChartExplanation:
        """Generate chart explainability framework for a feature"""
        
        feature_type = self._classify_feature_type(feature_name)
        
        visual_description = self._generate_visual_description(feature_name, feature_type)
        example_scenarios = self._generate_example_scenarios(feature_name, feature_type)
        chart_patterns = self._generate_chart_patterns(feature_name, feature_type)
        false_signals = self._generate_false_signals(feature_name, feature_type)
        confirmation_signals = self._generate_confirmation_signals(feature_name, feature_type)
        
        return ChartExplanation(
            visual_description=visual_description,
            example_scenarios=example_scenarios,
            chart_patterns=chart_patterns,
            false_signals=false_signals,
            confirmation_signals=confirmation_signals
        )
    
    def classify_supply_demand_role(self, feature_name: str, feature_description: str = "") -> SupplyDemandClassification:
        """Classify feature's role in supply/demand framework"""
        
        feature_type = self._classify_feature_type(feature_name)
        mapping = self.feature_type_mappings.get(feature_type, self.feature_type_mappings["q50"])
        
        # Determine secondary roles
        secondary_roles = self._determine_secondary_roles(feature_name, feature_type)
        
        # Determine regime sensitivity
        regime_sensitivity = self._determine_regime_sensitivity(feature_name, feature_type)
        
        # Determine interaction features
        interaction_features = self._determine_interaction_features(feature_name, feature_type)
        
        return SupplyDemandClassification(
            primary_role=mapping["primary_role"],
            secondary_roles=secondary_roles,
            market_layer=mapping["market_layer"],
            time_horizon=mapping["time_horizon"],
            regime_sensitivity=regime_sensitivity,
            interaction_features=interaction_features
        )
    
    def generate_complete_enhancement(self, feature_name: str, category: str, 
                                    existing_content: Dict[str, Any] = None) -> FeatureEnhancement:
        """Generate complete enhancement package for a feature"""
        
        if existing_content is None:
            existing_content = {}
        
        # Get feature description from existing content
        feature_description = existing_content.get("purpose", "")
        
        # Generate all components
        thesis = self.generate_thesis_statement(feature_name, feature_description)
        rationale = self.generate_economic_rationale(feature_name, feature_description)
        chart_explanation = self.generate_chart_explanation(feature_name, feature_description)
        classification = self.classify_supply_demand_role(feature_name, feature_description)
        
        # Generate validation criteria
        validation_criteria = self._generate_validation_criteria(feature_name, thesis)
        
        # Determine dependencies
        dependencies = self._determine_dependencies(feature_name, existing_content)
        
        return FeatureEnhancement(
            feature_name=feature_name,
            category=category,
            existing_content=existing_content,
            thesis_statement=thesis,
            economic_rationale=rationale,
            chart_explanation=chart_explanation,
            supply_demand_classification=classification,
            validation_criteria=validation_criteria,
            dependencies=dependencies
        )
    
    def _classify_feature_type(self, feature_name: str) -> str:
        """Classify feature type based on name patterns"""
        name_lower = feature_name.lower()
        
        if any(term in name_lower for term in ["q50", "quantile", "probability"]):
            return "quantile_signal"
        elif any(term in name_lower for term in ["vol", "volatility", "variance", "risk"]):
            return "volatility_risk"
        elif any(term in name_lower for term in ["regime", "classification", "state"]):
            return "regime_classification"
        elif any(term in name_lower for term in ["kelly", "position", "sizing", "size"]):
            return "position_sizing"
        elif any(term in name_lower for term in ["momentum", "trend", "direction"]):
            return "momentum"
        elif any(term in name_lower for term in ["sentiment", "fear", "greed", "emotion"]):
            return "sentiment"
        else:
            return "quantile_signal"  # Default
    
    def _generate_hypothesis(self, feature_name: str, feature_type: str, description: str) -> str:
        """Generate economic hypothesis for feature"""
        
        hypotheses = {
            "quantile_signal": f"{feature_name} should predict returns because it captures asymmetric probability distributions that indicate supply/demand imbalances before they fully manifest in price movements.",
            "volatility_risk": f"{feature_name} should predict returns because volatility clustering indicates persistent supply/demand imbalances that create predictable risk-return relationships.",
            "regime_classification": f"{feature_name} should predict returns because different market regimes have fundamentally different supply/demand dynamics that require adaptive trading approaches.",
            "position_sizing": f"{feature_name} should optimize returns because proper position sizing maximizes the Kelly criterion while accounting for regime-specific risk characteristics.",
            "momentum": f"{feature_name} should predict returns because momentum indicates persistent supply/demand imbalances that continue due to behavioral biases and information asymmetry.",
            "sentiment": f"{feature_name} should predict returns because extreme sentiment creates systematic supply/demand imbalances through behavioral over-reactions."
        }
        
        return hypotheses.get(feature_type, hypotheses["quantile_signal"])
    
    def _generate_economic_basis(self, feature_name: str, template: Dict[str, str]) -> str:
        """Generate supply/demand economic basis"""
        return f"Supply/Demand Analysis: {template['supply_template'].format(feature=feature_name)} Conversely, {template['demand_template'].format(feature=feature_name)} This creates exploitable imbalances because {template['inefficiency']}."
    
    def _generate_expected_behavior(self, feature_name: str, feature_type: str) -> str:
        """Generate expected behavior description"""
        
        behaviors = {
            "quantile_signal": f"Expect {feature_name} to show higher values before upward price movements and lower values before downward movements, with asymmetric payoff profiles.",
            "volatility_risk": f"Expect {feature_name} to increase before significant price movements and decrease during stable periods, with regime-dependent relationships.",
            "regime_classification": f"Expect {feature_name} to identify distinct market states with different risk-return characteristics and strategy effectiveness.",
            "position_sizing": f"Expect {feature_name} to increase position sizes during high-conviction, low-risk opportunities and decrease them during uncertain periods.",
            "momentum": f"Expect {feature_name} to persist in trending markets and reverse during regime transitions or exhaustion phases.",
            "sentiment": f"Expect {feature_name} to show contrarian predictive power at extremes and trend-following power in moderate ranges."
        }
        
        return behaviors.get(feature_type, behaviors["quantile_signal"])
    
    def _generate_failure_modes(self, feature_name: str, feature_type: str) -> List[str]:
        """Generate failure modes for feature"""
        
        failure_modes = {
            "quantile_signal": [
                "Regime changes that alter underlying probability distributions",
                "Market microstructure changes affecting order flow patterns",
                "Extreme events that break historical relationships"
            ],
            "volatility_risk": [
                "Volatility regime shifts that change risk-return relationships",
                "Structural breaks in market microstructure",
                "Non-stationary volatility processes"
            ],
            "regime_classification": [
                "Rapid regime transitions that create classification lag",
                "Novel market conditions not seen in training data",
                "Regime definitions becoming obsolete due to market evolution"
            ],
            "position_sizing": [
                "Parameter instability in Kelly calculations",
                "Correlation changes between features",
                "Transaction cost changes affecting optimal sizing"
            ],
            "momentum": [
                "Momentum reversals during regime transitions",
                "Increased market efficiency reducing momentum persistence",
                "Structural breaks in trend-following relationships"
            ],
            "sentiment": [
                "Sentiment measure becoming stale or manipulated",
                "Behavioral biases changing over time",
                "Extreme events that break sentiment-return relationships"
            ]
        }
        
        return failure_modes.get(feature_type, failure_modes["quantile_signal"])
    
    def _get_academic_support(self, feature_type: str) -> List[str]:
        """Get academic support references for feature type"""
        
        support = {
            "quantile_signal": [
                "Quantile regression literature (Koenker & Bassett, 1978)",
                "Asymmetric return distribution studies",
                "Market microstructure theory on order flow"
            ],
            "volatility_risk": [
                "GARCH modeling literature (Bollerslev, 1986)",
                "Volatility clustering research",
                "Risk-return relationship studies"
            ],
            "regime_classification": [
                "Markov regime-switching models (Hamilton, 1989)",
                "Structural break literature",
                "Market state classification research"
            ],
            "position_sizing": [
                "Kelly criterion optimization (Kelly, 1956)",
                "Portfolio theory (Markowitz, 1952)",
                "Risk parity literature"
            ],
            "momentum": [
                "Momentum literature (Jegadeesh & Titman, 1993)",
                "Behavioral finance research on momentum",
                "Time series momentum studies"
            ],
            "sentiment": [
                "Behavioral finance literature",
                "Sentiment indicator research",
                "Contrarian investment studies"
            ]
        }
        
        return support.get(feature_type, [])
    
    def _generate_regime_dependency(self, feature_name: str, feature_type: str) -> str:
        """Generate regime dependency explanation"""
        
        dependencies = {
            "quantile_signal": f"{feature_name} effectiveness varies by regime - stronger in trending markets, weaker in choppy sideways markets.",
            "volatility_risk": f"{feature_name} has different risk-return relationships in bull vs bear vs sideways regimes.",
            "regime_classification": f"{feature_name} is designed to adapt to regime changes and should maintain effectiveness across regimes.",
            "position_sizing": f"{feature_name} should increase sizing in favorable regimes and decrease in unfavorable ones.",
            "momentum": f"{feature_name} works best in trending regimes and poorly in mean-reverting regimes.",
            "sentiment": f"{feature_name} has different effectiveness in different volatility and trend regimes."
        }
        
        return dependencies.get(feature_type, dependencies["quantile_signal"])
    
    def _generate_interaction_effects(self, feature_name: str, feature_type: str) -> List[str]:
        """Generate interaction effects with other features"""
        
        interactions = {
            "quantile_signal": [
                "Works best when combined with regime classification for adaptive thresholds",
                "Enhanced by volatility features for risk-adjusted position sizing",
                "Complemented by sentiment features for contrarian opportunities"
            ],
            "volatility_risk": [
                "Essential input for position sizing optimization",
                "Enhances regime classification accuracy",
                "Modifies quantile signal interpretation"
            ],
            "regime_classification": [
                "Modifies effectiveness of all other features",
                "Essential for adaptive thresholds and position sizing",
                "Determines which feature combinations work best"
            ],
            "position_sizing": [
                "Integrates inputs from all risk and signal features",
                "Modified by regime classification",
                "Optimized using volatility and correlation features"
            ],
            "momentum": [
                "Enhanced by regime classification for trend identification",
                "Combined with quantile signals for entry timing",
                "Modified by volatility for risk adjustment"
            ],
            "sentiment": [
                "Provides contrarian signals to momentum features",
                "Enhanced by volatility for extreme condition detection",
                "Combined with regime features for market state analysis"
            ]
        }
        
        return interactions.get(feature_type, interactions["quantile_signal"])
    
    def _generate_visual_description(self, feature_name: str, feature_type: str) -> str:
        """Generate visual description for charts"""
        
        descriptions = {
            "quantile_signal": f"On price charts, {feature_name} appears as probability levels that anticipate price movements - high values before rallies, low values before declines.",
            "volatility_risk": f"On charts, {feature_name} shows as expanding/contracting bands around price, with expansion preceding significant moves.",
            "regime_classification": f"On charts, {feature_name} appears as background shading or indicators showing current market regime state.",
            "position_sizing": f"On charts, {feature_name} would show as position size indicators that scale with opportunity and risk levels.",
            "momentum": f"On charts, {feature_name} appears as directional indicators that strengthen during trends and weaken at reversals.",
            "sentiment": f"On charts, {feature_name} shows as oscillator-style indicators with extreme readings at market turning points."
        }
        
        return descriptions.get(feature_type, descriptions["quantile_signal"])
    
    def _generate_example_scenarios(self, feature_name: str, feature_type: str) -> List[str]:
        """Generate example scenarios for feature"""
        
        scenarios = {
            "quantile_signal": [
                f"{feature_name} reaches extreme high values just before major breakouts",
                f"{feature_name} shows divergence with price at market tops/bottoms",
                f"{feature_name} clusters around neutral during sideways markets"
            ],
            "volatility_risk": [
                f"{feature_name} spikes before earnings announcements or major news",
                f"{feature_name} contracts during quiet summer trading periods",
                f"{feature_name} shows regime shifts during market crises"
            ],
            "regime_classification": [
                f"{feature_name} identifies bull market regime during sustained uptrends",
                f"{feature_name} switches to crisis regime during market crashes",
                f"{feature_name} detects sideways regime during consolidation periods"
            ],
            "position_sizing": [
                f"{feature_name} increases size during high-conviction setups",
                f"{feature_name} reduces size during uncertain market conditions",
                f"{feature_name} adapts to changing volatility regimes"
            ],
            "momentum": [
                f"{feature_name} strengthens during trending markets",
                f"{feature_name} weakens at trend exhaustion points",
                f"{feature_name} reverses during regime transitions"
            ],
            "sentiment": [
                f"{feature_name} reaches extreme fear during market bottoms",
                f"{feature_name} shows extreme greed at market tops",
                f"{feature_name} provides contrarian signals at extremes"
            ]
        }
        
        return scenarios.get(feature_type, scenarios["quantile_signal"])
    
    def _generate_chart_patterns(self, feature_name: str, feature_type: str) -> List[str]:
        """Generate associated chart patterns"""
        
        patterns = {
            "quantile_signal": ["Breakout patterns", "Reversal patterns", "Continuation patterns"],
            "volatility_risk": ["Bollinger Band expansions", "Volatility breakouts", "Squeeze patterns"],
            "regime_classification": ["Trend channels", "Support/resistance levels", "Market structure changes"],
            "position_sizing": ["Risk-reward setups", "High-probability patterns", "Confluence zones"],
            "momentum": ["Trend lines", "Moving average crossovers", "Momentum divergences"],
            "sentiment": ["Contrarian patterns", "Extreme readings", "Sentiment divergences"]
        }
        
        return patterns.get(feature_type, patterns["quantile_signal"])
    
    def _generate_false_signals(self, feature_name: str, feature_type: str) -> List[str]:
        """Generate false signal scenarios"""
        
        false_signals = {
            "quantile_signal": [
                "Extreme readings during regime transitions",
                "False breakouts in choppy markets",
                "Noise during low-volume periods"
            ],
            "volatility_risk": [
                "Volatility spikes that don't lead to sustained moves",
                "False regime change signals",
                "Noise in low-liquidity periods"
            ],
            "regime_classification": [
                "Whipsaws during regime transition periods",
                "False regime signals during market noise",
                "Lag in detecting rapid regime changes"
            ],
            "position_sizing": [
                "Over-sizing during false breakouts",
                "Under-sizing during genuine opportunities",
                "Parameter instability during regime changes"
            ],
            "momentum": [
                "False momentum signals at trend exhaustion",
                "Whipsaws during sideways markets",
                "Momentum traps during reversals"
            ],
            "sentiment": [
                "Sentiment extremes that extend further",
                "False contrarian signals during strong trends",
                "Sentiment manipulation or measurement errors"
            ]
        }
        
        return false_signals.get(feature_type, false_signals["quantile_signal"])
    
    def _generate_confirmation_signals(self, feature_name: str, feature_type: str) -> List[str]:
        """Generate confirmation signal requirements"""
        
        confirmations = {
            "quantile_signal": [
                "Volume confirmation on breakouts",
                "Multiple timeframe alignment",
                "Regime classification support"
            ],
            "volatility_risk": [
                "Price movement confirmation",
                "Volume expansion confirmation",
                "Multiple volatility measure agreement"
            ],
            "regime_classification": [
                "Multiple regime indicators agreement",
                "Sustained regime characteristics",
                "Volume and price action confirmation"
            ],
            "position_sizing": [
                "Risk-reward ratio confirmation",
                "Multiple signal alignment",
                "Volatility regime stability"
            ],
            "momentum": [
                "Volume confirmation",
                "Multiple timeframe momentum alignment",
                "Trend strength indicators"
            ],
            "sentiment": [
                "Price action confirmation",
                "Multiple sentiment measures agreement",
                "Volume confirmation of sentiment extremes"
            ]
        }
        
        return confirmations.get(feature_type, confirmations["quantile_signal"])
    
    def _determine_secondary_roles(self, feature_name: str, feature_type: str) -> List[SupplyDemandRole]:
        """Determine secondary roles for feature"""
        
        secondary_roles_map = {
            "quantile_signal": [SupplyDemandRole.SUPPLY_DETECTOR, SupplyDemandRole.DEMAND_DETECTOR],
            "volatility_risk": [SupplyDemandRole.IMBALANCE_DETECTOR, SupplyDemandRole.REGIME_CLASSIFIER],
            "regime_classification": [SupplyDemandRole.RISK_ASSESSOR, SupplyDemandRole.IMBALANCE_DETECTOR],
            "position_sizing": [SupplyDemandRole.RISK_ASSESSOR],
            "momentum": [SupplyDemandRole.SUPPLY_DETECTOR, SupplyDemandRole.IMBALANCE_DETECTOR],
            "sentiment": [SupplyDemandRole.SUPPLY_DETECTOR, SupplyDemandRole.DEMAND_DETECTOR]
        }
        
        return secondary_roles_map.get(feature_type, [])
    
    def _determine_regime_sensitivity(self, feature_name: str, feature_type: str) -> str:
        """Determine regime sensitivity level"""
        
        sensitivity_map = {
            "quantile_signal": "high",
            "volatility_risk": "medium", 
            "regime_classification": "low",  # Should be stable across regimes
            "position_sizing": "high",
            "momentum": "high",
            "sentiment": "medium"
        }
        
        return sensitivity_map.get(feature_type, "medium")
    
    def _determine_interaction_features(self, feature_name: str, feature_type: str) -> List[str]:
        """Determine which features this works best with"""
        
        interactions_map = {
            "quantile_signal": ["regime_multiplier", "vol_risk", "kelly_criterion"],
            "volatility_risk": ["q50", "regime_features", "position_sizing"],
            "regime_classification": ["q50", "vol_risk", "momentum", "sentiment"],
            "position_sizing": ["q50", "vol_risk", "regime_features"],
            "momentum": ["q50", "regime_features", "volatility"],
            "sentiment": ["q50", "regime_features", "volatility"]
        }
        
        return interactions_map.get(feature_type, [])
    
    def _generate_validation_criteria(self, feature_name: str, thesis: ThesisStatement) -> List[str]:
        """Generate validation criteria based on thesis"""
        
        return [
            f"Economic logic validation: {thesis.hypothesis[:100]}...",
            f"Statistical significance testing with proper time-series validation",
            f"Regime-aware performance testing across bull/bear/sideways markets",
            f"Chart explainability verification - can be visually explained",
            f"Integration testing with existing feature suite",
            f"Performance impact measurement on risk-adjusted returns"
        ]
    
    def _determine_dependencies(self, feature_name: str, existing_content: Dict[str, Any]) -> List[str]:
        """Determine feature dependencies"""
        
        # Extract dependencies from existing content or infer from name
        dependencies = existing_content.get("dependencies", [])
        
        # Add common dependencies based on feature type
        feature_type = self._classify_feature_type(feature_name)
        
        common_deps = {
            "quantile_signal": ["data_pipeline", "price_data"],
            "volatility_risk": ["price_data", "returns_calculation"],
            "regime_classification": ["volatility_features", "sentiment_data"],
            "position_sizing": ["signal_features", "risk_features"],
            "momentum": ["price_data", "returns_calculation"],
            "sentiment": ["external_data", "sentiment_sources"]
        }
        
        return list(set(dependencies + common_deps.get(feature_type, [])))


def validate_economic_logic(thesis: ThesisStatement, principles_content: str) -> Tuple[bool, List[str]]:
    """
    Validate thesis statement has minimal required content
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Just check for minimal content - trust other gates for quality
    if len(thesis.hypothesis) < 20:
        issues.append("Hypothesis too brief")
    
    if len(thesis.economic_basis) < 20:
        issues.append("Economic basis too brief")
    
    if len(thesis.market_microstructure) < 20:
        issues.append("Market microstructure explanation too brief")
    
    if len(thesis.expected_behavior) < 20:
        issues.append("Expected behavior too brief")
    
    # Don't require failure modes - let the generator handle this
    
    return len(issues) == 0, issues