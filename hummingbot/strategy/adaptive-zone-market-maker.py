import logging
import time
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent, BuyOrderCompletedEvent, SellOrderCompletedEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.client.hummingbot_application import CandlesFactory, CandlesConfig
from hummingbot.connector.connector_base import ConnectorBase

import pandas as pd
import numpy as np


class MeanReversionMarketMaker(ScriptStrategyBase):
    """
    Mean Reversion Market Maker with Dynamic Risk Controls
    
    Key Features:
    1. Mean Reversion Core: Uses multiple timeframe moving averages
    2. Volatility-Aware Spreads: Adjusts spreads based on ATR
    3. Smart Inventory Management: Skews prices based on inventory levels
    4. Risk Management: Implements drawdown protection and trend filters
    5. Fee Optimization: Smart order placement to minimize costs
    """
    
    # Base configuration
    trading_pair = "ETH-USDT"
    exchange = "binance_paper_trade"
    base, quote = trading_pair.split('-')
    
    # Order parameters
    order_amount = Decimal("0.01")  # Base order size
    min_spread = Decimal("0.001")   # 0.1% minimum spread
    max_spread = Decimal("0.03")    # 3% maximum spread
    order_refresh_time = 30         # Seconds between order refresh
    
    # Mean reversion parameters
    short_window = 20              # Short-term MA window
    long_window = 50               # Long-term MA window
    mean_reversion_threshold = 2.0  # Standard deviations for mean reversion signal
    
    # Volatility parameters
    atr_window = 14                # ATR calculation window
    vol_spread_multiplier = 1.5    # Spread multiplier based on volatility
    
    # Inventory management
    target_base_pct = Decimal("0.5")  # Target 50% in base asset
    inventory_range = Decimal("0.1")   # Acceptable range around target
    max_inventory_skew = Decimal("0.2") # Maximum price skew from inventory
    
    # Risk management
    max_drawdown = Decimal("0.05")     # 5% max drawdown
    trend_strength_threshold = 70       # RSI threshold for trend strength
    circuit_breaker_enabled = True
    
    # Performance tracking
    start_portfolio_value = Decimal("0")
    current_portfolio_value = Decimal("0")
    trades_completed = 0
    
    # Markets configuration
    markets = {exchange: {trading_pair}}
    
    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        
        # Initialize candles for different timeframes
        self.candles = CandlesFactory.get_candle(
            CandlesConfig(
                connector=self.exchange,
                trading_pair=self.trading_pair,
                interval="1m",
                max_records=100
            )
        )
        self.candles.start()
        
        # Initialize portfolio value
        self.start_portfolio_value = self.get_portfolio_value()
        self.current_portfolio_value = self.start_portfolio_value
        
        self.log_with_clock(
            logging.INFO,
            "Strategy initialized. Starting portfolio value: "
            f"{self.start_portfolio_value:.4f} {self.quote}"
        )

    def on_stop(self):
        """Cleanup when strategy stops"""
        self.candles.stop()
        self.cancel_all_orders()
        
        final_value = self.get_portfolio_value()
        pnl = ((final_value - self.start_portfolio_value) / 
               self.start_portfolio_value * Decimal("100"))
        
        self.log_with_clock(
            logging.INFO,
            f"Strategy stopped. PnL: {pnl:.2f}% | Trades: {self.trades_completed}"
        )

    def on_tick(self):
        """Main strategy logic executed on each tick"""
        if not self.should_trade():
            return
            
        # Update market analysis
        signals = self.analyze_market()
        
        # Check risk conditions
        if self.check_risk_conditions(signals):
            self.cancel_all_orders()
            return
            
        # Create and place orders
        proposal = self.create_order_proposal(signals)
        self.place_orders(proposal)
        
    def analyze_market(self) -> dict:
        """Analyze market conditions and return trading signals"""
        df = self.get_processed_candles()
        
        if df.empty:
            return {
                "should_trade": False,
                "mean_rev_signal": 0,
                "volatility": 0,
                "trend_strength": 50
            }
            
        # Calculate mean reversion signal
        df['sma_short'] = df['close'].rolling(self.short_window).mean()
        df['sma_long'] = df['close'].rolling(self.long_window).mean()
        df['price_deviation'] = (df['close'] - df['sma_long']) / df['close'].rolling(20).std()
        
        # Calculate volatility
        df['atr'] = self.calculate_atr(df)
        current_atr = df['atr'].iloc[-1]
        avg_atr = df['atr'].rolling(20).mean().iloc[-1]
        volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        
        # Calculate trend strength using RSI
        df['rsi'] = self.calculate_rsi(df['close'])
        trend_strength = df['rsi'].iloc[-1]
        
        # Mean reversion signal (-1 to 1)
        mean_rev_signal = -df['price_deviation'].iloc[-1] / self.mean_reversion_threshold
        mean_rev_signal = max(min(mean_rev_signal, 1), -1)
        
        return {
            "should_trade": True,
            "mean_rev_signal": mean_rev_signal,
            "volatility": volatility_ratio,
            "trend_strength": trend_strength
        }
        
    def create_order_proposal(self, signals: dict) -> List[OrderCandidate]:
        """Create buy/sell orders based on market signals"""
        # Get reference price
        ref_price = self.get_reference_price()
        
        # Calculate spreads based on volatility
        base_spread = self.min_spread * (1 + signals["volatility"] * self.vol_spread_multiplier)
        base_spread = min(base_spread, self.max_spread)
        
        # Adjust spreads based on mean reversion signal
        mean_rev_adjustment = Decimal(str(signals["mean_rev_signal"] * 0.001))  # 0.1% max adjustment
        
        # Get inventory skew adjustment
        inventory_skew = self.calculate_inventory_skew()
        
        # Final spread calculations
        bid_spread = base_spread - mean_rev_adjustment + inventory_skew
        ask_spread = base_spread + mean_rev_adjustment - inventory_skew
        
        # Calculate final prices
        bid_price = ref_price * (Decimal("1") - bid_spread)
        ask_price = ref_price * (Decimal("1") + ask_spread)
        
        # Create order candidates
        orders = []
        
        # Only place buy order if we're not too heavy on base asset
        if inventory_skew < self.max_inventory_skew:
            orders.append(
                OrderCandidate(
                    trading_pair=self.trading_pair,
                    is_maker=True,
                    order_type=OrderType.LIMIT,
                    order_side=TradeType.BUY,
                    amount=self.order_amount,
                    price=bid_price
                )
            )
            
        # Only place sell order if we're not too light on base asset
        if inventory_skew > -self.max_inventory_skew:
            orders.append(
                OrderCandidate(
                    trading_pair=self.trading_pair,
                    is_maker=True,
                    order_type=OrderType.LIMIT,
                    order_side=TradeType.SELL,
                    amount=self.order_amount,
                    price=ask_price
                )
            )
            
        return orders

    def check_risk_conditions(self, signals: dict) -> bool:
        """Check if we should stop trading based on risk conditions"""
        # Check drawdown
        current_drawdown = self.calculate_drawdown()
        if current_drawdown < -self.max_drawdown:
            self.log_with_clock(
                logging.WARNING,
                f"Max drawdown ({self.max_drawdown:.1%}) exceeded. "
                f"Current: {current_drawdown:.1%}"
            )
            return True
            
        # Check trend strength
        if abs(signals["trend_strength"] - 50) > self.trend_strength_threshold:
            self.log_with_clock(
                logging.INFO,
                f"Strong trend detected (RSI: {signals['trend_strength']:.1f}). "
                "Pausing trading."
            )
            return True
            
        # Check volatility
        if signals["volatility"] > 3.0:  # Volatility 3x normal
            self.log_with_clock(
                logging.INFO,
                f"High volatility detected ({signals['volatility']:.1f}x normal). "
                "Pausing trading."
            )
            return True
            
        return False

    # Helper methods
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.atr_window).mean()

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_inventory_skew(self) -> Decimal:
        """Calculate price skew based on inventory position"""
        base_value = (self.connectors[self.exchange].get_balance(self.base) * 
                     self.get_reference_price())
        quote_value = self.connectors[self.exchange].get_balance(self.quote)
        total_value = base_value + quote_value
        
        if total_value == Decimal("0"):
            return Decimal("0")
            
        current_base_pct = base_value / total_value
        inventory_skew = (current_base_pct - self.target_base_pct) / self.inventory_range
        
        return Decimal(str(max(min(float(inventory_skew) * 0.001, 0.01), -0.01)))

    def calculate_drawdown(self) -> Decimal:
        """Calculate current drawdown from starting portfolio value"""
        self.current_portfolio_value = self.get_portfolio_value()
        return (self.current_portfolio_value - self.start_portfolio_value) / self.start_portfolio_value

    def get_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value in quote currency"""
        base_value = (self.connectors[self.exchange].get_balance(self.base) * 
                     self.get_reference_price())
        quote_value = self.connectors[self.exchange].get_balance(self.quote)
        return base_value + quote_value

    def get_reference_price(self) -> Decimal:
        """Get the reference price for trading"""
        return self.connectors[self.exchange].get_price_by_type(
            self.trading_pair,
            PriceType.MidPrice
        )

    def should_trade(self) -> bool:
        """Determine if we should be trading right now"""
        return (
            self.candles is not None and
            not self.candles.candles_df.empty and
            len(self.candles.candles_df) >= self.long_window
        )

    def did_fill_order(self, event: OrderFilledEvent):
        """Callback for order fills"""
        self.trades_completed += 1
        
        filled_pct = event.price / self.get_reference_price() - Decimal("1")
        filled_pct *= Decimal("100")
        
        self.log_with_clock(
            logging.INFO,
            f"Filled {event.amount} {self.base} {event.trade_type.name} @ "
            f"{event.price} ({filled_pct:+.2f}% from mid)"
        )

    def format_status(self) -> str:
        """Format status output for display"""
        if not self.ready_to_trade:
            return "Market connectors are not ready."
            
        lines = []
        
        # Portfolio stats
        current_value = self.get_portfolio_value()
        pnl = (current_value - self.start_portfolio_value) / self.start_portfolio_value
        
        lines.extend([
            "",
            "  Portfolio:",
            f"    Value: {current_value:.4f} {self.quote}",
            f"    PnL: {pnl:.2%}",
            f"    Trades: {self.trades_completed}"
        ])
        
        # Market analysis
        signals = self.analyze_market()
        lines.extend([
            "",
            "  Market Analysis:",
            f"    Mean Reversion Signal: {signals['mean_rev_signal']:.2f}",
            f"    Volatility Ratio: {signals['volatility']:.2f}x",
            f"    Trend Strength (RSI): {signals['trend_strength']:.1f}"
        ])
        
        # Risk metrics
        lines.extend([
            "",
            "  Risk Metrics:",
            f"    Drawdown: {self.calculate_drawdown():.2%}",
            f"    Inventory Skew: {self.calculate_inventory_skew():.2%}"
        ])
        
        return "\n".join(lines)
