# Memecoin Momentum Hunter Strategy

This document outlines a Hummingbot strategy designed to capitalize on short-term momentum in cryptocurrencies, particularly memecoins. The strategy uses multiple technical indicators to identify and trade coins with strong momentum.

## Strategy Concept

Memecoins often experience rapid price movements driven by social media attention and market sentiment rather than fundamentals. A successful strategy would:

1. Quickly identify coins with building momentum
2. Enter positions early in the momentum cycle
3. Use trailing stops to ride the momentum up
4. Exit before the inevitable correction

## Technical Approach

The strategy combines multiple indicators to confirm momentum:

- **Rate of Change (ROC)** - To measure the speed of price changes over very short timeframes (1-5 minutes)
- **Volume Analysis** - Sudden increases in volume often precede price movements in memecoins
- **RSI with Short Lookback** - To identify when momentum is building but not yet overextended
- **MACD with Fast Settings** - To confirm trend direction and strength

### Entry Logic

The strategy enters when:
- ROC exceeds a threshold (indicating rapid price appreciation)
- Volume is increasing significantly
- RSI is rising but below overbought levels (to avoid buying at the top)
- MACD shows a bullish signal

### Position Management

- Use position executor for entry and exit
- Implement trailing stops that tighten as profit increases
- Set multiple take-profit levels
- Implement a time-based exit to avoid holding positions too long

### Risk Management

- Small position sizes (1-2% of portfolio per trade)
- Strict stop losses (3-5%)
- Maximum drawdown limits
- Diversification across multiple memecoins

## Implementation

The strategy is implemented as a V2 strategy with a custom controller that extends `DirectionalTradingControllerBase`.

### Controller Implementation

Create this file at `controllers/directional_trading/memecoin_momentum.py`:

```python
import time
from decimal import Decimal
from typing import List, Optional

import pandas_ta as ta  # noqa: F401
import numpy as np
from pydantic import Field, validator

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TrailingStop


class MemecoinMomentumConfig(DirectionalTradingControllerConfigBase):
    controller_name: str = "memecoin_momentum"
    candles_config: List[CandlesConfig] = []
    
    # Candles configuration
    candles_connector: str = Field(
        default=None,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the connector for the candles data, leave empty to use the same exchange as the connector: ",)
    )
    candles_trading_pair: str = Field(
        default=None,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the trading pair for the candles data, leave empty to use the same trading pair as the connector: ",)
    )
    interval: str = Field(
        default="1m",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the candle interval (e.g., 1m, 3m, 5m): ",
            prompt_on_new=True))
    
    # ROC (Rate of Change) parameters
    roc_length: int = Field(
        default=5,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the ROC length (number of periods): ",
            prompt_on_new=True))
    roc_threshold: float = Field(
        default=2.0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the ROC threshold percentage (e.g., 2.0 for 2%): ",
            prompt_on_new=True))
    
    # Volume parameters
    volume_ma_length: int = Field(
        default=20,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the volume moving average length: ",
            prompt_on_new=True))
    volume_threshold: float = Field(
        default=2.0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the volume threshold multiplier: ",
            prompt_on_new=True))
    
    # RSI parameters
    rsi_length: int = Field(
        default=14,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the RSI length: ",
            prompt_on_new=True))
    rsi_overbought: int = Field(
        default=70,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the RSI overbought threshold: ",
            prompt_on_new=True))
    rsi_oversold: int = Field(
        default=30,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the RSI oversold threshold: ",
            prompt_on_new=True))
    
    # MACD parameters
    macd_fast: int = Field(
        default=12,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD fast period: ",
            prompt_on_new=True))
    macd_slow: int = Field(
        default=26,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD slow period: ",
            prompt_on_new=True))
    macd_signal: int = Field(
        default=9,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD signal period: ",
            prompt_on_new=True))
    
    # Position management
    trailing_stop_activation_price_delta: Decimal = Field(
        default=Decimal("0.01"),
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the trailing stop activation price delta (e.g., 0.01 for 1%): ",
            prompt_on_new=True))
    trailing_stop_trailing_delta: Decimal = Field(
        default=Decimal("0.005"),
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the trailing stop trailing delta (e.g., 0.005 for 0.5%): ",
            prompt_on_new=True))
    
    # Maximum holding time in seconds (e.g., 3600 for 1 hour)
    max_holding_time: int = Field(
        default=3600,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the maximum holding time in seconds: ",
            prompt_on_new=True))
    
    @validator("candles_connector", pre=True, always=True)
    def set_candles_connector(cls, v, values):
        if v is None or v == "":
            return values.get("connector_name")
        return v
    
    @validator("candles_trading_pair", pre=True, always=True)
    def set_candles_trading_pair(cls, v, values):
        if v is None or v == "":
            return values.get("trading_pair")
        return v


class MemecoinMomentumController(DirectionalTradingControllerBase):
    """
    Controller for the Memecoin Momentum strategy.
    
    This strategy identifies and capitalizes on short-term momentum in cryptocurrencies,
    particularly memecoins, using a combination of ROC, volume analysis, RSI, and MACD.
    """
    
    def __init__(self, config: MemecoinMomentumConfig, *args, **kwargs):
        self.config = config
        # Calculate max records needed based on the longest indicator lookback
        self.max_records = max(
            config.roc_length,
            config.volume_ma_length,
            config.rsi_length,
            config.macd_slow + config.macd_signal
        ) + 20  # Add buffer
        
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)
    
    async def update_processed_data(self):
        """
        Update the processed data with the latest market data and indicators.
        This method is called periodically by the strategy.
        """
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )
        
        if df.empty:
            self.logger().warning("No candle data available. Skipping update.")
            return
        
        # Calculate ROC (Rate of Change)
        df["roc"] = df["close"].pct_change(periods=self.config.roc_length) * 100
        
        # Calculate Volume Moving Average
        df["volume_ma"] = df["volume"].rolling(window=self.config.volume_ma_length).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]
        
        # Calculate RSI
        df.ta.rsi(length=self.config.rsi_length, append=True)
        
        # Calculate MACD
        df.ta.macd(
            fast=self.config.macd_fast,
            slow=self.config.macd_slow,
            signal=self.config.macd_signal,
            append=True
        )
        
        # Extract indicator values
        roc = df["roc"].iloc[-1]
        volume_ratio = df["volume_ratio"].iloc[-1]
        rsi = df[f"RSI_{self.config.rsi_length}"].iloc[-1]
        macd = df[f"MACD_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"].iloc[-1]
        macd_signal = df[f"MACDs_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"].iloc[-1]
        macd_hist = df[f"MACDh_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"].iloc[-1]
        
        # Generate signals based on combined indicators
        # Long signal: High ROC, High Volume, RSI not overbought, Positive MACD momentum
        long_condition = (
            roc > self.config.roc_threshold and
            volume_ratio > self.config.volume_threshold and
            rsi < self.config.rsi_overbought and
            rsi > 50 and  # Ensuring RSI is in bullish territory
            macd_hist > 0  # Positive MACD histogram
        )
        
        # Short signal: Negative ROC, High Volume, RSI not oversold, Negative MACD momentum
        # Note: For memecoins, we might focus primarily on long positions
        short_condition = (
            roc < -self.config.roc_threshold and
            volume_ratio > self.config.volume_threshold and
            rsi > self.config.rsi_oversold and
            rsi < 50 and  # Ensuring RSI is in bearish territory
            macd_hist < 0  # Negative MACD histogram
        )
        
        # Set signal
        signal = 0
        if long_condition:
            signal = 1
        elif short_condition:
            signal = -1
        
        # Update processed data
        self.processed_data["signal"] = signal
        self.processed_data["features"] = df
        self.processed_data["indicators"] = {
            "roc": roc,
            "volume_ratio": volume_ratio,
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist
        }
        
        # Log the current state
        self.logger().info(
            f"Memecoin Momentum Indicators - ROC: {roc:.2f}%, Volume Ratio: {volume_ratio:.2f}, "
            f"RSI: {rsi:.2f}, MACD Hist: {macd_hist:.6f}, Signal: {signal}"
        )
    
    def get_executor_config(self, trade_type: TradeType, price: Decimal, amount: Decimal) -> PositionExecutorConfig:
        """
        Create a position executor configuration based on the current market conditions.
        """
        # Create trailing stop configuration
        trailing_stop = TrailingStop(
            activation_price=self.config.trailing_stop_activation_price_delta,
            trailing_delta=self.config.trailing_stop_trailing_delta
        )
        
        # Create position executor config
        return PositionExecutorConfig(
            timestamp=time.time(),
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            side=trade_type,
            entry_price=price,
            amount=amount,
            stop_loss=self.config.stop_loss,
            take_profit=self.config.take_profit,
            time_limit=self.config.max_holding_time,
            trailing_stop=trailing_stop,
            leverage=self.config.leverage
        )
```

### Strategy Script Implementation

Create this file at `scripts/memecoin_momentum_strategy.py`:

```python
import os
from decimal import Decimal
from typing import Dict, List, Set

from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PositionMode, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction

# Import our custom controller
from controllers.directional_trading.memecoin_momentum import MemecoinMomentumConfig


class MemecoinMomentumStrategyConfig(StrategyV2ConfigBase):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    # Strategy will automatically populate markets and candles_config
    markets: Dict[str, Set[str]] = {}
    candles_config: List[CandlesConfig] = []
    controllers_config: List[str] = []
    
    # Exchange settings
    exchange: str = Field(
        default="binance",  # Change to your preferred exchange
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the exchange to trade on: "
        )
    )
    trading_pair: str = Field(
        default="DOGE-USDT",  # Example memecoin
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the trading pair: "
        )
    )
    
    # Order amount in quote currency (e.g., USDT)
    order_amount_quote: Decimal = Field(
        default=Decimal("20"),
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the order amount in quote currency: "
        )
    )
    
    # Risk parameters
    stop_loss: Decimal = Field(
        default=Decimal("0.03"),  # 3%
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the stop loss percentage (e.g., 0.03 for 3%): "
        )
    )
    take_profit: Decimal = Field(
        default=Decimal("0.05"),  # 5%
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the take profit percentage (e.g., 0.05 for 5%): "
        )
    )
    
    # For perpetual futures trading
    leverage: int = Field(
        default=1,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter leverage (1 for spot trading): "
        )
    )
    position_mode: PositionMode = Field(
        default="ONEWAY",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Position mode (HEDGE/ONEWAY) for perpetual trading: "
        )
    )
    
    # Controller configuration
    controller_config: MemecoinMomentumConfig = None


class MemecoinMomentumStrategy(StrategyV2Base):
    """
    Strategy for trading memecoins based on short-term momentum indicators.
    """
    
    @classmethod
    def init_markets(cls, config: MemecoinMomentumStrategyConfig):
        cls.markets = {config.exchange: {config.trading_pair}}
    
    def __init__(self, connectors: Dict[str, ConnectorBase], config: MemecoinMomentumStrategyConfig):
        # Initialize the controller config if not already set
        if config.controller_config is None:
            config.controller_config = MemecoinMomentumConfig(
                connector_name=config.exchange,
                trading_pair=config.trading_pair,
                stop_loss=config.stop_loss,
                take_profit=config.take_profit,
                leverage=config.leverage,
                # Other parameters will use defaults
            )
        
        # Initialize strategy
        super().__init__(connectors, config)
        self.config = config
        self.account_config_set = False
    
    def start(self, clock, timestamp):
        """
        Start the strategy.
        """
        self._last_timestamp = timestamp
        self.apply_initial_settings()
    
    def apply_initial_settings(self):
        """
        Apply initial settings for the strategy.
        """
        if not self.account_config_set:
            for connector_name, connector in self.connectors.items():
                if self.is_perpetual(connector_name):
                    connector.set_position_mode(self.config.position_mode)
                    connector.set_leverage(
                        trading_pair=self.config.trading_pair,
                        leverage=self.config.leverage
                    )
            self.account_config_set = True
    
    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        """
        Create actions based on controller signals.
        """
        create_actions = []
        
        # Get signal from the controller
        controller_id = self.config.controller_config.controller_name
        controller = self.controllers.get(controller_id)
        
        if controller is None:
            self.logger().error(f"Controller {controller_id} not found")
            return []
        
        signal = controller.processed_data.get("signal", 0)
        
        # Get active executors
        active_executors = self.filter_executors(
            executors=self.get_all_executors(),
            filter_func=lambda e: e.connector_name == self.config.exchange and 
                                 e.trading_pair == self.config.trading_pair and 
                                 e.is_active
        )
        
        # Check if we already have active positions
        active_longs = [e for e in active_executors if e.side == TradeType.BUY]
        active_shorts = [e for e in active_executors if e.side == TradeType.SELL]
        
        # Get mid price
        mid_price = self.market_data_provider.get_price_by_type(
            self.config.exchange,
            self.config.trading_pair,
            price_type="mid_price"
        )
        
        # Calculate order amount in base currency
        amount = self.config.order_amount_quote / mid_price
        
        # Create new position if signal is present and we don't have an active position
        if signal == 1 and len(active_longs) == 0:
            self.logger().info(f"Creating LONG position for {self.config.trading_pair}")
            create_actions.append(
                CreateExecutorAction(
                    executor_config=controller.get_executor_config(
                        trade_type=TradeType.BUY,
                        price=mid_price,
                        amount=amount
                    )
                )
            )
        elif signal == -1 and len(active_shorts) == 0:
            self.logger().info(f"Creating SHORT position for {self.config.trading_pair}")
            create_actions.append(
                CreateExecutorAction(
                    executor_config=controller.get_executor_config(
                        trade_type=TradeType.SELL,
                        price=mid_price,
                        amount=amount
                    )
                )
            )
        
        return create_actions
    
    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        """
        Create stop actions based on controller signals.
        """
        stop_actions = []
        
        # Get signal from the controller
        controller_id = self.config.controller_config.controller_name
        controller = self.controllers.get(controller_id)
        
        if controller is None:
            return []
        
        signal = controller.processed_data.get("signal", 0)
        
        # Get active executors
        active_executors = self.filter_executors(
            executors=self.get_all_executors(),
            filter_func=lambda e: e.connector_name == self.config.exchange and 
                                 e.trading_pair == self.config.trading_pair and 
                                 e.is_active
        )
        
        # Check if we need to stop any positions based on signal reversal
        active_longs = [e for e in active_executors if e.side == TradeType.BUY]
        active_shorts = [e for e in active_executors if e.side == TradeType.SELL]
        
        # If signal reverses, close existing positions
        if signal == -1 and len(active_longs) > 0:
            self.logger().info(f"Closing LONG positions due to signal reversal")
            stop_actions.extend([StopExecutorAction(executor_id=e.id) for e in active_longs])
        elif signal == 1 and len(active_shorts) > 0:
            self.logger().info(f"Closing SHORT positions due to signal reversal")
            stop_actions.extend([StopExecutorAction(executor_id=e.id) for e in active_shorts])
        
        return stop_actions
    
    def format_status(self) -> str:
        """
        Format the strategy status for display.
        """
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        
        lines = []
        
        # Get controller
        controller_id = self.config.controller_config.controller_name
        controller = self.controllers.get(controller_id)
        
        if controller is not None and "indicators" in controller.processed_data:
            indicators = controller.processed_data["indicators"]
            lines.extend([
                "",
                "  Momentum Indicators:",
                f"    ROC: {indicators['roc']:.2f}%",
                f"    Volume Ratio: {indicators['volume_ratio']:.2f}",
                f"    RSI: {indicators['rsi']:.2f}",
                f"    MACD Histogram: {indicators['macd_hist']:.6f}",
                f"    Signal: {controller.processed_data.get('signal', 0)}"
            ])
        
        # Add balance information
        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])
        
        # Add active positions
        active_executors = self.filter_executors(
            executors=self.get_all_executors(),
            filter_func=lambda e: e.connector_name == self.config.exchange and 
                                 e.trading_pair == self.config.trading_pair and 
                                 e.is_active
        )
        
        if active_executors:
            lines.extend(["", "  Active Positions:"])
            for executor in active_executors:
                lines.append(f"    {executor.trading_pair} {executor.side.name}: "
                             f"Amount: {executor.amount:.6f}, "
                             f"Entry: {executor.entry_price:.6f}, "
                             f"Current: {executor.current_price:.6f}, "
                             f"PnL: {executor.realized_pnl_quote + executor.unrealized_pnl_quote:.6f}")
        
        return "\n".join(lines)
```

## How to Use the Strategy

1. Create the controller file at `controllers/directional_trading/memecoin_momentum.py`
2. Create the strategy script at `scripts/memecoin_momentum_strategy.py`
3. Run the strategy with:

```
start --script memecoin_momentum_strategy.py
```

## Exchange Selection

For memecoins, you'd want to focus on:

1. **DEXs** like Uniswap, PancakeSwap, or SushiSwap where new tokens appear first
2. **CEXs with quick listings** like Gate.io, MEXC, or Kucoin

## Customization Options

You can customize this strategy by:

1. Adjusting indicator parameters (lengths, thresholds)
2. Modifying the signal generation logic
3. Changing risk parameters (stop-loss, take-profit)
4. Adding additional indicators or filters
5. Implementing exchange-specific features

## Risk Warning

Trading memecoins involves significant risk due to their high volatility and potential for manipulation. Always:

1. Use only risk capital you can afford to lose
2. Start with small position sizes
3. Test thoroughly in paper trading mode first
4. Monitor positions closely
5. Be prepared for rapid price movements in either direction

## Further Improvements

Potential enhancements to consider:

1. Add social media sentiment analysis (Twitter, Reddit, Telegram)
2. Implement volume profile analysis to identify accumulation
3. Add on-chain metrics for blockchain-based analysis
4. Create a multi-pair version to trade multiple memecoins simultaneously
5. Add dynamic position sizing based on volatility