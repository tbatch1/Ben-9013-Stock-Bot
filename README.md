# Ben---Bot
import pandas as pd
from datetime import datetime, timedelta
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from lumibot.entities import Asset, Order, TradingFee
from lumibot.credentials import IS_BACKTESTING
# For backtesting; using Yahoo for daily data
from lumibot.backtesting import YahooDataBacktesting

"""
Strategy Description:
---------------------
This strategy invests in a selected group of stocks from emerging technologies, AI, defense, and agriculture sectors by momentum every 6 months.
Before buying, it evaluates technical indicators (RSI, Bollinger Bands, Stochastic oscillator, and Chaikin Money Flow) to rate the buying opportunity on a scale of 0-10.
Only if the opportunity rating is 7 or above a limit order is placed for a swing trade position.
Additionally, positions are sized small (half of the equal allocation) to favor stocks with small share structures that can move easier when volume comes in.

User Query:
-----------
i dont want it to be s&p 500 stocks i want it to be emerging technologies ai defense agriculture stocks

This code was refined based on the user prompt: 'i dont want it to be s&p 500 stocks i want it to be emerging technologies ai defense agriculture stocks'
"""

class MomentumEmergingTechStrategy(Strategy):
    # Updated stock list to emerging technologies, AI, defense, and agriculture stocks
    EMERGING_STOCKS = [
        "NVDA",  # Leading in AI and GPU technology
        "AMD",   # Advanced micro devices in tech solutions
        "CRM",   # Emerging in cloud-based software/AI
        "LMT",   # Defense sector: Lockheed Martin
        "BA",    # Defense/aerospace sector: Boeing
        "ADM",   # Agriculture sector: Archer Daniels Midland
        "DE"     # Agriculture/industrial: Deere & Company
    ]
    
    def initialize(self):
        # Set bot sleep time; platform will run on each trading iteration (daily)
        self.sleeptime = "1D"
        # Initialize variable to track last rebalance date; None forces immediate rebalance on first iteration.
        self.vars.last_rebalance_date = None
        # Set momentum lookback period to 6 months (~126 trading days); buffer with extra days.
        self.vars.momentum_lookback_days = 130

    def compute_rsi(self, series, period=14):
        # Simple RSI calculation using pandas
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def evaluate_buy_opportunity(self, df):
        # Evaluate technical indicators and return a score (0-10)
        score = 0
        # Calculate RSI if enough data is available (using 14-day period)
        if len(df) >= 14:
            rsi = self.compute_rsi(df['close'], 14)
            # Lower RSI can indicate oversold conditions; add points accordingly.
            if rsi < 30:
                score += 3
            elif rsi < 40:
                score += 2
            elif rsi < 50:
                score += 1
        # Bollinger Bands: use 20-day moving average and std deviation
        if len(df) >= 20:
            ma = df['close'].rolling(20).mean().iloc[-1]
            std = df['close'].rolling(20).std().iloc[-1]
            upper = ma + 2 * std
            lower = ma - 2 * std
            # Percent position of the last close within the band
            if (upper - lower) != 0:
                boll_perc = (df['close'].iloc[-1] - lower) / (upper - lower)
            else:
                boll_perc = 0.5
            # Lower percent may indicate undervalued price relative to volatility
            if boll_perc < 0.3:
                score += 2
            elif boll_perc < 0.5:
                score += 1
        # Stochastic Oscillator: 14-day period %K calculation
        if len(df) >= 14:
            low14 = df['low'].rolling(14).min().iloc[-1]
            high14 = df['high'].rolling(14).max().iloc[-1]
            current_close = df['close'].iloc[-1]
            if (high14 - low14) != 0:
                stochastic = (current_close - low14) / (high14 - low14) * 100
            else:
                stochastic = 50
            # Oversold conditions reflected by low %K
            if stochastic < 20:
                score += 2
            elif stochastic < 50:
                score += 1
        # Chaikin Money Flow (CMF): measure accumulation/distribution over 20 days
        if len(df) >= 20:
            # Money Flow Multiplier calculation; avoid division by zero by replacing 0 with a small number.
            price_range = (df['high'] - df['low']).replace(0, 0.001)
            mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / price_range
            mf_volume = mf_multiplier * df['volume']
            money_flow = mf_volume.rolling(window=20).sum().iloc[-1]
            volume_sum = df['volume'].rolling(window=20).sum().iloc[-1]
            cmf = money_flow / volume_sum if volume_sum != 0 else 0
            # Positive CMF indicates buying pressure; add points if above thresholds.
            if cmf > 0.1:
                score += 2
            elif cmf > 0:
                score += 1
        return score

    def on_trading_iteration(self):
        # Get the current datetime from the framework
        current_dt = self.get_datetime()
        # Check when the last rebalance was done; if None, force rebalance.
        last_rebalance = self.vars.last_rebalance_date
        # Rebalance if 6 months (approx. 180 days) have passed
        if last_rebalance is None or (current_dt - last_rebalance).days >= 180:
            self.log_message("Rebalancing portfolio based on momentum...", color="green")
            # Update the last rebalance date
            self.vars.last_rebalance_date = current_dt
            
            # Dictionary to hold momentum scores for each stock
            momentum_scores = {}
            lookback = self.vars.momentum_lookback_days
            
            # Loop through each stock in the emerging technologies list and compute momentum
            for symbol in self.EMERGING_STOCKS:
                try:
                    # Retrieve daily historical prices for the lookback period
                    bars = self.get_historical_prices(symbol, length=lookback, timestep="day")
                    if bars is not None and not bars.df.empty:
                        df = bars.df
                        if len(df) > 1:
                            past_price = df.iloc[0]["close"]
                            current_price = df.iloc[-1]["close"]
                            momentum = (current_price / past_price) - 1
                            momentum_scores[symbol] = momentum
                        else:
                            self.log_message(f"Not enough data for {symbol} to compute momentum.", color="red")
                    else:
                        self.log_message(f"No historical data for {symbol}.", color="red")
                except Exception as e:
                    self.log_message(f"Error processing {symbol}: {e}", color="red")
            
            # Sort the symbols by momentum in descending order and select the top 10
            sorted_symbols = sorted(momentum_scores, key=momentum_scores.get, reverse=True)
            top_symbols = sorted_symbols[:10]
            self.log_message(f"Top symbols by momentum: {top_symbols}", color="blue")
            
            # Liquidate entire portfolio (except cash) before reinvesting
            positions = self.get_positions()
            for position in positions:
                # Skip cash; USD is represented as Asset "USD" with asset_type FOREX
                if position.asset.symbol == "USD":
                    continue
                order = self.create_order(position.asset, position.quantity, Order.OrderSide.SELL)
                self.submit_order(order)
            
            cash = self.get_cash()
            if cash <= 0:
                self.log_message("No cash available for new positions.", color="red")
                return
            
            # Allocate cash equally among the selected symbols
            allocation = cash / len(top_symbols)
            for symbol in top_symbols:
                try:
                    # Retrieve a shorter history for technical indicator evaluation (e.g., 30 days)
                    bars_tech = self.get_historical_prices(symbol, length=30, timestep="day")
                    if bars_tech is None or bars_tech.df.empty:
                        self.log_message(f"Insufficient technical data for {symbol}; skipping technical analysis.", color="red")
                        continue
                    df_tech = bars_tech.df
                    # Evaluate the opportunity rating based on technical indicators
                    rating = self.evaluate_buy_opportunity(df_tech)
                    self.log_message(f"Opportunity rating for {symbol}: {rating}", color="blue")
                    # Only proceed if rating is 7 or above
                    if rating >= 7:
                        price = self.get_last_price(symbol)
                        if price is None or price <= 0:
                            self.log_message(f"Price data unavailable for {symbol}; skipping.", color="red")
                            continue
                        # Use half the equal allocation to maintain small share structures
                        shares = int((allocation * 0.5) // price)
                        if shares <= 0:
                            self.log_message(f"Insufficient allocation to buy shares of {symbol}.", color="red")
                            continue
                        # Place a limit order slightly below the current price for a swing trade entry
                        limit_price = price * 0.99
                        order = self.create_order(symbol, shares, Order.OrderSide.BUY, limit_price=limit_price)
                        self.submit_order(order)
                        self.log_message(f"Submitted limit order: BUY {shares} shares of {symbol} at limit {limit_price}", color="green")
                    else:
                        self.log_message(f"{symbol} did not meet the opportunity threshold; skipping.", color="yellow")
                except Exception as e:
                    self.log_message(f"Error processing buy order for {symbol}: {e}", color="red")
        else:
            self.log_message("Not time for rebalancing yet.", color="blue")

if __name__ == "__main__":
    if IS_BACKTESTING:
        # Backtesting path with trading fees and SPY as benchmark asset
        trading_fee = TradingFee(percent_fee=0.001)
        MomentumEmergingTechStrategy.backtest(
            datasource_class=YahooDataBacktesting,
            benchmark_asset=Asset("SPY", Asset.AssetType.STOCK),
            buy_trading_fees=[trading_fee],
            sell_trading_fees=[trading_fee],
            quote_asset=Asset("USD", Asset.AssetType.FOREX),
            parameters=None  # No extra parameters needed
        )
    else:
        # Live Trading path
        trader = Trader()
        strategy = MomentumEmergingTechStrategy(quote_asset=Asset("USD", Asset.AssetType.FOREX))
        trader.add_strategy(strategy)
        trader.run_all()
