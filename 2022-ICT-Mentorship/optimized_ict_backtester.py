#!/usr/bin/env python3
"""
Optimized ICT Backtester - Handles Large Datasets Efficiently
Uses vectorized operations and chunking for 5M+ bars
Includes IBKR commissions, slippage, and comprehensive metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class OptimizedICTBacktester:
    """
    High-performance ICT backtester for massive datasets
    Includes realistic trading costs and comprehensive analytics
    """
    
    def __init__(self, initial_capital=25000, risk_per_trade=0.02, min_rr_ratio=1.5):
        """Initialize backtester with ICT 2022 settings"""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.min_rr_ratio = min_rr_ratio
        
        # Risk management
        self.max_risk_per_trade = 500.0  # Maximum $500 risk per trade
        
        # Trading costs
        self.ibkr_commission_per_share = 0.005  # $0.005 per share (IBKR Pro)
        self.ibkr_min_commission = 1.0  # $1 minimum per trade
        self.ibkr_max_commission = 0.01  # 1% of trade value maximum
        self.slippage_pips = 0.5  # 0.5 pip slippage per side
        
        # ICT settings
        self.max_fvg_age = 100  # bars
        
        # Data storage
        self.df = None
        self.trades = []
        self.open_trades = []
        self.equity_curve = []
        self.daily_returns = []
    
    def calculate_ibkr_commission(self, position_size, share_price):
        """Calculate IBKR commission fees"""
        commission = position_size * self.ibkr_commission_per_share
        trade_value = position_size * share_price
        
        # Apply minimum and maximum commission rules
        commission = max(commission, self.ibkr_min_commission)
        commission = min(commission, trade_value * self.ibkr_max_commission)
        
        return commission
    
    def calculate_slippage_cost(self, position_size, entry_price):
        """Calculate slippage cost in dollars"""
        # Convert pips to price points (assuming forex-like instrument)
        pip_value = 0.0001  # Standard pip size
        slippage_points = self.slippage_pips * pip_value
        slippage_cost = position_size * slippage_points * 5  # $5 per point
        return slippage_cost

    def load_data(self, filename="1min_data_fixed.csv", sample_size=None, start_date="2010-01-01"):
        """
        Load and prepare OHLCV data with date filtering
        
        Args:
            filename: CSV file path
            sample_size: Number of recent bars to use (None for all data)
            start_date: Start date for backtesting (format: YYYY-MM-DD)
        """
        print(f"📊 Loading dataset from {filename}...")
        
        # Check file size and load appropriately
        try:
            file_size = os.path.getsize(filename) / (1024**2)  # Size in MB
            print(f"⚠️  Large dataset detected: {file_size:.1f} MB")
            
            if sample_size:
                print(f"🎯 Using sample_size parameter: {sample_size:,} bars")
                # Load specific number of rows
                try:
                    # Try to load from the end (most recent data)
                    total_lines = sum(1 for line in open(filename)) - 1  # Subtract header
                    skip_rows = max(0, total_lines - sample_size)
                    self.df = pd.read_csv(filename, skiprows=range(1, skip_rows + 1))
                except:
                    self.df = pd.read_csv(filename, nrows=sample_size)
            else:
                # Load all data
                print(f"📊 Loading complete dataset...")
                self.df = pd.read_csv(filename)
                    
        except Exception as e:
            print(f"⚠️  File size check failed, loading normally: {e}")
            self.df = pd.read_csv(filename)
        
        # Prepare data
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.set_index('date', inplace=True)
        
        # Filter by start date if provided
        if start_date:
            print(f"📅 Filtering data from {start_date} onwards...")
            self.df = self.df[self.df.index >= start_date]
            print(f"   📊 After date filtering: {len(self.df):,} bars")
        
        # Extract time components for session filtering
        self.df['hour'] = self.df.index.hour
        self.df['minute'] = self.df.index.minute
        
        print(f"✅ DATASET LOADED!")
        print(f"   📊 Total bars: {len(self.df):,}")
        print(f"   📅 Date range: {self.df.index[0]} to {self.df.index[-1]}")
        print(f"   ⏰ Time span: {(self.df.index[-1] - self.df.index[0]).days} days")
        
        return self.df
    
    def identify_fvgs_vectorized(self):
        """
        OPTIMIZED: Vectorized FVG identification for large datasets
        """
        print("🚀 Identifying FVGs using vectorized operations...")
        
        # Use numpy arrays for speed
        highs = self.df['high'].values
        lows = self.df['low'].values
        n = len(highs)
        
        print(f"   Processing {n:,} bars with vectorized operations...")
        
        # Vectorized FVG detection
        # Shift arrays to get 3-candle sequences
        h1 = highs[:-2]  # First candle highs
        h3 = highs[2:]   # Third candle highs
        l1 = lows[:-2]   # First candle lows  
        l3 = lows[2:]    # Third candle lows
        
        # Bullish FVG: h1 < l3 (gap between candle 1 high and candle 3 low)
        bullish_mask = h1 < l3
        bullish_indices = np.where(bullish_mask)[0] + 1  # Middle candle index
        
        # Bearish FVG: l1 > h3 (gap between candle 1 low and candle 3 high)  
        bearish_mask = l1 > h3
        bearish_indices = np.where(bearish_mask)[0] + 1  # Middle candle index
        
        # Create FVG dataframe
        fvg_data = []
        
        # Add bullish FVGs
        for i, idx in enumerate(bullish_indices):
            original_idx = idx  # Middle candle
            creation_bar = idx + 1  # Completion bar
            
            fvg_data.append({
                'index': original_idx,
                'creation_bar': creation_bar,
                'timestamp': self.df.index[original_idx],
                'direction': 1,  # Bullish
                'top': l3[idx-1],  # Candle 3 low
                'bottom': h1[idx-1],  # Candle 1 high
                'range': l3[idx-1] - h1[idx-1],
                'mitigated': False
            })
        
        # Add bearish FVGs
        for i, idx in enumerate(bearish_indices):
            original_idx = idx  # Middle candle
            creation_bar = idx + 1  # Completion bar
            
            fvg_data.append({
                'index': original_idx,
                'creation_bar': creation_bar, 
                'timestamp': self.df.index[original_idx],
                'direction': -1,  # Bearish
                'top': l1[idx-1],  # Candle 1 low
                'bottom': h3[idx-1],  # Candle 3 high
                'range': l1[idx-1] - h3[idx-1],
                'mitigated': False
            })
        
        # Convert to DataFrame and sort by creation order
        self.fvgs = pd.DataFrame(fvg_data)
        if len(self.fvgs) > 0:
            self.fvgs = self.fvgs.sort_values('creation_bar').reset_index(drop=True)
        
        print(f"✅ FVG identification complete in vectorized mode!")
        print(f"   📊 Total FVGs found: {len(self.fvgs):,}")
        
        if len(self.fvgs) > 0:
            bullish_fvgs = len(self.fvgs[self.fvgs['direction'] == 1])
            bearish_fvgs = len(self.fvgs[self.fvgs['direction'] == -1])
            print(f"   📈 Bullish FVGs: {bullish_fvgs:,}")
            print(f"   📉 Bearish FVGs: {bearish_fvgs:,}")
            print(f"   📏 Average gap size: {self.fvgs['range'].mean():.5f}")
        
        return self.fvgs
    
    def is_trading_session(self, bar_index):
        """ICT session filtering"""
        current_hour = self.df.iloc[bar_index]['hour']
        return 6 <= current_hour <= 16
    
    def find_entry_signals_optimized(self, current_bar):
        """
        OPTIMIZED: Signal detection with batch processing
        """
        signals = []
        
        # Must be in trading session
        if not self.is_trading_session(current_bar):
            return signals
        
        if len(self.fvgs) == 0:
            return signals
        
        current_high = self.df.iloc[current_bar]['high']
        current_low = self.df.iloc[current_bar]['low']
        current_time = self.df.index[current_bar]
        
        # Get active FVGs efficiently using boolean indexing
        min_fvg_index = max(0, current_bar - self.max_fvg_age)
        
        active_mask = (
            (~self.fvgs['mitigated']) &
            (self.fvgs['index'] >= min_fvg_index) &
            (self.fvgs['creation_bar'] < current_bar) &
            (self.fvgs['creation_bar'] <= current_bar - 1)
        )
        
        active_fvgs = self.fvgs[active_mask]
        
        if len(active_fvgs) == 0:
            return signals
        
        # Vectorized retest check
        retest_mask = (
            (current_low <= active_fvgs['top']) &
            (current_high >= active_fvgs['bottom'])
        )
        
        retesting_fvgs = active_fvgs[retest_mask]
        
        for _, fvg in retesting_fvgs.iterrows():
            entry_price = (fvg['top'] + fvg['bottom']) / 2
            fvg_age = current_bar - fvg['creation_bar']
            
            if fvg['direction'] == 1:  # Bullish FVG
                gap_size = fvg['range']
                target = entry_price + (gap_size * 1.5)
                stop_loss = fvg['bottom'] - (gap_size * 0.2)
                
                risk = abs(entry_price - stop_loss)
                reward = abs(target - entry_price)
                
                if risk > 0 and reward / risk >= self.min_rr_ratio:
                    signals.append({
                        'type': 'long',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'fvg_age': fvg_age,
                        'risk_reward': reward / risk,
                        'session': 'Trading_Hours',
                        'entry_time': current_time,
                        'fvg_creation_bar': fvg['creation_bar'],
                        'current_bar': current_bar
                    })
            
            elif fvg['direction'] == -1:  # Bearish FVG
                gap_size = fvg['range']
                target = entry_price - (gap_size * 1.5)
                stop_loss = fvg['top'] + (gap_size * 0.2)
                
                risk = abs(stop_loss - entry_price)
                reward = abs(entry_price - target)
                
                if risk > 0 and reward / risk >= self.min_rr_ratio:
                    signals.append({
                        'type': 'short',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'fvg_age': fvg_age,
                        'risk_reward': reward / risk,
                        'session': 'Trading_Hours',
                        'entry_time': current_time,
                        'fvg_creation_bar': fvg['creation_bar'],
                        'current_bar': current_bar
                    })
        
        return signals
    
    def update_fvg_mitigation_batch(self, current_bar):
        """OPTIMIZED: Batch FVG mitigation updates"""
        if len(self.fvgs) == 0:
            return
        
        current_high = self.df.iloc[current_bar]['high']
        current_low = self.df.iloc[current_bar]['low']
        
        # Vectorized mitigation check
        active_fvgs = self.fvgs[~self.fvgs['mitigated']]
        
        if len(active_fvgs) == 0:
            return
        
        # Bullish FVG mitigation: price breaks below bottom
        bullish_mitigated = (
            (active_fvgs['direction'] == 1) &
            (current_low <= active_fvgs['bottom'])
        )
        
        # Bearish FVG mitigation: price breaks above top  
        bearish_mitigated = (
            (active_fvgs['direction'] == -1) &
            (current_high >= active_fvgs['top'])
        )
        
        # Update mitigation status
        mitigated_mask = bullish_mitigated | bearish_mitigated
        mitigated_indices = active_fvgs[mitigated_mask].index
        
        if len(mitigated_indices) > 0:
            self.fvgs.loc[mitigated_indices, 'mitigated'] = True
    
    def run_optimized_backtest(self):
        """Run OPTIMIZED ICT backtest"""
        print("\n🚀 OPTIMIZED ICT BACKTEST")
        print("="*50)
        print("⚡ Vectorized FVG identification")
        print("⚡ Batch processing for speed")
        print("⚡ Memory-efficient operations")
        print("✅ FVG Entry Rule: Only from 4th candle onwards")
        print("✅ Trading Hours: 6 AM - 4 PM")
        
        if self.df is None:
            print("❌ No data loaded")
            return None
        
        total_bars = len(self.df)
        print(f"\n⚡ Processing {total_bars:,} bars with optimized algorithms...")
        
        # Identify FVGs with vectorized operations
        self.identify_fvgs_vectorized()
        
        trades_found = 0
        session_bars = 0
        progress_interval = max(1000, total_bars // 100)  # 1% intervals
        
        print(f"\n🎯 Starting optimized trade signal detection...")
        
        for bar_index in range(100, total_bars):
            
            # Batch FVG mitigation updates every 100 bars
            if bar_index % 100 == 0:
                self.update_fvg_mitigation_batch(bar_index)
            
            # Count session bars
            if self.is_trading_session(bar_index):
                session_bars += 1
            
            # Update open trades
            self.update_open_trades(bar_index)
            
            # Look for new signals ONLY if no open trades
            if len(self.open_trades) == 0:
                signals = self.find_entry_signals_optimized(bar_index)
                
                if signals:
                    # Take best R:R signal
                    best_signal = max(signals, key=lambda x: x['risk_reward'])
                    self.open_trade(best_signal, bar_index)
                    trades_found += 1
                    
                    # Log trade
                    bar_time = self.df.index[bar_index].strftime('%Y-%m-%d %H:%M')
                    creation_gap = bar_index - best_signal['fvg_creation_bar']
                    print(f"  💰 Trade {trades_found}: {best_signal['type'].upper()} @ {bar_time} "
                          f"(R:R {best_signal['risk_reward']:.2f}) [+{creation_gap} bars after FVG]")
            
            # Progress updates
            if bar_index % progress_interval == 0:
                progress = (bar_index / total_bars) * 100
                print(f"  📊 Progress: {progress:.1f}% | Session bars: {session_bars:,} | Trades: {trades_found}")
        
        # Close remaining trades
        if self.open_trades:
            final_price = self.df.iloc[-1]['close']
            for trade in self.open_trades.copy():
                trade['exit_price'] = final_price
                trade['exit_reason'] = 'end_of_data'
                self.close_trade(trade)
        
        print(f"\n🎉 OPTIMIZED ICT BACKTEST COMPLETED!")
        print(f"  📊 Total bars processed: {total_bars:,}")
        print(f"  🕐 Trading session bars: {session_bars:,} ({(session_bars/total_bars)*100:.1f}%)")
        print(f"  💰 Trades generated: {len(self.trades)}")
        print(f"  💵 Final capital: ${self.current_capital:,.2f}")
        
        return self.generate_results()
    
    def open_trade(self, signal, bar_index):
        """Open new trade with improved position sizing and cost calculation"""
        entry_time = self.df.index[bar_index]
        
        # Calculate risk amount with cap
        risk_amount = min(
            self.current_capital * self.risk_per_trade,
            self.max_risk_per_trade
        )
        
        # Position sizing based on stop loss distance
        price_risk = abs(signal['entry_price'] - signal['stop_loss'])
        if price_risk <= 0:
            return  # Invalid trade
        
        # Calculate position size (assuming $5 per point movement)
        position_size = max(1, int(risk_amount / (price_risk * 5)))
        
        # Calculate trading costs
        commission = self.calculate_ibkr_commission(position_size, signal['entry_price'])
        slippage_cost = self.calculate_slippage_cost(position_size, signal['entry_price'])
        total_entry_costs = commission + slippage_cost
        
        # Adjust entry price for slippage
        if signal['type'] == 'long':
            adjusted_entry = signal['entry_price'] + (self.slippage_pips * 0.0001)
        else:
            adjusted_entry = signal['entry_price'] - (self.slippage_pips * 0.0001)
        
        trade = {
            'id': len(self.trades) + 1,
            'entry_time': entry_time,
            'entry_bar': bar_index,
            'type': signal['type'],
            'entry_price': signal['entry_price'],
            'adjusted_entry_price': adjusted_entry,
            'stop_loss': signal['stop_loss'],
            'target': signal['target'],
            'position_size': position_size,
            'risk_reward': signal['risk_reward'],
            'fvg_age': signal['fvg_age'],
            'entry_reason': f'FVG_Retest_Age_{signal["fvg_age"]}',
            'session': signal['session'],
            'risk_amount_dollars': risk_amount,
            'fvg_creation_bar': signal['fvg_creation_bar'],
            'bars_after_creation': signal['current_bar'] - signal['fvg_creation_bar'],
            'entry_commission': commission,
            'entry_slippage': slippage_cost,
            'total_entry_costs': total_entry_costs
        }
        
        # Deduct entry costs immediately
        self.current_capital -= total_entry_costs
        
        self.open_trades.append(trade)
    
    def update_open_trades(self, bar_index):
        """Update open trades and check exits with slippage"""
        current_bar = self.df.iloc[bar_index]
        current_time = self.df.index[bar_index]
        
        trades_to_close = []
        
        for trade in self.open_trades:
            if trade['type'] == 'long':
                if current_bar['high'] >= trade['target']:
                    # Apply slippage on exit (against trader)
                    exit_price = trade['target'] - (self.slippage_pips * 0.0001)
                    trade['exit_price'] = exit_price
                    trade['exit_reason'] = 'target'
                    trade['exit_time'] = current_time
                    trades_to_close.append(trade)
                elif current_bar['low'] <= trade['stop_loss']:
                    # Apply slippage on stop loss (against trader)
                    exit_price = trade['stop_loss'] - (self.slippage_pips * 0.0001)
                    trade['exit_price'] = exit_price
                    trade['exit_reason'] = 'stop'
                    trade['exit_time'] = current_time
                    trades_to_close.append(trade)
            else:  # short
                if current_bar['low'] <= trade['target']:
                    # Apply slippage on exit (against trader)
                    exit_price = trade['target'] + (self.slippage_pips * 0.0001)
                    trade['exit_price'] = exit_price
                    trade['exit_reason'] = 'target'
                    trade['exit_time'] = current_time
                    trades_to_close.append(trade)
                elif current_bar['high'] >= trade['stop_loss']:
                    # Apply slippage on stop loss (against trader)
                    exit_price = trade['stop_loss'] + (self.slippage_pips * 0.0001)
                    trade['exit_price'] = exit_price
                    trade['exit_reason'] = 'stop'
                    trade['exit_time'] = current_time
                    trades_to_close.append(trade)
        
        for trade in trades_to_close:
            self.close_trade(trade)
        
        # Track equity curve
        self.equity_curve.append({
            'timestamp': current_time,
            'equity': self.current_capital,
            'bar_index': bar_index
        })
    
    def close_trade(self, trade):
        """Close trade and calculate P&L with all costs included"""
        # Calculate exit commission and slippage
        exit_commission = self.calculate_ibkr_commission(trade['position_size'], trade['exit_price'])
        exit_slippage = self.calculate_slippage_cost(trade['position_size'], trade['exit_price'])
        total_exit_costs = exit_commission + exit_slippage
        
        # Calculate gross P&L
        if trade['type'] == 'long':
            pnl_points = trade['exit_price'] - trade['adjusted_entry_price']
        else:
            pnl_points = trade['adjusted_entry_price'] - trade['exit_price']
        
        gross_pnl_dollars = pnl_points * trade['position_size'] * 5
        
        # Calculate net P&L after all costs
        total_costs = trade['total_entry_costs'] + total_exit_costs
        net_pnl_dollars = gross_pnl_dollars - total_exit_costs  # Entry costs already deducted
        
        trade['pnl_points'] = pnl_points
        trade['gross_pnl_dollars'] = gross_pnl_dollars
        trade['exit_commission'] = exit_commission
        trade['exit_slippage'] = exit_slippage
        trade['total_exit_costs'] = total_exit_costs
        trade['total_costs'] = total_costs
        trade['net_pnl_dollars'] = net_pnl_dollars
        
        self.current_capital += net_pnl_dollars
        
        if trade in self.open_trades:
            self.open_trades.remove(trade)
        self.trades.append(trade)
    
    def calculate_advanced_metrics(self, trades_df):
        """Calculate comprehensive trading metrics"""
        if len(trades_df) == 0:
            return {}
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['net_pnl_dollars'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L metrics
        total_gross_pnl = trades_df['gross_pnl_dollars'].sum()
        total_costs = trades_df['total_costs'].sum()
        total_net_pnl = trades_df['net_pnl_dollars'].sum()
        
        # Win/Loss analysis
        winning_trades_df = trades_df[trades_df['net_pnl_dollars'] > 0]
        losing_trades_df = trades_df[trades_df['net_pnl_dollars'] < 0]
        
        avg_win = winning_trades_df['net_pnl_dollars'].mean() if len(winning_trades_df) > 0 else 0
        avg_loss = losing_trades_df['net_pnl_dollars'].mean() if len(losing_trades_df) > 0 else 0
        
        largest_win = winning_trades_df['net_pnl_dollars'].max() if len(winning_trades_df) > 0 else 0
        largest_loss = losing_trades_df['net_pnl_dollars'].min() if len(losing_trades_df) > 0 else 0
        
        # Risk metrics
        profit_factor = abs(winning_trades_df['net_pnl_dollars'].sum() / losing_trades_df['net_pnl_dollars'].sum()) if len(losing_trades_df) > 0 and losing_trades_df['net_pnl_dollars'].sum() != 0 else float('inf')
        
        # Calculate daily returns for Sharpe ratio
        if len(self.equity_curve) > 0:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            equity_df['date'] = equity_df['timestamp'].dt.date
            
            daily_equity = equity_df.groupby('date')['equity'].last().reset_index()
            daily_equity['daily_return'] = daily_equity['equity'].pct_change()
            daily_returns = daily_equity['daily_return'].dropna()
            
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
            
            # Calculate maximum drawdown
            cumulative = (1 + daily_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Trading costs analysis
        avg_commission_per_trade = trades_df['entry_commission'].mean() + trades_df['exit_commission'].mean()
        avg_slippage_per_trade = trades_df['entry_slippage'].mean() + trades_df['exit_slippage'].mean()
        cost_ratio = (total_costs / abs(total_gross_pnl)) * 100 if total_gross_pnl != 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_gross_pnl': total_gross_pnl,
            'total_costs': total_costs,
            'total_net_pnl': total_net_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_commission_per_trade': avg_commission_per_trade,
            'avg_slippage_per_trade': avg_slippage_per_trade,
            'cost_ratio': cost_ratio,
            'total_return': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        }
    
    def plot_equity_curve(self):
        """Create comprehensive equity curve visualization"""
        if len(self.equity_curve) == 0:
            print("❌ No equity data available for plotting")
            return
        
        # Prepare data
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ICT Strategy Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Equity Curve
        ax1.plot(equity_df['timestamp'], equity_df['equity'], linewidth=2, color='#2E86AB')
        ax1.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        ax1.set_title('Equity Curve', fontweight='bold')
        ax1.set_ylabel('Account Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Format y-axis as currency
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Drawdown
        equity_values = equity_df['equity'].values
        peak = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - peak) / peak * 100
        
        ax2.fill_between(equity_df['timestamp'], drawdown, 0, alpha=0.7, color='red')
        ax2.set_title('Drawdown %', fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Trade Distribution
        if len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            pnl_values = trades_df['net_pnl_dollars']
            
            ax3.hist(pnl_values, bins=30, alpha=0.7, color='#A23B72', edgecolor='black')
            ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax3.set_title('Trade P&L Distribution', fontweight='bold')
            ax3.set_xlabel('Net P&L ($)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # 4. Monthly Returns Heatmap
        equity_df['year'] = equity_df['timestamp'].dt.year
        equity_df['month'] = equity_df['timestamp'].dt.month
        
        # Calculate monthly returns
        monthly_data = []
        for year in equity_df['year'].unique():
            for month in range(1, 13):
                month_data = equity_df[(equity_df['year'] == year) & (equity_df['month'] == month)]
                if len(month_data) > 0:
                    if month == 1:
                        # First month, calculate from initial capital
                        prev_value = self.initial_capital
                        if year > equity_df['year'].min():
                            # Get last value from previous year
                            prev_year_data = equity_df[equity_df['year'] == year-1]
                            if len(prev_year_data) > 0:
                                prev_value = prev_year_data['equity'].iloc[-1]
                    else:
                        # Get last value from previous month
                        prev_month_data = equity_df[(equity_df['year'] == year) & (equity_df['month'] == month-1)]
                        if len(prev_month_data) > 0:
                            prev_value = prev_month_data['equity'].iloc[-1]
                        else:
                            continue
                    
                    current_value = month_data['equity'].iloc[-1]
                    monthly_return = ((current_value - prev_value) / prev_value) * 100
                    monthly_data.append({'year': year, 'month': month, 'return': monthly_return})
        
        if monthly_data:
            monthly_df = pd.DataFrame(monthly_data)
            monthly_pivot = monthly_df.pivot(index='year', columns='month', values='return')
            
            sns.heatmap(monthly_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                       ax=ax4, cbar_kws={'label': 'Return (%)'})
            ax4.set_title('Monthly Returns Heatmap (%)', fontweight='bold')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Year')
        
        plt.tight_layout()
        plt.savefig('ict_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Performance charts saved as 'ict_performance_analysis.png'")

    def generate_results(self):
        """Generate comprehensive performance report with advanced metrics"""
        if not self.trades:
            print("\n❌ No trades generated")
            return None
        
        trades_df = pd.DataFrame(self.trades)
        metrics = self.calculate_advanced_metrics(trades_df)
        
        print(f"\n📊 COMPREHENSIVE ICT BACKTEST RESULTS")
        print("="*60)
        print(f"🎯 STRATEGY COMPLIANCE:")
        print(f"   ✅ All entries ≥1 bar after FVG creation")
        print(f"   ✅ Proper 3-candle FVG formation")
        print(f"   ✅ Entry only on retest")
        print(f"   ✅ IBKR commission structure applied")
        print(f"   ✅ {self.slippage_pips} pip slippage included")
        print(f"   ✅ Maximum ${self.max_risk_per_trade} risk per trade")
        
        print(f"\n💰 PERFORMANCE SUMMARY:")
        print(f"   Initial Capital: ${self.initial_capital:,.2f}")
        print(f"   Final Capital: ${self.current_capital:,.2f}")
        print(f"   Total Return: {metrics['total_return']:+.2f}%")
        print(f"   Gross P&L: ${metrics['total_gross_pnl']:,.2f}")
        print(f"   Total Costs: ${metrics['total_costs']:,.2f}")
        print(f"   Net P&L: ${metrics['total_net_pnl']:,.2f}")
        
        print(f"\n📈 TRADE STATISTICS:")
        print(f"   Total Trades: {metrics['total_trades']:,}")
        print(f"   Winning Trades: {metrics['winning_trades']:,}")
        print(f"   Losing Trades: {metrics['losing_trades']:,}")
        print(f"   Win Rate: {metrics['win_rate']:.1f}%")
        print(f"   Average Win: ${metrics['avg_win']:,.2f}")
        print(f"   Average Loss: ${metrics['avg_loss']:,.2f}")
        print(f"   Largest Win: ${metrics['largest_win']:,.2f}")
        print(f"   Largest Loss: ${metrics['largest_loss']:,.2f}")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        
        print(f"\n📊 RISK METRICS:")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"   Average R:R: {trades_df['risk_reward'].mean():.2f}")
        
        print(f"\n💸 TRADING COSTS ANALYSIS:")
        print(f"   Average Commission/Trade: ${metrics['avg_commission_per_trade']:.2f}")
        print(f"   Average Slippage/Trade: ${metrics['avg_slippage_per_trade']:.2f}")
        print(f"   Cost Ratio: {metrics['cost_ratio']:.2f}% of gross P&L")
        
        # Timing analysis
        print(f"\n⏰ FVG TIMING ANALYSIS:")
        avg_bars_after = trades_df['bars_after_creation'].mean()
        min_bars_after = trades_df['bars_after_creation'].min()
        max_bars_after = trades_df['bars_after_creation'].max()
        print(f"   Average bars after FVG creation: {avg_bars_after:.1f}")
        print(f"   Minimum bars after creation: {min_bars_after}")
        print(f"   Maximum bars after creation: {max_bars_after}")
        
        # Exit reason analysis
        exit_reasons = trades_df['exit_reason'].value_counts()
        print(f"\n🎯 EXIT ANALYSIS:")
        for reason, count in exit_reasons.items():
            percentage = (count / len(trades_df)) * 100
            print(f"   {reason.title()}: {count} trades ({percentage:.1f}%)")
        
        # Time analysis
        trades_df['entry_hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
        hourly_trades = trades_df['entry_hour'].value_counts().sort_index()
        print(f"\n🕘 HOURLY DISTRIBUTION:")
        for hour, count in hourly_trades.items():
            print(f"   {hour:02d}:00: {count} trades")
        
        # Export results
        trades_df.to_csv('optimized_ict_trades_with_costs.csv', index=False)
        
        # Export equity curve
        if len(self.equity_curve) > 0:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.to_csv('ict_equity_curve.csv', index=False)
            print(f"\n📁 Exported: ict_equity_curve.csv")
        
        print(f"📁 Exported: optimized_ict_trades_with_costs.csv")
        
        # Generate performance charts
        print(f"\n📊 Generating performance visualizations...")
        self.plot_equity_curve()
        
        return trades_df

def main():
    """Run optimized ICT backtester with realistic trading costs"""
    print("🚀 OPTIMIZED ICT BACKTESTER WITH TRADING COSTS")
    print("="*60)
    print("⚡ Handles large datasets efficiently")
    print("⚡ Vectorized FVG identification") 
    print("⚡ Batch processing optimizations")
    print("💰 IBKR commission structure")
    print("📉 Realistic slippage modeling")
    print("🛡️  $500 maximum risk per trade")
    print("📊 Comprehensive performance metrics")
    print("📈 Equity curve visualization")
    print("🎯 PROPER FVG TIMING RULES:")
    print("   ✅ 3 candles create FVG")
    print("   ✅ Entry only from 4th candle onwards")
    print("   ✅ Price must retest FVG area")
    
    backtester = OptimizedICTBacktester(
        initial_capital=25000,
        risk_per_trade=0.02,
        min_rr_ratio=1.5
    )
    
    # Load data starting from 2010
    print(f"\n📅 STARTING BACKTEST FROM 2010")
    backtester.load_data("1min_data_fixed.csv", start_date="2010-01-01")
    results = backtester.run_optimized_backtest()
    
    if results is not None:
        print(f"\n🎉 Complete ICT backtest with trading costs completed successfully!")
        print(f"⚡ Processed efficiently with vectorized operations")
        print(f"📅 Full historical analysis from 2010 onwards")
        print(f"💰 Realistic IBKR commissions and slippage applied")
        print(f"📊 Comprehensive metrics and equity curve generated")
    else:
        print(f"\n⚠️  No valid FVG retests found")

if __name__ == "__main__":
    main() 