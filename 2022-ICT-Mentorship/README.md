# ICT Smart Money Concepts Implementation for NQ Futures

This repository implements the ICT (Inner Circle Trader) 2022 Mentorship Trading Model using the `smartmoneyconcepts` Python library with your NQ 1-minute data.

## 📋 Overview

Based on the ICT 2022 Mentorship, this implementation provides:

- **Fair Value Gaps (FVG)** analysis and identification
- **Liquidity Pool** detection and targeting
- **Order Block** identification
- **Market Structure** analysis (BOS/CHoCH)
- **Session-based** analysis (NY Killzone)
- **Trading Setup** identification following ICT methodology

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Quick Analysis**
   ```bash
   python quick_start.py
   ```

3. **Full Analysis**
   ```bash
   python smart_money_analysis.py
   ```

4. **Interactive Analysis**
   ```bash
   jupyter notebook smart_money_notebook.ipynb
   ```

## 📁 Files Structure

```
├── requirements.txt                          # Python dependencies
├── quick_start.py                           # Quick analysis script
├── smart_money_analysis.py                  # Complete analysis system
├── smart_money_notebook.ipynb               # Interactive Jupyter notebook
├── glbx-mdp3-20100606-20250513.ohlcv-1m.csv # Your NQ 1-minute data (699MB)
├── ICT_2022_Mentorship_Trading_Model_Summary.md # Comprehensive ICT concepts summary
├── 2022-ICT-Mentorship-Episode-*.json       # Original transcription files
└── README.md                                # This file
```

## 🎯 ICT Trading Model Implementation

### Core Components

1. **Market Bias Determination**
   - Higher timeframe trend analysis
   - Liquidity pool identification
   - Previous high/low levels

2. **Setup Identification**
   - Fair Value Gap presence
   - Session timing (NY Killzone)
   - Market structure alignment

3. **Entry Execution**
   - FVG-based entries
   - Liquidity targeting
   - Risk management

4. **Trade Management**
   - Partial profit taking
   - Stop loss management
   - Position sizing

### Key Concepts Implemented

#### Fair Value Gaps (FVG)
- Algorithmic detection of price imbalances
- Bullish and bearish gap identification
- Mitigation tracking
- Entry zone calculation

#### Liquidity Pools
- Buy-side liquidity (above previous highs)
- Sell-side liquidity (below previous lows)
- Sweep detection
- Target calculation

#### Market Structure
- Swing high/low identification
- Break of Structure (BOS)
- Change of Character (CHoCH)
- Trend determination

#### Session Analysis
- NY AM Session (8:30-11:00 AM EST) - Primary Killzone
- Lunch Hour (12:00-1:00 PM EST)
- NY PM Session (1:30-4:00 PM EST)

## 📊 Usage Examples

### Quick Analysis
```python
from smart_money_analysis import SmartMoneyAnalyzer

# Initialize analyzer
analyzer = SmartMoneyAnalyzer()

# Run complete analysis
analyzer.run_complete_analysis(
    start_date="2024-01-01",
    max_rows=50000,
    lookback_periods=1000
)
```

### Custom Analysis
```python
# Load data
df = analyzer.load_and_prepare_data("2024-01-01", None, 25000)

# Calculate indicators
indicators = analyzer.calculate_indicators()

# Find setups
setups = analyzer.identify_trading_setups(500)

# Create chart
fig = analyzer.create_analysis_chart(1000)
```

## 🎨 Visualization

The system generates interactive charts showing:
- Candlestick price action
- Fair Value Gaps (colored rectangles)
- Order Blocks
- Swing highs and lows
- Liquidity levels
- BOS/CHoCH markers
- Volume analysis

Charts are saved as `smart_money_chart.html` for interactive viewing.

## 📈 Trading Setup Criteria

Following the ICT model, setups are identified when:

1. **Fair Value Gap exists** (primary entry mechanism)
2. **Valid trading session** (NY Killzone preferred)
3. **Liquidity target available** (opposite side)
4. **Market structure alignment** (trend confirmation)

### Entry Rules
- Enter when price returns to FVG
- Use FVG boundaries for entry zone
- Target opposite liquidity pool
- Maintain 2:1 minimum risk-reward

### Risk Management
- Stop beyond FVG or structure
- Position sizing based on account risk
- Start with micro contracts (MNQ)
- Paper trade until consistent

## 🔧 Configuration

### Data Parameters
```python
# In smart_money_analysis.py
START_DATE = "2024-01-01"  # Analysis start date
END_DATE = None            # None for all data
MAX_ROWS = 50000          # Performance limit
SWING_LENGTH = 50         # Structure sensitivity
```

### Indicator Settings
```python
# Fair Value Gaps
join_consecutive=True      # Merge consecutive FVGs

# Liquidity
range_percent=0.01        # 1% range for liquidity zones

# Sessions
session="New York kill zone"  # Primary trading session
```

## 📚 Educational Resources

### ICT Concepts Summary
Read `ICT_2022_Mentorship_Trading_Model_Summary.md` for:
- Complete methodology explanation
- Core concept definitions
- Trading rules and principles
- Risk management guidelines

### Transcription Files
Original ICT lessons in JSON format:
- 47 episode transcriptions
- Detailed explanations of concepts
- Real-time trading examples
- Student Q&A sessions

## 🎯 Next Steps

1. **Study the Theory**
   - Read the ICT summary document
   - Review transcription files
   - Understand market manipulation concepts

2. **Practice Analysis**
   - Run daily analysis on recent data
   - Identify setups using the criteria
   - Study confluence zones

3. **Paper Trading**
   - Practice with identified setups
   - Focus on FVG entries only
   - Track performance metrics

4. **Gradual Implementation**
   - Start with micro contracts
   - Build confidence through consistency
   - Gradually increase position size

## ⚠️ Important Notes

### Risk Disclaimer
- This is for educational purposes only
- Past performance doesn't guarantee future results
- Always use proper risk management
- Start with paper trading

### Model Expectations
- 60-70% win rate is realistic
- Requires months of practice
- Psychology is as important as technique
- Patience and discipline are crucial

### Performance Tips
- Limit data size for faster processing
- Use recent data for current relevance
- Focus on NY session for best setups
- Quality over quantity in trade selection

## 🛠️ Troubleshooting

### Common Issues

1. **Import Error: smartmoneyconcepts**
   ```bash
   pip install smartmoneyconcepts
   ```

2. **Data Loading Issues**
   - Check CSV file format
   - Ensure column names match OHLCV
   - Verify date format

3. **Memory Issues**
   - Reduce MAX_ROWS parameter
   - Use smaller lookback periods
   - Close other applications

4. **Slow Performance**
   - Limit data to recent periods
   - Use smaller swing_length
   - Reduce chart lookback_periods

### Support
- Check the GitHub repository: https://github.com/joshyattridge/smart-money-concepts
- Review ICT educational materials
- Practice with paper trading platforms

## 📊 Generated Outputs

After running the analysis, you'll have:

1. **smart_money_chart.html** - Interactive price chart with all indicators
2. **trading_setups.csv** - Identified trading opportunities
3. **active_fvgs.csv** - Current unmitigated Fair Value Gaps
4. **recent_liquidity.csv** - Current liquidity levels
5. **Console report** - Comprehensive analysis summary

## 🎓 Learning Path

1. **Week 1-2**: Study ICT concepts and theory
2. **Week 3-4**: Run analysis and identify patterns
3. **Week 5-8**: Paper trade identified setups
4. **Week 9-12**: Refine understanding and build consistency
5. **Month 4+**: Consider live trading with micro contracts

Remember: The goal is to understand market manipulation and develop a mechanical, rule-based approach to trading following ICT's methodology.

---

*Based on ICT 2022 Mentorship - A simplified trading model for index futures using smart money concepts.* 