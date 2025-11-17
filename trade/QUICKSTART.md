# Quick Start Guide

## Installation

1. **Run the setup script:**
   ```bash
   ./setup.sh
   ```

2. **Configure API keys:**
   Edit the `.env` file and add your keys:
   ```bash
   ALPHA_VANTAGE_API_KEY=your_key_here
   NEWS_API_KEY=your_key_here
   ```
   
   Get free API keys from:
   - Alpha Vantage: https://www.alphavantage.co/support/#api-key
   - NewsAPI: https://newsapi.org/register

3. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

## Usage

### Basic Analysis
```bash
python main.py AAPL
```

### Detailed Analysis
```bash
python main.py TSLA --verbose
```

### Example Output

```
======================================================================
  ANALYZING AAPL
======================================================================

📊 Collecting market data...
  ✓ Price data: 126 days
  ✓ Fundamental data retrieved
  ✓ News articles: 10
  ✓ Options data available
  ✓ Next earnings: 2025-11-28 (13 days)

📈 Running technical analysis...
  ✓ Technical direction: BULLISH
  ✓ Technical score: 0.65

📰 Analyzing news sentiment...
  ✓ Sentiment: POSITIVE
  ✓ Sentiment score: 0.32

🤖 Generating recommendation...

======================================================================
  TRADING RECOMMENDATION
======================================================================

🎯 Ticker: AAPL
💰 Current Price: $185.50

📍 RECOMMENDATION: CALL
   Direction: BULLISH
   Confidence: HIGH (73%)

💵 SUGGESTED STRIKE PRICES
Strategy: Slightly out-of-the-money
  1. $189.21 - Conservative (2% OTM)
  2. $194.78 - Moderate (5% OTM)
  3. $198.45 - Aggressive (Upper BB)

📅 EXPIRATION RECOMMENDATION
Timeframe: 30-60 days
Date Range: 2025-12-15 to 2026-01-14
Reason: Standard medium-term expiration for directional trades

⚠️  RISK FACTORS
  • ⚠️ Earnings in 13 days - high volatility risk
  • ✓ No major risk factors identified
```

## What the Tool Analyzes

### 1. Technical Indicators
- **RSI (Relative Strength Index)**: Identifies overbought/oversold conditions
- **MACD**: Detects momentum changes and trend direction
- **Bollinger Bands**: Shows volatility and potential breakouts
- **Moving Averages**: Identifies trend direction (50-day and 200-day)

### 2. News Sentiment
- Analyzes recent news articles about the stock
- Calculates overall sentiment (positive/negative/neutral)
- Weighs recent news more heavily

### 3. Fundamental Data
- P/E ratio and valuation metrics
- Revenue and earnings growth
- Profit margins
- Analyst recommendations

### 4. Options Market Data
- Put/Call ratio (market sentiment)
- Implied volatility (option prices)
- Options volume analysis

## Understanding the Recommendation

### Confidence Levels
- **HIGH (>70%)**: Strong signals across multiple indicators
- **MEDIUM (50-70%)**: Moderate agreement between indicators
- **LOW (<50%)**: Weak or mixed signals

### Strike Price Strategies
- **Conservative**: 2% out-of-the-money, higher probability
- **Moderate**: 5% out-of-the-money, balanced risk/reward
- **Aggressive**: Based on technical targets, higher risk/reward

### Expiration Dates
The tool considers:
- Time needed for the move to develop
- Upcoming earnings dates
- Current implied volatility
- Standard recommendation: 30-60 days

## Tips for Using the Tool

1. **Run analysis on multiple tickers** to compare opportunities
2. **Use verbose mode** (`--verbose`) to see all signals
3. **Check before and after market events** (earnings, Fed announcements)
4. **Compare with your own analysis** - never trade based solely on one tool
5. **Monitor risk factors** carefully, especially earnings dates

## Important Notes

⚠️ **This tool:**
- Does NOT guarantee profits
- Should be used alongside your own research
- Is for educational purposes only
- Cannot predict unexpected market events
- Assumes you understand options trading risks

✅ **Best practices:**
- Always paper trade first to test strategies
- Never risk more than you can afford to lose
- Understand options Greeks (Delta, Theta, Vega)
- Keep position sizes reasonable (1-5% of portfolio per trade)
- Have an exit plan before entering any trade

## Troubleshooting

### "No data found" error
- Check if ticker symbol is correct
- Verify internet connection
- Ensure API keys are configured

### Import errors
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt`

### API rate limits
- Free API keys have rate limits
- Wait a few seconds between requests
- Consider upgrading API plans for frequent use

## Next Steps

1. Test with well-known tickers (AAPL, MSFT, TSLA, SPY)
2. Compare recommendations against real market movements
3. Keep a trading journal to track accuracy
4. Customize indicator weights in `config.py` based on your findings
5. Add your own technical indicators or data sources

Happy trading! 📈
