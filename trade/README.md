# Stock Options Trading Assistant

An intelligent trading assistant that analyzes stock data, technical indicators, news sentiment, and fundamental metrics to provide informed Call/Put recommendations for options trading.

## Features

### Core Analysis
- 📊 Real-time stock price and volume analysis
- 📈 Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- 📰 News sentiment analysis from multiple sources
- 💡 Call/Put recommendations with confidence scores
- 🎯 Strike price and expiration date suggestions
- ⚠️ Risk factor identification

### Advanced Options Tools
- 🧮 **Greeks Calculation** - Delta, Gamma, Theta, Vega for each strike
- 📊 **Probability of Profit** - Statistical probability based on Black-Scholes
- 💰 **Premium Estimates** - Theoretical option pricing
- 🎲 **Breakeven Analysis** - Exact price targets needed for profitability
- ⏱️ **Time Decay** - Daily Theta loss calculations
- 📉 **IV Analysis** - Implied volatility percentile tracking

### Multiple Interfaces
- 💻 **CLI** - Fast command-line analysis
- 🌐 **Web UI** - Beautiful visual interface at http://localhost:5001
- 🔄 **Comparison Tool** - Compare multiple tickers side-by-side

## Setup

1. **Clone the repository**
   ```bash
   cd /Users/kainoa/development/stock-proj
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**
   - Copy `.env.example` to `.env`
   - Add your API keys:
     - Alpha Vantage: https://www.alphavantage.co/support/#api-key
     - NewsAPI: https://newsapi.org/register

5. **Run the assistant**
   ```bash
   python main.py
   ```

## Project Structure

```
stock-proj/
├── main.py                 # CLI entry point
├── config.py              # Configuration settings
├── data_collector.py      # Market data fetching
├── indicators.py          # Technical analysis
├── sentiment_analyzer.py  # News sentiment analysis
├── recommendation.py      # Trade recommendation engine
├── requirements.txt       # Python dependencies
├── .env.example          # Example environment variables
└── README.md             # This file
```

## Usage

### Quick Analysis (CLI)
```bash
python main.py AAPL
python main.py TSLA --verbose  # Detailed analysis with all signals
```

### Compare Multiple Tickers
```bash
python compare.py AAPL MSFT NVDA TSLA
```
Ranks tickers by best trading opportunity

### Web Interface
```bash
python app.py
```
Then open http://localhost:5001 in your browser

### What You Get

**For Each Ticker:**
- Trade direction (CALL/PUT) with confidence level
- **3 Strike price options** with Greeks:
  - Estimated premium cost
  - Delta (position sensitivity)
  - Theta (daily time decay)
  - Probability of profit
  - Breakeven price
- Recommended expiration dates based on:
  - Earnings dates
  - Technical timeframes
  - IV levels
- Complete risk analysis
- All technical/fundamental/sentiment signals

## Disclaimer

⚠️ **This tool is for educational purposes only. It does not guarantee profits and should not be considered financial advice. Always do your own research and consider your risk tolerance before trading options.**
