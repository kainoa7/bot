# Options Trading Assistant

A comprehensive web-based options trading assistant that provides intelligent analysis and position management tools for options traders.

## Features

### 🆕 New Trade Analysis
Analyze any stock ticker to get intelligent Call/Put recommendations with:
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Moving Averages
- **Sentiment Analysis**: News sentiment from multiple sources
- **Fundamental Analysis**: P/E ratios, revenue growth, profit margins
- **Options Market Data**: Implied volatility, put-call ratios, Greeks
- **Strike Price Recommendations**: ITM, ATM, and OTM options with detailed analysis
- **Expiration Date Guidance**: Optimal expiration dates based on earnings and technicals
- **Risk Assessment**: Comprehensive risk factor identification

### 📊 Position Management
Deep analysis for existing options positions with actionable insights:

#### Critical Insights Card
- Clear HOLD/SELL/ROLL recommendation with confidence percentage
- Key reasoning in 1-2 sentences
- Urgency indicators (green/yellow/red)

#### News Intelligence Card
- Top 5-7 recent news articles with sentiment scores
- Breaking news alerts from last 24 hours
- Critical news detection (earnings, FDA, lawsuits, etc.)
- News momentum tracking (improving/deteriorating/stable)
- Overall news impact score

#### Price Trend Analysis Card
- 7-day, 14-day, and 30-day price momentum
- Support and resistance levels
- Distance from key moving averages (20-day, 50-day, 200-day)
- Volatility trend analysis (increasing/decreasing)
- Price performance metrics

#### Earnings Intelligence Card
- Days until earnings with urgency indicator
- Historical earnings move analysis (average % move)
- IV crush risk assessment (pre vs post-earnings IV)
- Earnings surprise history (beat/miss pattern)
- Hold-through or exit-before recommendations

#### Technical Health Card
- RSI trend (rising/falling)
- MACD momentum strength
- Bollinger Band squeeze detection
- Volume analysis (above/below average)
- Trend strength score (0-100)
- Overall technical direction

#### Exit Strategy Card
- Specific price targets (take profit, stop loss)
- Time-based exit recommendations
- Risk/reward ratio for holding vs exiting now
- Multiple exit scenarios with reasoning

#### Action Plan
- Prioritized action items (high/medium/low)
- Time-sensitive alerts
- Price target alerts
- Earnings-related actions

#### Alternative Strategies
- Roll out (extend time)
- Roll up/down (adjust strike)
- Convert to spreads
- Cut and redeploy options

## Setup

1. **Clone the repository**
   ```bash
   cd trade
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys (Optional)**
   - Copy `.env.example` to `.env` if you have one
   - Add your API keys for enhanced news data (NewsAPI)
   - The app works without API keys using yfinance data

5. **Run the web server**
   ```bash
   python app.py
   ```

6. **Open in browser**
   Navigate to: http://localhost:5001

## Usage

### New Trade Analysis
1. Enter a ticker symbol (e.g., AAPL, TSLA, NVDA)
2. Click "Analyze" to get comprehensive trade recommendations
3. Review:
   - Recommended direction (CALL/PUT)
   - Strike price options with Greeks
   - Optimal expiration dates
   - Risk factors and component scores

### Position Management
1. Fill in your position details:
   - Ticker symbol
   - Position type (Call/Put)
   - Strike price
   - Premium paid
   - Number of contracts
   - Entry date
   - Expiration date
   - (Optional) Entry price and notes
2. Click "Analyze Position"
3. Review comprehensive analysis:
   - Clear recommendation with confidence
   - News intelligence and momentum
   - Price trends and technicals
   - Earnings intelligence
   - Exit strategy with targets
   - Action plan

## Project Structure

```
trade/
├── app.py                  # Flask web server
├── config.py              # Configuration settings
├── data_collector.py      # Market data fetching
├── indicators.py          # Technical analysis
├── sentiment_analyzer.py # News sentiment analysis
├── recommendation.py      # New trade recommendations
├── position_analyzer.py  # Existing position analysis
├── options_calculator.py # Greeks calculations
├── templates/
│   └── index.html        # Main UI
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Key Features Explained

### News Intelligence
- **Breaking News**: Articles from the last 24 hours
- **Critical News**: Identifies earnings, FDA approvals, lawsuits, mergers, etc.
- **News Momentum**: Tracks sentiment trend over time (improving/deteriorating)
- **Sentiment Scores**: Each article analyzed for positive/negative/neutral sentiment

### Price Trend Analysis
- **Momentum**: Calculates price change over 7, 14, and 30 days
- **Support/Resistance**: Identifies key price levels
- **Moving Averages**: Distance from 20, 50, and 200-day MAs
- **Volatility Trend**: Tracks whether volatility is increasing or decreasing

### Earnings Intelligence
- **Historical Data**: Analyzes past earnings beats/misses
- **IV Crush Risk**: Assesses risk of volatility collapse after earnings
- **Urgency Levels**: CRITICAL (≤3 days), HIGH (≤7 days), MEDIUM (≤14 days), LOW (>14 days)
- **Recommendations**: Whether to hold through or exit before earnings

### Technical Health
- **RSI Trend**: Whether RSI is rising or falling
- **MACD Strength**: Strong or weak momentum signals
- **Bollinger Squeeze**: Detects low volatility periods (potential breakouts)
- **Volume Analysis**: Above or below average volume
- **Trend Strength**: 0-100 score indicating trend strength

### Exit Strategy
- **Take Profit Targets**: Multiple price levels with reasoning
- **Stop Loss Targets**: Risk management price levels
- **Time-Based Exits**: Dates to exit before expiration or earnings
- **Risk/Reward**: Calculated ratio for holding vs exiting

## Disclaimer

⚠️ **This tool is for educational purposes only. It does not guarantee profits and should not be considered financial advice. Always do your own research and consult with a financial advisor before making trading decisions. Options trading involves significant risk and is not suitable for all investors.**

## Technology Stack

- **Backend**: Python, Flask
- **Data Sources**: yfinance, NewsAPI (optional)
- **Analysis**: NumPy, SciPy, pandas, TextBlob
- **Frontend**: HTML, CSS, JavaScript

## License

This project is provided as-is for educational purposes.
