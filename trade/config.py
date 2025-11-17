"""
Configuration settings for the stock options trading assistant
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')

# Technical Indicator Settings
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BB_PERIOD = 20
BB_STD = 2

MA_SHORT = 50
MA_LONG = 200

# Sentiment Analysis Settings
NEWS_LOOKBACK_DAYS = 7
SENTIMENT_THRESHOLD_POSITIVE = 0.1
SENTIMENT_THRESHOLD_NEGATIVE = -0.1

# Options Settings
DEFAULT_DTE_MIN = 30  # Minimum days to expiration
DEFAULT_DTE_MAX = 60  # Maximum days to expiration
IV_PERCENTILE_HIGH = 75  # High IV percentile threshold
IV_PERCENTILE_LOW = 25   # Low IV percentile threshold

# Recommendation Weights
WEIGHT_TECHNICAL = 0.4
WEIGHT_SENTIMENT = 0.3
WEIGHT_FUNDAMENTAL = 0.2
WEIGHT_OPTIONS = 0.1

# Confidence Thresholds
CONFIDENCE_HIGH = 0.7
CONFIDENCE_MEDIUM = 0.5
CONFIDENCE_LOW = 0.3
