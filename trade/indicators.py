"""
Technical indicators calculator
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import config


class TechnicalAnalyzer:
    """Calculates technical indicators and provides trading signals"""
    
    def __init__(self, price_data: pd.DataFrame):
        self.df = price_data.copy()
        self.signals = {}
        
    def calculate_rsi(self, period: int = None) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            period: RSI period (default from config)
            
        Returns:
            Series with RSI values
        """
        period = period or config.RSI_PERIOD
        
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        exp1 = self.df['Close'].ewm(span=config.MACD_FAST, adjust=False).mean()
        exp2 = self.df['Close'].ewm(span=config.MACD_SLOW, adjust=False).mean()
        
        macd = exp1 - exp2
        signal = macd.ewm(span=config.MACD_SIGNAL, adjust=False).mean()
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def calculate_bollinger_bands(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle = self.df['Close'].rolling(window=config.BB_PERIOD).mean()
        std = self.df['Close'].rolling(window=config.BB_PERIOD).std()
        
        upper = middle + (std * config.BB_STD)
        lower = middle - (std * config.BB_STD)
        
        return upper, middle, lower
    
    def calculate_moving_averages(self) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate moving averages
        
        Returns:
            Tuple of (Short MA, Long MA)
        """
        ma_short = self.df['Close'].rolling(window=config.MA_SHORT).mean()
        ma_long = self.df['Close'].rolling(window=config.MA_LONG).mean()
        
        return ma_short, ma_long
    
    def calculate_all_indicators(self) -> Dict:
        """
        Calculate all technical indicators and return current values
        
        Returns:
            Dictionary with all indicator values
        """
        # RSI
        rsi = self.calculate_rsi()
        current_rsi = rsi.iloc[-1]
        
        # MACD
        macd, signal, histogram = self.calculate_macd()
        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]
        current_histogram = histogram.iloc[-1]
        
        # Bollinger Bands
        upper, middle, lower = self.calculate_bollinger_bands()
        current_price = self.df['Close'].iloc[-1]
        bb_position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
        
        # Moving Averages
        ma_short, ma_long = self.calculate_moving_averages()
        current_ma_short = ma_short.iloc[-1]
        current_ma_long = ma_long.iloc[-1]
        
        return {
            'rsi': current_rsi,
            'macd': current_macd,
            'macd_signal': current_signal,
            'macd_histogram': current_histogram,
            'bb_upper': upper.iloc[-1],
            'bb_middle': middle.iloc[-1],
            'bb_lower': lower.iloc[-1],
            'bb_position': bb_position,  # 0-1 scale, position within bands
            'ma_short': current_ma_short,
            'ma_long': current_ma_long,
            'current_price': current_price,
        }
    
    def analyze_signals(self) -> Dict:
        """
        Analyze all indicators and generate trading signals
        
        Returns:
            Dictionary with signal analysis and score
        """
        indicators = self.calculate_all_indicators()
        
        signals = []
        bullish_score = 0
        bearish_score = 0
        
        # RSI Analysis
        if indicators['rsi'] < config.RSI_OVERSOLD:
            signals.append("RSI oversold - bullish signal")
            bullish_score += 2
        elif indicators['rsi'] > config.RSI_OVERBOUGHT:
            signals.append("RSI overbought - bearish signal")
            bearish_score += 2
        elif indicators['rsi'] < 50:
            signals.append("RSI below 50 - weak bearish")
            bearish_score += 0.5
        else:
            signals.append("RSI above 50 - weak bullish")
            bullish_score += 0.5
        
        # MACD Analysis
        if indicators['macd'] > indicators['macd_signal']:
            if indicators['macd_histogram'] > 0:
                signals.append("MACD bullish crossover")
                bullish_score += 2
            else:
                signals.append("MACD above signal - bullish")
                bullish_score += 1
        else:
            if indicators['macd_histogram'] < 0:
                signals.append("MACD bearish crossover")
                bearish_score += 2
            else:
                signals.append("MACD below signal - bearish")
                bearish_score += 1
        
        # Bollinger Bands Analysis
        if indicators['bb_position'] < 0.2:
            signals.append("Price near lower Bollinger Band - bullish")
            bullish_score += 1.5
        elif indicators['bb_position'] > 0.8:
            signals.append("Price near upper Bollinger Band - bearish")
            bearish_score += 1.5
        
        # Moving Average Analysis
        if indicators['current_price'] > indicators['ma_short'] > indicators['ma_long']:
            signals.append("Golden cross pattern - strong bullish")
            bullish_score += 2
        elif indicators['current_price'] < indicators['ma_short'] < indicators['ma_long']:
            signals.append("Death cross pattern - strong bearish")
            bearish_score += 2
        elif indicators['current_price'] > indicators['ma_short']:
            signals.append("Price above short MA - bullish")
            bullish_score += 1
        else:
            signals.append("Price below short MA - bearish")
            bearish_score += 1
        
        # Calculate overall score (-1 to 1)
        total_score = bullish_score + bearish_score
        if total_score > 0:
            normalized_score = (bullish_score - bearish_score) / total_score
        else:
            normalized_score = 0
        
        return {
            'indicators': indicators,
            'signals': signals,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'technical_score': normalized_score,  # -1 (bearish) to 1 (bullish)
            'direction': 'BULLISH' if normalized_score > 0 else 'BEARISH',
            'strength': abs(normalized_score),
        }


if __name__ == "__main__":
    # Test the technical analyzer
    from data_collector import DataCollector
    
    collector = DataCollector("AAPL")
    price_data = collector.get_price_data()
    
    analyzer = TechnicalAnalyzer(price_data)
    analysis = analyzer.analyze_signals()
    
    print("Technical Analysis Results:")
    print(f"Direction: {analysis['direction']}")
    print(f"Technical Score: {analysis['technical_score']:.2f}")
    print(f"Bullish Score: {analysis['bullish_score']}")
    print(f"Bearish Score: {analysis['bearish_score']}")
    print("\nSignals:")
    for signal in analysis['signals']:
        print(f"  - {signal}")
