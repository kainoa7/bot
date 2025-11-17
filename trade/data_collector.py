"""
Data collection module for fetching stock prices, fundamentals, and news
"""
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import time
import config


class DataCollector:
    """Collects market data from various sources"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        
    def get_price_data(self, period: str = "6mo") -> pd.DataFrame:
        """
        Fetch historical price data with retry logic
        
        Args:
            period: Time period (1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            
        Returns:
            DataFrame with OHLCV data
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Recreate ticker object to avoid cached connection issues
                stock = yf.Ticker(self.ticker)
                df = stock.history(period=period)
                
                if df.empty:
                    if attempt < max_retries - 1:
                        print(f"⚠️  No data on attempt {attempt + 1}, retrying in 2 seconds...")
                        time.sleep(2)
                        continue
                    raise ValueError(f"No data found for {self.ticker}. Check ticker symbol or try again later.")
                
                return df
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  Attempt {attempt + 1} failed: {str(e)}, retrying...")
                    time.sleep(2)
                    continue
                raise Exception(f"Error fetching price data after {max_retries} attempts: {str(e)}")
    
    def get_fundamentals(self) -> Dict:
        """
        Fetch fundamental data
        
        Returns:
            Dictionary with key fundamental metrics
        """
        try:
            info = self.stock.info
            
            fundamentals = {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'profit_margin': info.get('profitMargins', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_price': info.get('currentPrice', 0),
                'target_mean_price': info.get('targetMeanPrice', 0),
                'recommendation': info.get('recommendationKey', 'none'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
            }
            
            return fundamentals
        except Exception as e:
            print(f"Warning: Could not fetch all fundamentals: {str(e)}")
            return {}
    
    def get_earnings_date(self) -> Optional[datetime]:
        """
        Get next earnings date
        
        Returns:
            Next earnings date or None if not available
        """
        try:
            calendar = self.stock.calendar
            if calendar is not None and 'Earnings Date' in calendar:
                earnings_date = calendar['Earnings Date']
                if isinstance(earnings_date, pd.Timestamp):
                    return earnings_date
                elif len(earnings_date) > 0:
                    return pd.Timestamp(earnings_date[0])
            return None
        except:
            return None
    
    def get_options_data(self) -> Dict:
        """
        Fetch options chain data
        
        Returns:
            Dictionary with options metrics
        """
        try:
            # Get available expiration dates
            expirations = self.stock.options
            
            if not expirations:
                return {'available': False}
            
            # Get current stock price
            current_price = self.stock.info.get('currentPrice', 0)
            
            # Get options data for nearest expiration
            nearest_exp = expirations[0]
            opt_chain = self.stock.option_chain(nearest_exp)
            
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Calculate put/call ratio
            call_volume = calls['volume'].sum()
            put_volume = puts['volume'].sum()
            pc_ratio = put_volume / call_volume if call_volume > 0 else 0
            
            # Get implied volatility
            atm_calls = calls[calls['strike'].between(current_price * 0.95, current_price * 1.05)]
            atm_puts = puts[puts['strike'].between(current_price * 0.95, current_price * 1.05)]
            
            avg_call_iv = atm_calls['impliedVolatility'].mean() if not atm_calls.empty else 0
            avg_put_iv = atm_puts['impliedVolatility'].mean() if not atm_puts.empty else 0
            avg_iv = (avg_call_iv + avg_put_iv) / 2
            
            return {
                'available': True,
                'expirations': expirations,
                'put_call_ratio': pc_ratio,
                'implied_volatility': avg_iv * 100,  # Convert to percentage
                'call_volume': call_volume,
                'put_volume': put_volume,
                'calls_chain': calls,
                'puts_chain': puts,
                'nearest_expiration': nearest_exp,
            }
        except Exception as e:
            print(f"Warning: Could not fetch options data: {str(e)}")
            return {'available': False}
    
    def get_specific_option_price(self, strike: float, expiration: str, 
                                   option_type: str = 'call') -> Dict:
        """
        Get specific option contract details
        
        Args:
            strike: Strike price
            expiration: Expiration date string
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with option details
        """
        try:
            opt_chain = self.stock.option_chain(expiration)
            chain = opt_chain.calls if option_type == 'call' else opt_chain.puts
            
            # Find closest strike
            option = chain.iloc[(chain['strike'] - strike).abs().argsort()[:1]]
            
            if option.empty:
                return {'available': False}
            
            row = option.iloc[0]
            return {
                'available': True,
                'strike': row['strike'],
                'bid': row.get('bid', 0),
                'ask': row.get('ask', 0),
                'last_price': row.get('lastPrice', 0),
                'volume': row.get('volume', 0),
                'open_interest': row.get('openInterest', 0),
                'implied_volatility': row.get('impliedVolatility', 0) * 100,
            }
        except Exception as e:
            print(f"Warning: Could not fetch option price: {str(e)}")
            return {'available': False}
    
    def get_news(self, lookback_days: int = 7) -> List[Dict]:
        """
        Fetch recent news articles
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            List of news articles with title, description, and URL
        """
        news = []
        
        # Try NewsAPI first
        if config.NEWS_API_KEY and config.NEWS_API_KEY != 'your_news_api_key_here':
            try:
                from_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
                url = f"https://newsapi.org/v2/everything"
                params = {
                    'q': self.ticker,
                    'from': from_date,
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'apiKey': config.NEWS_API_KEY,
                }
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    articles = response.json().get('articles', [])
                    for article in articles[:10]:  # Limit to 10 articles
                        news.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt', ''),
                        })
            except Exception as e:
                print(f"Warning: NewsAPI request failed: {str(e)}")
        
        # Fallback to yfinance news
        if not news:
            try:
                yf_news = self.stock.news
                for article in yf_news[:10]:
                    news.append({
                        'title': article.get('title', ''),
                        'description': article.get('summary', ''),
                        'url': article.get('link', ''),
                        'published_at': datetime.fromtimestamp(article.get('providerPublishTime', 0)).isoformat(),
                    })
            except Exception as e:
                print(f"Warning: Could not fetch news: {str(e)}")
        
        return news
    
    def get_historical_earnings_data(self) -> Dict:
        """
        Get historical earnings data and move analysis
        
        Returns:
            Dictionary with earnings history and statistics
        """
        try:
            # Get earnings history - yfinance may not have this for all stocks
            try:
                earnings = self.stock.earnings_history
            except AttributeError:
                # earnings_history might not be available
                earnings = None
            
            if earnings is None or (hasattr(earnings, '__len__') and len(earnings) == 0):
                return {
                    'available': False,
                    'average_move': 0,
                    'beat_count': 0,
                    'miss_count': 0,
                    'meet_count': 0,
                    'total_quarters': 0,
                    'beat_rate': 0,
                }
            
            # Check if it's a DataFrame
            if not isinstance(earnings, pd.DataFrame):
                return {
                    'available': False,
                    'average_move': 0,
                    'beat_count': 0,
                    'miss_count': 0,
                    'meet_count': 0,
                    'total_quarters': 0,
                    'beat_rate': 0,
                }
            
            moves = []
            beats = 0
            misses = 0
            meets = 0
            
            for _, row in earnings.iterrows():
                # Calculate move percentage if available
                if 'epsActual' in row and 'epsEstimate' in row:
                    if pd.notna(row['epsActual']) and pd.notna(row['epsEstimate']):
                        if row['epsEstimate'] != 0:
                            move_pct = ((row['epsActual'] - row['epsEstimate']) / abs(row['epsEstimate'])) * 100
                            moves.append(abs(move_pct))
                            
                            if row['epsActual'] > row['epsEstimate']:
                                beats += 1
                            elif row['epsActual'] < row['epsEstimate']:
                                misses += 1
                            else:
                                meets += 1
            
            avg_move = sum(moves) / len(moves) if moves else 0
            
            return {
                'available': True,
                'average_move': avg_move,
                'beat_count': beats,
                'miss_count': misses,
                'meet_count': meets,
                'total_quarters': len(earnings),
                'beat_rate': (beats / len(earnings) * 100) if len(earnings) > 0 else 0,
            }
        except Exception as e:
            print(f"Warning: Could not fetch historical earnings data: {str(e)}")
            return {
                'available': False,
                'average_move': 0,
                'beat_count': 0,
                'miss_count': 0,
                'meet_count': 0,
                'total_quarters': 0,
                'beat_rate': 0,
            }
    
    def get_price_performance_metrics(self, periods: List[int] = [7, 14, 30]) -> Dict:
        """
        Calculate price performance metrics for different periods
        
        Args:
            periods: List of days to calculate performance for
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            price_data = self.get_price_data(period='1y')
            if price_data.empty:
                return {}
            
            current_price = price_data['Close'].iloc[-1]
            metrics = {}
            
            for days in periods:
                if len(price_data) > days:
                    past_price = price_data['Close'].iloc[-days-1]
                    change_pct = ((current_price - past_price) / past_price) * 100
                    metrics[f'{days}d_change'] = change_pct
                    metrics[f'{days}d_price'] = past_price
            
            # Calculate distance from moving averages
            ma_20 = price_data['Close'].rolling(20).mean().iloc[-1] if len(price_data) >= 20 else None
            ma_50 = price_data['Close'].rolling(50).mean().iloc[-1] if len(price_data) >= 50 else None
            ma_200 = price_data['Close'].rolling(200).mean().iloc[-1] if len(price_data) >= 200 else None
            
            metrics['current_price'] = current_price
            if ma_20:
                metrics['distance_from_ma20'] = ((current_price - ma_20) / ma_20) * 100
            if ma_50:
                metrics['distance_from_ma50'] = ((current_price - ma_50) / ma_50) * 100
            if ma_200:
                metrics['distance_from_ma200'] = ((current_price - ma_200) / ma_200) * 100
            
            # Calculate volatility trend
            if len(price_data) >= 30:
                recent_vol = price_data['Close'].pct_change().tail(14).std() * 100
                older_vol = price_data['Close'].pct_change().tail(30).head(16).std() * 100
                metrics['volatility_trend'] = 'INCREASING' if recent_vol > older_vol else 'DECREASING'
                metrics['recent_volatility'] = recent_vol
                metrics['older_volatility'] = older_vol
            
            # Calculate support/resistance levels
            if len(price_data) >= 30:
                recent_high = price_data['High'].tail(30).max()
                recent_low = price_data['Low'].tail(30).min()
                metrics['resistance_level'] = recent_high
                metrics['support_level'] = recent_low
                metrics['distance_to_resistance'] = ((recent_high - current_price) / current_price) * 100
                metrics['distance_to_support'] = ((current_price - recent_low) / current_price) * 100
            
            return metrics
        except Exception as e:
            print(f"Warning: Could not calculate price performance metrics: {str(e)}")
            return {}


if __name__ == "__main__":
    # Test the data collector
    collector = DataCollector("AAPL")
    
    print("Fetching price data...")
    prices = collector.get_price_data()
    print(f"Got {len(prices)} days of price data")
    print(prices.tail())
    
    print("\nFetching fundamentals...")
    fundamentals = collector.get_fundamentals()
    print(fundamentals)
    
    print("\nFetching news...")
    news = collector.get_news()
    print(f"Found {len(news)} articles")
    for article in news[:3]:
        print(f"- {article['title']}")
