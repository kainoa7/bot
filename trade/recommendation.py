"""
Recommendation engine that combines all analysis to provide trade suggestions
"""
from typing import Dict, List
from datetime import datetime, timedelta
import pytz
import config
from options_calculator import OptionsCalculator


class RecommendationEngine:
    """Generates trading recommendations based on all available data"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.technical_analysis = None
        self.sentiment_analysis = None
        self.fundamentals = None
        self.options_data = None
        self.earnings_date = None
        self.current_price = 0
        
    def add_technical_analysis(self, analysis: Dict):
        """Add technical analysis results"""
        self.technical_analysis = analysis
        self.current_price = analysis['indicators']['current_price']
        
    def add_sentiment_analysis(self, analysis: Dict):
        """Add sentiment analysis results"""
        self.sentiment_analysis = analysis
        
    def add_fundamentals(self, fundamentals: Dict):
        """Add fundamental data"""
        self.fundamentals = fundamentals
        
    def add_options_data(self, options_data: Dict):
        """Add options market data"""
        self.options_data = options_data
        
    def add_earnings_date(self, earnings_date):
        """Add earnings date"""
        self.earnings_date = earnings_date
        
    def calculate_fundamental_score(self) -> Dict:
        """
        Calculate fundamental score
        
        Returns:
            Dictionary with fundamental score and signals
        """
        if not self.fundamentals:
            return {'score': 0, 'signals': ['No fundamental data available']}
        
        signals = []
        score = 0
        
        # P/E Ratio analysis
        pe = self.fundamentals.get('pe_ratio', 0)
        if 0 < pe < 15:
            signals.append("Low P/E ratio - undervalued")
            score += 1
        elif pe > 30:
            signals.append("High P/E ratio - potentially overvalued")
            score -= 1
        
        # Growth analysis
        revenue_growth = self.fundamentals.get('revenue_growth', 0)
        earnings_growth = self.fundamentals.get('earnings_growth', 0)
        
        if revenue_growth > 0.15:
            signals.append("Strong revenue growth")
            score += 1
        elif revenue_growth < 0:
            signals.append("Declining revenue")
            score -= 1
            
        if earnings_growth > 0.15:
            signals.append("Strong earnings growth")
            score += 1
        elif earnings_growth < 0:
            signals.append("Declining earnings")
            score -= 1
        
        # Profitability
        profit_margin = self.fundamentals.get('profit_margin', 0)
        if profit_margin > 0.2:
            signals.append("Strong profit margins")
            score += 0.5
        elif profit_margin < 0:
            signals.append("Negative profit margins")
            score -= 1
        
        # Analyst recommendation
        recommendation = self.fundamentals.get('recommendation', 'none')
        if recommendation in ['strong_buy', 'buy']:
            signals.append(f"Analyst recommendation: {recommendation}")
            score += 1
        elif recommendation in ['sell', 'strong_sell']:
            signals.append(f"Analyst recommendation: {recommendation}")
            score -= 1
        
        # Normalize score to -1 to 1
        normalized_score = max(-1, min(1, score / 5))
        
        return {
            'score': normalized_score,
            'signals': signals,
        }
    
    def calculate_options_score(self) -> Dict:
        """
        Calculate options market score
        
        Returns:
            Dictionary with options score and signals
        """
        if not self.options_data or not self.options_data.get('available'):
            return {'score': 0, 'signals': ['No options data available']}
        
        signals = []
        score = 0
        
        # Put/Call Ratio analysis
        pc_ratio = self.options_data.get('put_call_ratio', 0)
        if pc_ratio > 1.0:
            signals.append(f"High put/call ratio ({pc_ratio:.2f}) - bearish sentiment")
            score -= 1
        elif pc_ratio < 0.7:
            signals.append(f"Low put/call ratio ({pc_ratio:.2f}) - bullish sentiment")
            score += 1
        
        # Implied Volatility analysis
        iv = self.options_data.get('implied_volatility', 0)
        if iv > 40:
            signals.append(f"High implied volatility ({iv:.1f}%) - increased risk/reward")
            # High IV doesn't directly indicate direction
        elif iv < 20:
            signals.append(f"Low implied volatility ({iv:.1f}%) - decreased premiums")
        
        # Normalize score
        normalized_score = max(-1, min(1, score))
        
        return {
            'score': normalized_score,
            'signals': signals,
            'iv': iv,
        }
    
    def generate_recommendation(self) -> Dict:
        """
        Generate final trading recommendation
        
        Returns:
            Complete recommendation with all details
        """
        # Calculate component scores
        technical_score = self.technical_analysis['technical_score'] if self.technical_analysis else 0
        sentiment_score = self.sentiment_analysis['score'] if self.sentiment_analysis else 0
        fundamental = self.calculate_fundamental_score()
        options_analysis = self.calculate_options_score()
        
        # Weighted average
        final_score = (
            technical_score * config.WEIGHT_TECHNICAL +
            sentiment_score * config.WEIGHT_SENTIMENT +
            fundamental['score'] * config.WEIGHT_FUNDAMENTAL +
            options_analysis['score'] * config.WEIGHT_OPTIONS
        )
        
        # Determine direction
        if final_score > 0.1:
            direction = 'CALL'
            direction_label = 'BULLISH'
        elif final_score < -0.1:
            direction = 'PUT'
            direction_label = 'BEARISH'
        else:
            direction = 'NEUTRAL'
            direction_label = 'NEUTRAL'
        
        # Calculate confidence
        confidence = abs(final_score)
        if confidence >= config.CONFIDENCE_HIGH:
            confidence_label = 'HIGH'
        elif confidence >= config.CONFIDENCE_MEDIUM:
            confidence_label = 'MEDIUM'
        else:
            confidence_label = 'LOW'
        
        # Generate expiration recommendation
        expiration_recommendation = self._recommend_expiration()
        
        # Generate strike price recommendation
        strike_recommendation = self._recommend_strike(direction)
        
        # Compile all signals
        all_signals = []
        if self.technical_analysis:
            all_signals.extend(self.technical_analysis['signals'])
        if self.sentiment_analysis:
            all_signals.append(f"Sentiment: {self.sentiment_analysis['reason']}")
        all_signals.extend(fundamental['signals'])
        all_signals.extend(options_analysis['signals'])
        
        # Generate risk factors
        risk_factors = self._identify_risks(options_analysis)
        
        return {
            'ticker': self.ticker,
            'recommendation': direction,
            'direction_label': direction_label,
            'confidence': confidence,
            'confidence_label': confidence_label,
            'current_price': self.current_price,
            'strike_prices': strike_recommendation,
            'expiration': expiration_recommendation,
            'component_scores': {
                'technical': technical_score,
                'sentiment': sentiment_score,
                'fundamental': fundamental['score'],
                'options': options_analysis['score'],
                'final': final_score,
            },
            'signals': all_signals,
            'risk_factors': risk_factors,
            'implied_volatility': options_analysis.get('iv', 0),
        }
    
    def _recommend_expiration(self) -> Dict:
        """Recommend expiration dates based on analysis"""
        min_dte = config.DEFAULT_DTE_MIN
        max_dte = config.DEFAULT_DTE_MAX
        
        # Get current time in US/Pacific (market timezone)
        pst = pytz.timezone('US/Pacific')
        now = datetime.now(pst)
        
        # Adjust based on earnings date
        if self.earnings_date:
            # Make earnings_date timezone-aware if it isn't already
            if self.earnings_date.tzinfo is None:
                earnings_dt = pst.localize(self.earnings_date)
            else:
                earnings_dt = self.earnings_date.astimezone(pst)
            
            # Calculate days to earnings (accounting for time of day)
            days_to_earnings = (earnings_dt.date() - now.date()).days
            
            if 0 < days_to_earnings < 45:
                return {
                    'recommended': 'Before earnings',
                    'min_date': now + timedelta(days=7),
                    'max_date': earnings_dt - timedelta(days=1),
                    'reason': f'Earnings in {days_to_earnings} days - consider before earnings for volatility play',
                }
        
        # Standard recommendation
        min_date = now + timedelta(days=min_dte)
        max_date = now + timedelta(days=max_dte)
        
        return {
            'recommended': f'{min_dte}-{max_dte} days',
            'min_date': min_date,
            'max_date': max_date,
            'reason': 'Standard medium-term expiration for directional trades',
        }
    
    def _recommend_strike(self, direction: str) -> Dict:
        """Recommend strike prices based on direction and volatility"""
        if direction == 'NEUTRAL':
            return {
                'recommended': 'At-the-money',
                'strikes': [self.current_price],
                'labels': ['ATM'],
                'greeks': [],
            }
        
        # Calculate target based on technical levels
        if self.technical_analysis:
            indicators = self.technical_analysis['indicators']
            
            if direction == 'CALL':
                # Bullish strikes
                targets = [
                    self.current_price * 1.02,  # 2% OTM
                    self.current_price * 1.05,  # 5% OTM
                    indicators.get('bb_upper', self.current_price * 1.08),  # Upper BB
                ]
                labels = ['Conservative (2% OTM)', 'Moderate (5% OTM)', 'Aggressive (Upper BB)']
            else:
                # Bearish strikes
                targets = [
                    self.current_price * 0.98,  # 2% OTM
                    self.current_price * 0.95,  # 5% OTM
                    indicators.get('bb_lower', self.current_price * 0.92),  # Lower BB
                ]
                labels = ['Conservative (2% OTM)', 'Moderate (5% OTM)', 'Aggressive (Lower BB)']
            
            # Calculate Greeks for each strike
            greeks_list = []
            iv = self.options_data.get('implied_volatility', 30) / 100 if self.options_data else 0.30
            dte = config.DEFAULT_DTE_MIN / 365  # Use minimum DTE for calculations
            
            for strike in targets:
                try:
                    calc = OptionsCalculator(
                        stock_price=self.current_price,
                        strike=strike,
                        time_to_expiry=dte,
                        volatility=iv
                    )
                    
                    option_type = 'call' if direction == 'CALL' else 'put'
                    greeks = calc.get_all_greeks(option_type)
                    greeks['prob_profit'] = calc.probability_profit(option_type, premium=greeks['theoretical_price'])
                    greeks['breakeven'] = calc.breakeven_price(option_type, premium=greeks['theoretical_price'])
                    
                    greeks_list.append(greeks)
                except:
                    greeks_list.append({})
            
            return {
                'recommended': 'Slightly out-of-the-money',
                'strikes': targets,
                'labels': labels,
                'greeks': greeks_list,
            }
        
        return {
            'recommended': 'At-the-money',
            'strikes': [self.current_price],
            'labels': ['ATM'],
            'greeks': [],
        }
    
    def _identify_risks(self, options_analysis: Dict) -> List[str]:
        """Identify key risk factors"""
        risks = []
        
        # Get current time in US/Pacific (market timezone)
        pst = pytz.timezone('US/Pacific')
        now = datetime.now(pst)
        
        # Earnings risk
        if self.earnings_date:
            # Make earnings_date timezone-aware if it isn't already
            if self.earnings_date.tzinfo is None:
                earnings_dt = pst.localize(self.earnings_date)
            else:
                earnings_dt = self.earnings_date.astimezone(pst)
            
            days_to_earnings = (earnings_dt.date() - now.date()).days
            if 0 < days_to_earnings < 14:
                risks.append(f"⚠️ Earnings in {days_to_earnings} days - high volatility risk")
        
        # High IV risk
        iv = options_analysis.get('iv', 0)
        if iv > 50:
            risks.append(f"⚠️ High implied volatility ({iv:.1f}%) - expensive premiums")
        
        # Low confidence risk
        if self.technical_analysis and self.sentiment_analysis:
            tech_direction = self.technical_analysis['direction']
            sent_signal = self.sentiment_analysis['signal']
            if tech_direction != sent_signal and sent_signal != 'NEUTRAL':
                risks.append("⚠️ Technical and sentiment analysis diverge - mixed signals")
        
        # Fundamental concerns
        if self.fundamentals:
            if self.fundamentals.get('debt_to_equity', 0) > 2:
                risks.append("⚠️ High debt-to-equity ratio")
            if self.fundamentals.get('earnings_growth', 0) < -0.1:
                risks.append("⚠️ Declining earnings")
        
        if not risks:
            risks.append("✓ No major risk factors identified")
        
        return risks


if __name__ == "__main__":
    # Test the recommendation engine
    from data_collector import DataCollector
    from indicators import TechnicalAnalyzer
    from sentiment_analyzer import SentimentAnalyzer
    
    ticker = "AAPL"
    
    # Collect data
    collector = DataCollector(ticker)
    price_data = collector.get_price_data()
    fundamentals = collector.get_fundamentals()
    news = collector.get_news()
    options_data = collector.get_options_data()
    earnings_date = collector.get_earnings_date()
    
    # Analyze
    tech_analyzer = TechnicalAnalyzer(price_data)
    tech_analysis = tech_analyzer.analyze_signals()
    
    sent_analyzer = SentimentAnalyzer(news)
    sent_analysis = sent_analyzer.get_sentiment_signal()
    
    # Generate recommendation
    engine = RecommendationEngine(ticker)
    engine.add_technical_analysis(tech_analysis)
    engine.add_sentiment_analysis(sent_analysis)
    engine.add_fundamentals(fundamentals)
    engine.add_options_data(options_data)
    engine.add_earnings_date(earnings_date)
    
    recommendation = engine.generate_recommendation()
    
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION FOR {ticker}")
    print(f"{'='*60}")
    print(f"Direction: {recommendation['recommendation']} ({recommendation['direction_label']})")
    print(f"Confidence: {recommendation['confidence_label']} ({recommendation['confidence']:.2f})")
    print(f"Current Price: ${recommendation['current_price']:.2f}")
