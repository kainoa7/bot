"""
Position Analysis Module
Analyzes existing options positions and provides actionable recommendations
"""

import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
from data_collector import DataCollector
from indicators import TechnicalAnalyzer
from sentiment_analyzer import SentimentAnalyzer
import pandas as pd
import pytz

def analyze_existing_position(ticker, position_type, strike_price, premium_paid, 
                              contracts, entry_date, expiration_date):
    """
    Analyze an existing options position and provide recommendations
    
    Args:
        ticker: Stock symbol
        position_type: 'call' or 'put'
        strike_price: Strike price of the option
        premium_paid: Premium paid per contract
        contracts: Number of contracts
        entry_date: Date position was entered (YYYY-MM-DD)
        expiration_date: Option expiration date (YYYY-MM-DD)
    
    Returns:
        dict: Comprehensive position analysis
    """
    
    try:
        # Get current data
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        
        # Parse dates
        entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
        expiration_dt = datetime.strptime(expiration_date, '%Y-%m-%d')
        
        # Get current time in PST timezone
        pst = pytz.timezone('America/Los_Angeles')
        today_utc = datetime.now(pytz.UTC)
        today_pst = today_utc.astimezone(pst)
        today = today_pst.replace(tzinfo=None)  # Remove timezone for comparison
        
        # Calculate time metrics
        days_held = (today - entry_dt).days
        days_to_expiration = (expiration_dt - today).days
        total_days = (expiration_dt - entry_dt).days
        
        # Calculate P&L
        breakeven = strike_price + premium_paid if position_type == 'call' else strike_price - premium_paid
        
        # Current option value (simplified Black-Scholes approximation)
        current_value = estimate_option_value(
            current_price, strike_price, days_to_expiration, 
            position_type, info.get('impliedVolatility', 0.3)
        )
        
        cost_basis = premium_paid * contracts * 100
        current_position_value = current_value * contracts * 100
        pnl = current_position_value - cost_basis
        pnl_percent = (pnl / cost_basis * 100) if cost_basis > 0 else 0
        
        # Get historical data
        collector = DataCollector(ticker)
        price_data = collector.get_price_data(period='1y')
        
        # Calculate Greeks
        greeks = calculate_greeks(
            current_price, strike_price, days_to_expiration,
            info.get('impliedVolatility', 0.3), position_type
        )
        
        # Get sentiment and news with enhanced analysis
        news = collector.get_news()
        sent_analyzer = SentimentAnalyzer(news)
        sentiment_data = sent_analyzer.get_sentiment_signal()
        news_momentum = sent_analyzer.get_news_momentum()
        breaking_news = sent_analyzer.get_breaking_news()
        critical_news = sent_analyzer.get_critical_news()
        
        # Get enhanced news analysis
        enhanced_news = {
            'top_articles': sentiment_data.get('headlines', [])[:7],
            'momentum': news_momentum,
            'breaking_news': breaking_news,
            'critical_news': critical_news,
            'overall_sentiment': sentiment_data.get('overall_sentiment', 0),
            'news_count': sentiment_data.get('news_count', 0),
        }
        
        # Get earnings date and enhanced earnings intelligence
        earnings_date = get_next_earnings_date(stock)
        days_to_earnings = None
        if earnings_date:
            # Parse earnings date - assume earnings happens at start of that day (midnight)
            earnings_dt = datetime.strptime(earnings_date, '%Y-%m-%d')
            
            # Calculate time difference in hours
            time_until_earnings = earnings_dt - today
            total_hours = time_until_earnings.total_seconds() / 3600
            
            # Convert to days, rounding up if there are any hours remaining
            # This ensures that if it's Sunday evening and earnings is Wednesday,
            # we show 3 days (not 2) since there are still ~2.5 days remaining
            if total_hours <= 0:
                days_to_earnings = 0
            else:
                # Round up to nearest day if there are any hours remaining
                days_to_earnings = int(total_hours / 24) + (1 if (total_hours % 24) > 0 else 0)
        
        # Get historical earnings data
        earnings_history = collector.get_historical_earnings_data()
        
        # Enhanced earnings intelligence
        earnings_intel = get_earnings_intelligence(
            earnings_date, days_to_earnings, earnings_history,
            info.get('impliedVolatility', 0.3), current_price
        )
        
        # Get price performance metrics
        price_metrics = collector.get_price_performance_metrics([7, 14, 30])
        
        # Enhanced price trend analysis
        price_trends = analyze_price_trends(
            price_data, current_price, strike_price, position_type
        )
        
        # Enhanced technical indicators
        tech_analyzer = TechnicalAnalyzer(price_data)
        tech_analysis = tech_analyzer.analyze_signals()
        enhanced_technicals = get_enhanced_technicals(
            tech_analysis, price_data, current_price
        )
        
        # Calculate probability of profit
        prob_profit = calculate_profit_probability(
            current_price, breakeven, days_to_expiration,
            info.get('impliedVolatility', 0.3), position_type
        )
        
        # Generate enhanced recommendation with all new data
        recommendation = generate_enhanced_recommendation(
            pnl_percent, days_to_expiration, prob_profit, greeks,
            sentiment_data, days_to_earnings, position_type,
            price_trends, earnings_intel, enhanced_technicals, news_momentum
        )
        
        # Generate enhanced action plan
        action_plan = generate_enhanced_action_plan(
            ticker, current_price, strike_price, breakeven, 
            days_to_expiration, pnl_percent, position_type,
            greeks, days_to_earnings, sentiment_data, price_trends,
            earnings_intel, enhanced_technicals
        )
        
        # Generate exit strategy
        exit_strategy = generate_exit_strategy(
            current_price, strike_price, breakeven, days_to_expiration,
            pnl_percent, position_type, greeks, earnings_intel, price_trends
        )
        
        # Generate timeline with day-by-day recommendations
        timeline = generate_position_timeline(
            entry_date, expiration_date, contracts, current_price, strike_price,
            breakeven, position_type, greeks, earnings_intel, price_trends,
            sentiment_data, enhanced_technicals, days_to_expiration, pnl_percent
        )
        
        # Alternative strategies
        alternatives = generate_alternatives(
            ticker, current_price, strike_price, expiration_date,
            position_type, premium_paid, current_value, pnl
        )
        
        return {
            'ticker': ticker,
            'position_type': position_type.upper(),
            'current_price': current_price,
            'strike_price': strike_price,
            'breakeven': breakeven,
            'premium_paid': premium_paid,
            'contracts': contracts,
            
            # Time metrics
            'days_held': days_held,
            'days_to_expiration': days_to_expiration,
            'total_days': total_days,
            'expiration_date': expiration_date,
            
            # P&L
            'cost_basis': cost_basis,
            'current_value': current_position_value,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            
            # Greeks
            'greeks': greeks,
            
            # Market data
            'current_iv': info.get('impliedVolatility', 0) * 100,
            'prob_profit': prob_profit * 100,
            
            # Earnings
            'earnings_date': earnings_date,
            'days_to_earnings': days_to_earnings,
            
            # Enhanced News Analysis
            'news_analysis': enhanced_news,
            'sentiment_score': sentiment_data.get('overall_sentiment', 0),
            'news_count': sentiment_data.get('news_count', 0),
            
            # Price Trends
            'price_trends': price_trends,
            'price_metrics': price_metrics,
            
            # Earnings Intelligence
            'earnings_intelligence': earnings_intel,
            
            # Enhanced Technicals
            'technical_analysis': enhanced_technicals,
            
            # Recommendation
            'recommendation': recommendation,
            'action_plan': action_plan,
            'exit_strategy': exit_strategy,
            'timeline': timeline,
            'alternatives': alternatives,
            
            # Status
            'position_status': get_position_status(pnl_percent, days_to_expiration, prob_profit)
        }
        
    except Exception as e:
        raise Exception(f"Error analyzing position: {str(e)}")


def estimate_option_value(stock_price, strike, days_to_exp, option_type, iv):
    """Estimate current option value using simplified Black-Scholes"""
    if days_to_exp <= 0:
        # Expired
        if option_type == 'call':
            return max(0, stock_price - strike)
        else:
            return max(0, strike - stock_price)
    
    # Simplified intrinsic + time value
    intrinsic = 0
    if option_type == 'call':
        intrinsic = max(0, stock_price - strike)
    else:
        intrinsic = max(0, strike - stock_price)
    
    # Time value decreases as expiration approaches
    time_value = iv * np.sqrt(days_to_exp / 365) * stock_price * 0.4
    
    return intrinsic + time_value


def calculate_greeks(stock_price, strike, days_to_exp, iv, option_type):
    """Calculate option Greeks"""
    if days_to_exp <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    # Risk-free rate (approximate)
    r = 0.05
    
    # Time to expiration in years
    T = days_to_exp / 365.0
    
    # Calculate d1 and d2
    d1 = (np.log(stock_price / strike) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    
    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)
    
    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (stock_price * iv * np.sqrt(T))
    
    # Theta (daily decay)
    if option_type == 'call':
        theta = (-stock_price * norm.pdf(d1) * iv / (2 * np.sqrt(T)) 
                 - r * strike * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-stock_price * norm.pdf(d1) * iv / (2 * np.sqrt(T)) 
                 + r * strike * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    # Vega (per 1% change in IV)
    vega = stock_price * norm.pdf(d1) * np.sqrt(T) / 100
    
    return {
        'delta': round(delta, 3),
        'gamma': round(gamma, 4),
        'theta': round(theta, 2),
        'vega': round(vega, 2)
    }


def calculate_profit_probability(current_price, breakeven, days_to_exp, iv, option_type):
    """Calculate probability of profit at expiration"""
    if days_to_exp <= 0:
        if option_type == 'call':
            return 1.0 if current_price > breakeven else 0.0
        else:
            return 1.0 if current_price < breakeven else 0.0
    
    # Standard deviation of price movement
    T = days_to_exp / 365.0
    std_dev = current_price * iv * np.sqrt(T)
    
    # Z-score
    if option_type == 'call':
        z = (breakeven - current_price) / std_dev
        prob = 1 - norm.cdf(z)
    else:
        z = (current_price - breakeven) / std_dev
        prob = norm.cdf(z)
    
    return max(0, min(1, prob))


def get_next_earnings_date(stock):
    """Get next earnings date"""
    try:
        calendar = stock.calendar
        if calendar is not None and 'Earnings Date' in calendar:
            earnings = calendar['Earnings Date']
            if isinstance(earnings, list) and len(earnings) > 0:
                earnings_dt = earnings[0]
            else:
                earnings_dt = earnings
            
            if isinstance(earnings_dt, str):
                return earnings_dt
            else:
                return earnings_dt.strftime('%Y-%m-%d')
    except:
        pass
    return None


def get_position_status(pnl_percent, days_to_exp, prob_profit):
    """Determine overall position health status"""
    if pnl_percent > 20 and prob_profit > 0.6:
        return 'healthy'
    elif pnl_percent > 0 and prob_profit > 0.4:
        return 'moderate'
    elif days_to_exp < 7:
        return 'urgent'
    elif pnl_percent < -30:
        return 'critical'
    else:
        return 'watch'


def generate_position_recommendation(pnl_percent, days_to_exp, prob_profit, 
                                     greeks, sentiment, days_to_earnings, position_type):
    """Generate primary recommendation"""
    
    # Strong profit - consider taking
    if pnl_percent > 50:
        return {
            'action': 'TAKE PROFIT',
            'confidence': 85,
            'reason': f'Strong profit of {pnl_percent:.1f}%. Lock in gains before time decay accelerates.'
        }
    
    # Near expiration with profit
    if days_to_exp <= 7 and pnl_percent > 10:
        return {
            'action': 'CLOSE POSITION',
            'confidence': 80,
            'reason': f'Only {days_to_exp} days left. Take {pnl_percent:.1f}% profit before expiration.'
        }
    
    # Near expiration with loss
    if days_to_exp <= 5 and pnl_percent < -20:
        return {
            'action': 'CUT LOSSES',
            'confidence': 75,
            'reason': f'Time running out ({days_to_exp} days). Minimize loss before worthless expiration.'
        }
    
    # Earnings approaching
    if days_to_earnings and 0 < days_to_earnings <= 5:
        if pnl_percent > 20:
            return {
                'action': 'CLOSE BEFORE EARNINGS',
                'confidence': 70,
                'reason': f'Earnings in {days_to_earnings} days. Secure {pnl_percent:.1f}% profit to avoid volatility risk.'
            }
        else:
            return {
                'action': 'HOLD FOR EARNINGS',
                'confidence': 60,
                'reason': f'Earnings in {days_to_earnings} days. Volatility may provide recovery opportunity.'
            }
    
    # Deep loss
    if pnl_percent < -50 and prob_profit < 0.3:
        return {
            'action': 'CONSIDER ROLLING',
            'confidence': 65,
            'reason': f'{pnl_percent:.1f}% loss with low recovery probability. Rolling may extend opportunity.'
        }
    
    # Positive momentum
    if sentiment['overall_sentiment'] > 0.3 and pnl_percent > 0 and days_to_exp > 14:
        return {
            'action': 'HOLD',
            'confidence': 70,
            'reason': f'Positive momentum and {days_to_exp} days remaining. Let position develop.'
        }
    
    # Theta decay concern
    if abs(greeks['theta']) > 5 and days_to_exp < 14:
        return {
            'action': 'WATCH CLOSELY',
            'confidence': 60,
            'reason': f'High time decay (${abs(greeks["theta"]):.2f}/day). Monitor daily for exit opportunity.'
        }
    
    # Delta analysis - low delta means low price sensitivity
    if abs(greeks['delta']) < 0.2 and days_to_exp < 21:
        return {
            'action': 'CONSIDER EXITING',
            'confidence': 65,
            'reason': f'Low delta ({greeks["delta"]:.3f}) means position is less sensitive to price moves. Time decay may outweigh potential gains.'
        }
    
    # High delta with profit - lock in gains
    if abs(greeks['delta']) > 0.7 and pnl_percent > 15:
        return {
            'action': 'CONSIDER TAKING PROFIT',
            'confidence': 75,
            'reason': f'High delta ({abs(greeks["delta"]):.3f}) position with {pnl_percent:.1f}% profit. Consider locking in gains before volatility changes.'
        }
    
    # Gamma risk - high gamma means delta changes rapidly
    if greeks['gamma'] > 0.01 and days_to_exp < 10:
        return {
            'action': 'MONITOR CLOSELY',
            'confidence': 70,
            'reason': f'High gamma ({greeks["gamma"]:.4f}) means position sensitivity changes quickly near expiration. Watch for rapid price moves.'
        }
    
    # Vega risk - high vega means IV sensitivity
    if greeks['vega'] > 0.3 and days_to_earnings and days_to_earnings <= 7:
        return {
            'action': 'EXIT BEFORE EARNINGS',
            'confidence': 75,
            'reason': f'High vega ({greeks["vega"]:.2f}) means position is very sensitive to IV changes. Exit before earnings to avoid IV crush.'
        }
    
    # Default - hold and monitor
    return {
        'action': 'HOLD & MONITOR',
        'confidence': 55,
        'reason': f'Position within normal range. {days_to_exp} days remaining. Watch for better exit point.'
    }


def generate_action_plan(ticker, current_price, strike, breakeven, days_to_exp, 
                        pnl_percent, position_type, greeks, days_to_earnings, sentiment):
    """Generate step-by-step action plan"""
    
    plan = []
    
    # Immediate actions based on earnings
    if days_to_earnings and 0 < days_to_earnings <= 3:
        plan.append({
            'priority': 'high',
            'action': f'⚠️ EARNINGS ALERT: {days_to_earnings} day(s) until earnings announcement',
            'details': 'Decide before close of day: Hold through earnings or exit to avoid volatility.'
        })
    
    # Profit targets
    if pnl_percent < 30:
        target_price = strike + (breakeven - strike) * 1.5 if position_type == 'call' else strike - (strike - breakeven) * 1.5
        plan.append({
            'priority': 'medium',
            'action': f'PROFIT TARGET: Set alert at ${target_price:.2f}',
            'details': f'If stock reaches ${target_price:.2f}, consider taking profit (~30%+ gain).'
        })
    
    # Stop loss
    if pnl_percent > -30:
        stop_price = strike * 0.95 if position_type == 'call' else strike * 1.05
        plan.append({
            'priority': 'medium',
            'action': f'STOP LOSS: Set alert at ${stop_price:.2f}',
            'details': f'Exit if stock drops to ${stop_price:.2f} to limit losses.'
        })
    
    # Time-based actions
    if days_to_exp > 14:
        plan.append({
            'priority': 'low',
            'action': f'REVIEW in 7 days (around {(datetime.now() + timedelta(days=7)).strftime("%b %d")})',
            'details': 'Re-evaluate position health and adjust strategy if needed.'
        })
    elif days_to_exp > 7:
        plan.append({
            'priority': 'medium',
            'action': f'DECISION POINT: {days_to_exp} days left',
            'details': 'Time decay accelerating. Decide this week: hold to expiration or exit early.'
        })
    else:
        plan.append({
            'priority': 'high',
            'action': f'⏰ URGENT: Only {days_to_exp} days until expiration',
            'details': 'Close position or prepare to exercise if ITM. Don\'t let expire worthless.'
        })
    
    # Sentiment-based
    if sentiment['overall_sentiment'] < -0.3:
        plan.append({
            'priority': 'medium',
            'action': '📰 NEGATIVE NEWS DETECTED',
            'details': f'Recent sentiment is bearish. Consider exiting {"calls" if position_type == "call" else "puts"} if trend continues.'
        })
    elif sentiment['overall_sentiment'] > 0.3:
        plan.append({
            'priority': 'low',
            'action': '📈 POSITIVE MOMENTUM',
            'details': f'Recent sentiment is bullish. Favorable for {"calls" if position_type == "call" else "puts"}.'
        })
    
    return plan


def generate_alternatives(ticker, current_price, strike, expiration_date, 
                         position_type, premium_paid, current_value, pnl):
    """Generate alternative strategy options"""
    
    alternatives = []
    
    # Roll out (extend expiration)
    exp_dt = datetime.strptime(expiration_date, '%Y-%m-%d')
    new_exp = exp_dt + timedelta(days=30)
    roll_cost = (premium_paid - current_value) * 0.5  # Estimated
    
    alternatives.append({
        'strategy': 'Roll Out (Extend Time)',
        'description': f'Close current position, open new {position_type.upper()} at ${strike:.2f} expiring {new_exp.strftime("%b %d")}',
        'cost': f'~${abs(roll_cost * 100):.0f} additional per contract',
        'pros': 'More time for stock to move your way, reduces time decay pressure',
        'cons': 'Additional cost, position stays in limbo longer'
    })
    
    # Roll up/down (adjust strike)
    if position_type == 'call':
        new_strike = current_price * 1.05
        alternatives.append({
            'strategy': 'Roll Up (Higher Strike)',
            'description': f'Close current ${strike:.2f} call, open ${new_strike:.2f} call (same expiration)',
            'cost': f'Collect credit, reduce max profit',
            'pros': 'Locks in some profit, reduces risk if stock reverses',
            'cons': 'Caps upside potential, may miss bigger moves'
        })
    else:
        new_strike = current_price * 0.95
        alternatives.append({
            'strategy': 'Roll Down (Lower Strike)',
            'description': f'Close current ${strike:.2f} put, open ${new_strike:.2f} put (same expiration)',
            'cost': f'Collect credit, reduce max profit',
            'pros': 'Locks in some profit, reduces risk if stock reverses',
            'cons': 'Caps downside capture, may miss bigger moves'
        })
    
    # Convert to spread
    if position_type == 'call':
        sell_strike = strike * 1.1
        alternatives.append({
            'strategy': 'Convert to Call Spread',
            'description': f'Keep ${strike:.2f} call, SELL ${sell_strike:.2f} call against it',
            'cost': f'Collect premium, reduce cost basis',
            'pros': 'Reduces risk, lowers break-even, still profit from move up',
            'cons': 'Caps maximum profit at sold strike level'
        })
    else:
        sell_strike = strike * 0.9
        alternatives.append({
            'strategy': 'Convert to Put Spread',
            'description': f'Keep ${strike:.2f} put, SELL ${sell_strike:.2f} put against it',
            'cost': f'Collect premium, reduce cost basis',
            'pros': 'Reduces risk, lowers break-even, still profit from move down',
            'cons': 'Caps maximum profit at sold strike level'
        })
    
    # Cut and redeploy
    alternatives.append({
        'strategy': 'Cut Loss & Redeploy',
        'description': f'Close position now, use remaining capital for better setup',
        'cost': f'Accept ${abs(pnl):.0f} loss',
        'pros': 'Stop the bleeding, free up capital for opportunities',
        'cons': 'Realize the loss, no chance of recovery on this trade'
    })
    
    return alternatives


def analyze_price_trends(price_data, current_price, strike_price, position_type):
    """Analyze price trends and momentum"""
    if price_data.empty or len(price_data) < 30:
        return {}
    
    # Calculate momentum for different periods
    momentum_7d = ((current_price - price_data['Close'].iloc[-8]) / price_data['Close'].iloc[-8] * 100) if len(price_data) >= 8 else 0
    momentum_14d = ((current_price - price_data['Close'].iloc[-15]) / price_data['Close'].iloc[-15] * 100) if len(price_data) >= 15 else 0
    momentum_30d = ((current_price - price_data['Close'].iloc[-31]) / price_data['Close'].iloc[-31] * 100) if len(price_data) >= 31 else 0
    
    # Calculate support/resistance
    recent_high = price_data['High'].tail(30).max()
    recent_low = price_data['Low'].tail(30).min()
    
    # Distance from strike
    distance_from_strike = ((current_price - strike_price) / strike_price) * 100
    
    return {
        'momentum_7d': momentum_7d,
        'momentum_14d': momentum_14d,
        'momentum_30d': momentum_30d,
        'support_level': recent_low,
        'resistance_level': recent_high,
        'distance_from_strike': distance_from_strike,
        'trend_direction': 'BULLISH' if momentum_7d > 0 else 'BEARISH',
        'trend_strength': abs(momentum_7d),
    }


def get_earnings_intelligence(earnings_date, days_to_earnings, earnings_history, current_iv, current_price):
    """Comprehensive earnings intelligence"""
    intel = {
        'earnings_date': earnings_date,
        'days_to_earnings': days_to_earnings,
        'urgency': 'NONE',
        'average_move': 0,
        'beat_rate': 0,
        'iv_crush_risk': 'UNKNOWN',
        'recommendation': 'HOLD',
    }
    
    if days_to_earnings is not None:
        if days_to_earnings <= 3:
            intel['urgency'] = 'CRITICAL'
        elif days_to_earnings <= 7:
            intel['urgency'] = 'HIGH'
        elif days_to_earnings <= 14:
            intel['urgency'] = 'MEDIUM'
        else:
            intel['urgency'] = 'LOW'
        
        # Historical earnings data
        if earnings_history.get('available'):
            intel['average_move'] = earnings_history.get('average_move', 0)
            intel['beat_rate'] = earnings_history.get('beat_rate', 0)
            intel['beat_count'] = earnings_history.get('beat_count', 0)
            intel['miss_count'] = earnings_history.get('miss_count', 0)
        
        # IV crush risk assessment
        if current_iv > 0.4:  # High IV
            intel['iv_crush_risk'] = 'HIGH'
            intel['recommendation'] = 'EXIT_BEFORE'
        elif current_iv > 0.25:
            intel['iv_crush_risk'] = 'MEDIUM'
            intel['recommendation'] = 'CONSIDER_EXIT'
        else:
            intel['iv_crush_risk'] = 'LOW'
            intel['recommendation'] = 'HOLD_THROUGH'
    
    return intel


def get_enhanced_technicals(tech_analysis, price_data, current_price):
    """Get enhanced technical indicator analysis"""
    indicators = tech_analysis.get('indicators', {})
    
    # RSI trend
    rsi = indicators.get('rsi', 50)
    rsi_trend = 'RISING' if rsi > 50 else 'FALLING'
    
    # MACD momentum strength
    macd_hist = indicators.get('macd_histogram', 0)
    macd_strength = 'STRONG' if abs(macd_hist) > 1 else 'WEAK'
    
    # Bollinger Band squeeze
    bb_upper = indicators.get('bb_upper', current_price)
    bb_lower = indicators.get('bb_lower', current_price)
    bb_width = ((bb_upper - bb_lower) / current_price) * 100 if current_price > 0 else 0
    bb_squeeze = 'YES' if bb_width < 3 else 'NO'
    
    # Volume analysis
    if len(price_data) >= 20:
        avg_volume = price_data['Volume'].tail(20).mean()
        recent_volume = price_data['Volume'].iloc[-1]
        volume_status = 'ABOVE_AVERAGE' if recent_volume > avg_volume else 'BELOW_AVERAGE'
    else:
        volume_status = 'UNKNOWN'
    
    # Trend strength score (0-100)
    trend_score = (tech_analysis.get('strength', 0) + 1) * 50  # Convert -1 to 1 scale to 0-100
    
    return {
        'rsi': rsi,
        'rsi_trend': rsi_trend,
        'macd_strength': macd_strength,
        'macd_histogram': macd_hist,
        'bb_squeeze': bb_squeeze,
        'bb_width': bb_width,
        'volume_status': volume_status,
        'trend_strength_score': trend_score,
        'trend_direction': tech_analysis.get('direction', 'NEUTRAL'),
        'technical_score': tech_analysis.get('technical_score', 0),
    }


def generate_enhanced_recommendation(pnl_percent, days_to_exp, prob_profit, greeks,
                                    sentiment, days_to_earnings, position_type,
                                    price_trends, earnings_intel, technicals, news_momentum):
    """Generate enhanced recommendation with all analysis factors"""
    
    # Base recommendation logic
    base_rec = generate_position_recommendation(
        pnl_percent, days_to_exp, prob_profit, greeks,
        sentiment, days_to_earnings, position_type
    )
    
    # Enhance with new factors
    confidence = base_rec['confidence']
    action = base_rec['action']
    reason = base_rec['reason']
    
    # Adjust confidence based on technicals
    tech_score = technicals.get('trend_strength_score', 50)
    if tech_score > 70:
        confidence = min(confidence + 5, 95)
        reason += " Strong technical momentum supports this."
    elif tech_score < 30:
        confidence = max(confidence - 5, 20)
        reason += " Weak technicals suggest caution."
    
    # Adjust for Greeks-based risks
    # High Theta + low time = urgency
    if abs(greeks['theta']) > 3 and days_to_exp < 10:
        confidence = min(confidence + 5, 95)
        reason += f" High time decay (${abs(greeks['theta']):.2f}/day) increases urgency."
    
    # Low Delta = less price sensitivity = lower confidence in holding
    if abs(greeks['delta']) < 0.3:
        confidence = max(confidence - 5, 20)
        reason += f" Low delta ({greeks['delta']:.3f}) reduces price move impact."
    
    # High Vega + earnings = IV crush risk
    if greeks['vega'] > 0.25 and earnings_intel.get('urgency') in ['CRITICAL', 'HIGH']:
        confidence = min(confidence + 10, 95)
        if 'EXIT' not in action:
            action = 'EXIT_BEFORE_EARNINGS'
        reason += f" High vega ({greeks['vega']:.2f}) makes IV crush very risky."
    
    # Adjust for earnings urgency
    if earnings_intel.get('urgency') == 'CRITICAL':
        if pnl_percent > 10:
            action = 'EXIT_BEFORE_EARNINGS'
            confidence = 80
            reason = f"Earnings in {days_to_earnings} days. Secure {pnl_percent:.1f}% profit to avoid IV crush risk."
    
    # Adjust for news momentum
    momentum_trend = news_momentum.get('trend', 'STABLE')
    if momentum_trend == 'DETERIORATING' and position_type == 'call':
        confidence = max(confidence - 10, 20)
        reason += " Negative news momentum detected."
    elif momentum_trend == 'IMPROVING' and position_type == 'call':
        confidence = min(confidence + 5, 95)
        reason += " Positive news momentum supports position."
    
    return {
        'action': action,
        'confidence': int(confidence),
        'reason': reason,
    }


def generate_enhanced_action_plan(ticker, current_price, strike, breakeven, days_to_exp,
                                 pnl_percent, position_type, greeks, days_to_earnings,
                                 sentiment, price_trends, earnings_intel, technicals):
    """Generate enhanced action plan with all factors"""
    
    plan = []
    
    # Critical insights first
    if earnings_intel.get('urgency') == 'CRITICAL':
        plan.append({
            'priority': 'high',
            'action': f'⚠️ CRITICAL: Earnings in {days_to_earnings} day(s)',
            'details': f"IV crush risk: {earnings_intel.get('iv_crush_risk', 'UNKNOWN')}. Recommendation: {earnings_intel.get('recommendation', 'HOLD')}."
        })
    
    # Price targets based on trends
    if price_trends:
        resistance = price_trends.get('resistance_level', strike * 1.1)
        support = price_trends.get('support_level', strike * 0.9)
        
        if position_type == 'call':
            plan.append({
                'priority': 'medium',
                'action': f'PROFIT TARGET: ${resistance:.2f}',
                'details': f'Take profit if stock reaches resistance at ${resistance:.2f}'
            })
            plan.append({
                'priority': 'medium',
                'action': f'STOP LOSS: ${support:.2f}',
                'details': f'Exit if stock breaks support at ${support:.2f}'
            })
    
    # Time-based actions
    if days_to_exp > 14:
        plan.append({
            'priority': 'low',
            'action': f'REVIEW in 7 days',
            'details': 'Re-evaluate position health and adjust strategy if needed.'
        })
    elif days_to_exp > 7:
        plan.append({
            'priority': 'medium',
            'action': f'DECISION POINT: {days_to_exp} days left',
            'details': 'Time decay accelerating. Decide this week: hold to expiration or exit early.'
        })
    else:
        plan.append({
            'priority': 'high',
            'action': f'⏰ URGENT: Only {days_to_exp} days until expiration',
            'details': 'Close position or prepare to exercise if ITM. Don\'t let expire worthless.'
        })
    
    # Technical indicators
    if technicals.get('rsi_trend') == 'FALLING' and position_type == 'call':
        plan.append({
            'priority': 'medium',
            'action': '📉 RSI Declining',
            'details': 'RSI trend weakening. Monitor closely for exit opportunity.'
        })
    
    return plan


def generate_exit_strategy(current_price, strike, breakeven, days_to_exp,
                          pnl_percent, position_type, greeks, earnings_intel, price_trends):
    """Generate comprehensive exit strategy"""
    
    strategy = {
        'take_profit_targets': [],
        'stop_loss_targets': [],
        'time_based_exits': [],
        'risk_reward': {},
    }
    
    # Take profit targets
    if position_type == 'call':
        tp1 = strike + (breakeven - strike) * 1.2
        tp2 = strike + (breakeven - strike) * 1.5
        tp3 = price_trends.get('resistance_level', strike * 1.1) if price_trends else strike * 1.1
        
        strategy['take_profit_targets'] = [
            {'price': tp1, 'reason': '20% profit target'},
            {'price': tp2, 'reason': '50% profit target'},
            {'price': tp3, 'reason': 'Resistance level'},
        ]
        
        # Stop loss
        sl = price_trends.get('support_level', strike * 0.9) if price_trends else strike * 0.9
        strategy['stop_loss_targets'] = [
            {'price': sl, 'reason': 'Support level break'}
        ]
    else:  # put
        tp1 = strike - (strike - breakeven) * 1.2
        tp2 = strike - (strike - breakeven) * 1.5
        tp3 = price_trends.get('support_level', strike * 0.9) if price_trends else strike * 0.9
        
        strategy['take_profit_targets'] = [
            {'price': tp1, 'reason': '20% profit target'},
            {'price': tp2, 'reason': '50% profit target'},
            {'price': tp3, 'reason': 'Support level'},
        ]
        
        # Stop loss
        sl = price_trends.get('resistance_level', strike * 1.1) if price_trends else strike * 1.1
        strategy['stop_loss_targets'] = [
            {'price': sl, 'reason': 'Resistance level break'}
        ]
    
    # Time-based exits
    if days_to_exp > 14:
        strategy['time_based_exits'].append({
            'date': (datetime.now() + timedelta(days=days_to_exp - 7)).strftime('%Y-%m-%d'),
            'reason': '7 days before expiration - avoid time decay'
        })
    
    if earnings_intel.get('days_to_earnings') and earnings_intel.get('days_to_earnings') <= 7:
        strategy['time_based_exits'].append({
            'date': (datetime.now() + timedelta(days=earnings_intel['days_to_earnings'] - 1)).strftime('%Y-%m-%d'),
            'reason': 'Day before earnings - avoid IV crush'
        })
    
    # Risk/reward
    if strategy['take_profit_targets']:
        tp_price = strategy['take_profit_targets'][0]['price']
        sl_price = strategy['stop_loss_targets'][0]['price'] if strategy['stop_loss_targets'] else breakeven
        
        if position_type == 'call':
            potential_profit = max(0, tp_price - current_price) * 100
            potential_loss = max(0, current_price - sl_price) * 100
        else:
            potential_profit = max(0, current_price - tp_price) * 100
            potential_loss = max(0, sl_price - current_price) * 100
        
        if potential_loss > 0:
            rr_ratio = potential_profit / potential_loss
        else:
            rr_ratio = 999
        
        strategy['risk_reward'] = {
            'ratio': rr_ratio,
            'potential_profit': potential_profit,
            'potential_loss': potential_loss,
        }
    
    return strategy


def generate_position_timeline(entry_date, expiration_date, contracts, current_price, strike,
                               breakeven, position_type, greeks, earnings_intel, price_trends,
                               sentiment, technicals, days_to_exp, pnl_percent):
    """
    Generate day-by-day timeline with recommendations for when to sell contracts
    
    Args:
        entry_date: Entry date string (YYYY-MM-DD)
        expiration_date: Expiration date string (YYYY-MM-DD)
        contracts: Number of contracts
        current_price: Current stock price
        strike: Strike price
        breakeven: Breakeven price
        position_type: 'call' or 'put'
        greeks: Greeks dictionary
        earnings_intel: Earnings intelligence dict
        price_trends: Price trends dict
        sentiment: Sentiment data
        technicals: Technical analysis dict
        days_to_exp: Days to expiration
        pnl_percent: Current P&L percentage
    
    Returns:
        List of day entries with recommendations
    """
    timeline = []
    entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
    exp_dt = datetime.strptime(expiration_date, '%Y-%m-%d')
    
    # Get current date in PST
    pst = pytz.timezone('America/Los_Angeles')
    today_utc = datetime.now(pytz.UTC)
    today_pst = today_utc.astimezone(pst)
    today = today_pst.replace(tzinfo=None)
    
    # Calculate days from entry to expiration
    total_days = (exp_dt - entry_dt).days
    
    # Generate timeline for each day
    for day_offset in range(total_days + 1):
        current_day = entry_dt + timedelta(days=day_offset)
        days_from_entry = day_offset
        days_until_exp = (exp_dt - current_day).days
        
        # Skip past dates
        if current_day < today:
            continue
        
        # Determine recommendation for this day
        recommendation = analyze_day_recommendation(
            current_day, days_until_exp, days_from_entry, contracts,
            earnings_intel, price_trends, sentiment, technicals,
            greeks, pnl_percent, position_type, breakeven, strike
        )
        
        timeline.append({
            'date': current_day.strftime('%Y-%m-%d'),
            'day_name': current_day.strftime('%A'),
            'days_from_entry': days_from_entry,
            'days_until_expiration': days_until_exp,
            'recommendation': recommendation['action'],
            'contracts_to_sell': recommendation['contracts_to_sell'],
            'reason': recommendation['reason'],
            'priority': recommendation['priority'],
            'indicators': recommendation['indicators'],
            'is_today': current_day.date() == today.date(),
            'is_earnings_day': (earnings_intel.get('earnings_date') and 
                               current_day.strftime('%Y-%m-%d') == earnings_intel['earnings_date']),
        })
    
    return timeline


def analyze_day_recommendation(day, days_until_exp, days_from_entry, total_contracts,
                               earnings_intel, price_trends, sentiment, technicals,
                               greeks, current_pnl, position_type, breakeven, strike):
    """
    Analyze what to do on a specific day
    
    Returns:
        dict with action, contracts_to_sell, reason, priority, indicators
    """
    # Default: hold all contracts
    action = 'HOLD'
    contracts_to_sell = 0
    reason = 'Continue monitoring position'
    priority = 'low'
    indicators = []
    
    # Check if it's earnings day
    if earnings_intel.get('earnings_date'):
        earnings_day = datetime.strptime(earnings_intel['earnings_date'], '%Y-%m-%d')
        days_to_earnings = (earnings_day - day).days
        
        # Day before earnings - high priority exit
        if days_to_earnings == 1:
            if current_pnl > 10:
                action = 'SELL_ALL'
                contracts_to_sell = total_contracts
                reason = f"Earnings tomorrow. Secure {current_pnl:.1f}% profit before IV crush risk."
                priority = 'critical'
                indicators.append('⚠️ Earnings Tomorrow')
                indicators.append('💰 Profit Protection')
            elif greeks.get('vega', 0) > 0.25:
                action = 'SELL_ALL'
                contracts_to_sell = total_contracts
                reason = f"High vega ({greeks['vega']:.2f}) + earnings tomorrow = high IV crush risk. Exit now."
                priority = 'critical'
                indicators.append('⚠️ Earnings Tomorrow')
                indicators.append('📉 High IV Risk')
            else:
                action = 'SELL_PARTIAL'
                contracts_to_sell = max(1, total_contracts // 2)
                reason = f"Earnings tomorrow. Sell {contracts_to_sell} contracts to reduce risk, hold rest for potential move."
                priority = 'high'
                indicators.append('⚠️ Earnings Tomorrow')
                indicators.append('🛡️ Risk Reduction')
        
        # Earnings day
        elif days_to_earnings == 0:
            action = 'SELL_ALL'
            contracts_to_sell = total_contracts
            reason = "Earnings today. Exit before announcement to avoid IV crush."
            priority = 'critical'
            indicators.append('📊 Earnings Today')
            indicators.append('🚨 Exit Immediately')
    
    # Last 3 days before expiration
    if days_until_exp <= 3 and days_until_exp > 0:
        if current_pnl > 5:
            # Profitable - sell all to avoid time decay
            action = 'SELL_ALL'
            contracts_to_sell = total_contracts
            reason = f"Only {days_until_exp} days left. Take {current_pnl:.1f}% profit before expiration."
            priority = 'high'
            indicators.append('⏰ Time Running Out')
            indicators.append('💰 Lock In Profit')
        elif current_pnl < -30:
            # Deep loss - cut losses
            action = 'SELL_ALL'
            contracts_to_sell = total_contracts
            reason = f"Only {days_until_exp} days left with {current_pnl:.1f}% loss. Cut losses before expiration."
            priority = 'high'
            indicators.append('⏰ Time Running Out')
            indicators.append('🛑 Stop Loss')
        else:
            # Small loss or break even - sell partial
            action = 'SELL_PARTIAL'
            contracts_to_sell = max(1, total_contracts // 2)
            reason = f"Only {days_until_exp} days left. Sell {contracts_to_sell} contracts to reduce risk."
            priority = 'medium'
            indicators.append('⏰ Time Running Out')
            indicators.append('🛡️ Risk Management')
    
    # Last week before expiration (4-7 days)
    elif days_until_exp <= 7 and days_until_exp > 3:
        if current_pnl > 20:
            # Good profit - sell 50-75% to lock in gains
            sell_pct = 0.75 if current_pnl > 30 else 0.5
            contracts_to_sell = max(1, int(total_contracts * sell_pct))
            action = 'SELL_PARTIAL'
            reason = f"{days_until_exp} days left with {current_pnl:.1f}% profit. Sell {contracts_to_sell} contracts to lock in gains."
            priority = 'high'
            indicators.append('📈 Strong Profit')
            indicators.append('💰 Lock In Gains')
        elif abs(greeks.get('theta', 0)) > 5:
            # High time decay - sell partial
            contracts_to_sell = max(1, total_contracts // 2)
            action = 'SELL_PARTIAL'
            reason = f"High time decay (${abs(greeks['theta']):.2f}/day). Sell {contracts_to_sell} contracts to reduce theta exposure."
            priority = 'medium'
            indicators.append('⏳ High Theta')
            indicators.append('🛡️ Reduce Decay')
    
    # Week 2 before expiration (8-14 days)
    elif days_until_exp <= 14 and days_until_exp > 7:
        if current_pnl > 30:
            # Very good profit - sell 25-50%
            contracts_to_sell = max(1, int(total_contracts * 0.33))
            action = 'SELL_PARTIAL'
            reason = f"{current_pnl:.1f}% profit with {days_until_exp} days left. Take partial profit on {contracts_to_sell} contracts."
            priority = 'medium'
            indicators.append('📈 Strong Profit')
            indicators.append('💰 Partial Profit')
        elif abs(greeks.get('delta', 0)) < 0.2:
            # Low delta - position not moving much
            contracts_to_sell = max(1, total_contracts // 3)
            action = 'SELL_PARTIAL'
            reason = f"Low delta ({greeks['delta']:.3f}) means low price sensitivity. Sell {contracts_to_sell} contracts to reduce exposure."
            priority = 'low'
            indicators.append('📊 Low Delta')
            indicators.append('🛡️ Reduce Exposure')
    
    # Check technical indicators
    if technicals.get('rsi_trend') == 'FALLING' and position_type == 'call':
        if action == 'HOLD':
            contracts_to_sell = max(1, total_contracts // 4)
            action = 'SELL_PARTIAL'
            reason = f"RSI declining. Consider selling {contracts_to_sell} contracts to reduce risk."
            priority = 'medium'
            indicators.append('📉 RSI Declining')
    
    # Check sentiment
    if sentiment.get('overall_sentiment', 0) < -0.3 and position_type == 'call':
        if action == 'HOLD' and days_until_exp > 7:
            contracts_to_sell = max(1, total_contracts // 3)
            action = 'SELL_PARTIAL'
            reason = f"Negative sentiment detected. Sell {contracts_to_sell} contracts to reduce risk."
            priority = 'medium'
            indicators.append('📰 Negative News')
    
    # High profit scenario - always consider taking some profit
    if current_pnl > 50 and action == 'HOLD':
        contracts_to_sell = max(1, int(total_contracts * 0.5))
        action = 'SELL_PARTIAL'
        reason = f"Exceptional {current_pnl:.1f}% profit. Take profit on {contracts_to_sell} contracts."
        priority = 'high'
        indicators.append('🎯 Exceptional Profit')
        indicators.append('💰 Take Profit')
    
    return {
        'action': action,
        'contracts_to_sell': contracts_to_sell,
        'reason': reason,
        'priority': priority,
        'indicators': indicators,
    }
