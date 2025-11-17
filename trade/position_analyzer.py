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
        today = datetime.now()
        
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
        
        # Get sentiment and news
        news = collector.get_news()
        sent_analyzer = SentimentAnalyzer(news)
        sentiment_data = sent_analyzer.get_sentiment_signal()
        
        # Get earnings date
        earnings_date = get_next_earnings_date(stock)
        days_to_earnings = None
        if earnings_date:
            earnings_dt = datetime.strptime(earnings_date, '%Y-%m-%d')
            days_to_earnings = (earnings_dt - today).days
        
        # Calculate probability of profit
        prob_profit = calculate_profit_probability(
            current_price, breakeven, days_to_expiration,
            info.get('impliedVolatility', 0.3), position_type
        )
        
        # Generate recommendation
        recommendation = generate_position_recommendation(
            pnl_percent, days_to_expiration, prob_profit, greeks,
            sentiment_data, days_to_earnings, position_type
        )
        
        # Generate action plan
        action_plan = generate_action_plan(
            ticker, current_price, strike_price, breakeven, 
            days_to_expiration, pnl_percent, position_type,
            greeks, days_to_earnings, sentiment_data
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
            
            # Sentiment
            'sentiment_score': sentiment_data['overall_sentiment'],
            'news_count': sentiment_data['news_count'],
            'recent_news': sentiment_data['headlines'][:3],
            
            # Recommendation
            'recommendation': recommendation,
            'action_plan': action_plan,
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
