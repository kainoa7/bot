#!/usr/bin/env python3
"""
Stock Options Trading Assistant - Main CLI
"""
import sys
import argparse
from datetime import datetime
from data_collector import DataCollector
from indicators import TechnicalAnalyzer
from sentiment_analyzer import SentimentAnalyzer
from recommendation import RecommendationEngine


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def print_section(title: str):
    """Print section title"""
    print(f"\n{title}")
    print("-" * len(title))


def format_price(price: float) -> str:
    """Format price with dollar sign"""
    return f"${price:.2f}"


def analyze_ticker(ticker: str, verbose: bool = False):
    """
    Analyze a stock ticker and provide options trading recommendation
    
    Args:
        ticker: Stock ticker symbol
        verbose: Show detailed analysis
    """
    try:
        print_header(f"ANALYZING {ticker.upper()}")
        
        # Step 1: Collect Data
        print("\n📊 Collecting market data...")
        collector = DataCollector(ticker)
        
        try:
            price_data = collector.get_price_data()
            print(f"  ✓ Price data: {len(price_data)} days")
        except Exception as e:
            print(f"  ✗ Error fetching price data: {str(e)}")
            return
        
        fundamentals = collector.get_fundamentals()
        if fundamentals:
            print(f"  ✓ Fundamental data retrieved")
        
        news = collector.get_news()
        print(f"  ✓ News articles: {len(news)}")
        
        options_data = collector.get_options_data()
        if options_data.get('available'):
            print(f"  ✓ Options data available")
        
        earnings_date = collector.get_earnings_date()
        if earnings_date:
            days_to_earnings = (earnings_date - datetime.now()).days
            print(f"  ✓ Next earnings: {earnings_date.strftime('%Y-%m-%d')} ({days_to_earnings} days)")
        
        # Step 2: Technical Analysis
        print("\n📈 Running technical analysis...")
        tech_analyzer = TechnicalAnalyzer(price_data)
        tech_analysis = tech_analyzer.analyze_signals()
        print(f"  ✓ Technical direction: {tech_analysis['direction']}")
        print(f"  ✓ Technical score: {tech_analysis['technical_score']:.2f}")
        
        # Step 3: Sentiment Analysis
        print("\n📰 Analyzing news sentiment...")
        sent_analyzer = SentimentAnalyzer(news)
        sent_analysis = sent_analyzer.get_sentiment_signal()
        print(f"  ✓ Sentiment: {sent_analysis['signal']}")
        print(f"  ✓ Sentiment score: {sent_analysis['score']:.2f}")
        
        # Step 4: Generate Recommendation
        print("\n🤖 Generating recommendation...")
        engine = RecommendationEngine(ticker)
        engine.add_technical_analysis(tech_analysis)
        engine.add_sentiment_analysis(sent_analysis)
        engine.add_fundamentals(fundamentals)
        engine.add_options_data(options_data)
        engine.add_earnings_date(earnings_date)
        
        recommendation = engine.generate_recommendation()
        
        # Display Results
        display_recommendation(recommendation, verbose)
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def display_recommendation(rec: dict, verbose: bool = False):
    """Display the trading recommendation"""
    
    print_header("TRADING RECOMMENDATION")
    
    # Main recommendation
    print(f"\n🎯 Ticker: {rec['ticker']}")
    print(f"💰 Current Price: {format_price(rec['current_price'])}")
    print(f"\n📍 RECOMMENDATION: {rec['recommendation']}")
    print(f"   Direction: {rec['direction_label']}")
    print(f"   Confidence: {rec['confidence_label']} ({rec['confidence']:.0%})")
    
    # Strike prices
    print_section("💵 SUGGESTED STRIKE PRICES")
    strike_info = rec['strike_prices']
    print(f"Strategy: {strike_info['recommended']}")
    greeks_available = strike_info.get('greeks', [])
    
    for i, (strike, label) in enumerate(zip(strike_info['strikes'], strike_info.get('labels', []))):
        print(f"\n  {i+1}. {format_price(strike)} - {label}")
        
        if greeks_available and i < len(greeks_available) and greeks_available[i]:
            greeks = greeks_available[i]
            print(f"     • Estimated Premium: {format_price(greeks.get('theoretical_price', 0))}")
            print(f"     • Delta: {greeks.get('delta', 0):.3f} | Theta: ${greeks.get('theta', 0):.2f}/day")
            print(f"     • Probability of Profit: {greeks.get('prob_profit', 0):.1%}")
            print(f"     • Breakeven: {format_price(greeks.get('breakeven', 0))}")
    
    # Expiration
    print_section("📅 EXPIRATION RECOMMENDATION")
    exp_info = rec['expiration']
    print(f"Timeframe: {exp_info['recommended']}")
    print(f"Date Range: {exp_info['min_date'].strftime('%Y-%m-%d')} to {exp_info['max_date'].strftime('%Y-%m-%d')}")
    print(f"Reason: {exp_info['reason']}")
    
    # Options metrics
    if rec['implied_volatility'] > 0:
        print_section("📊 OPTIONS METRICS")
        print(f"Implied Volatility: {rec['implied_volatility']:.1f}%")
    
    # Risk factors
    print_section("⚠️  RISK FACTORS")
    for risk in rec['risk_factors']:
        print(f"  • {risk}")
    
    # Detailed analysis (verbose mode)
    if verbose:
        print_section("📊 COMPONENT SCORES")
        scores = rec['component_scores']
        print(f"  Technical:   {scores['technical']:+.2f}")
        print(f"  Sentiment:   {scores['sentiment']:+.2f}")
        print(f"  Fundamental: {scores['fundamental']:+.2f}")
        print(f"  Options:     {scores['options']:+.2f}")
        print(f"  ─────────────────────")
        print(f"  Final Score: {scores['final']:+.2f}")
        
        print_section("🔍 ANALYSIS SIGNALS")
        for i, signal in enumerate(rec['signals'], 1):
            print(f"  {i}. {signal}")
    
    # Disclaimer
    print_section("⚠️  DISCLAIMER")
    print("This analysis is for educational purposes only and should not be considered")
    print("financial advice. Options trading involves substantial risk and is not suitable")
    print("for all investors. Always conduct your own research and consult with a licensed")
    print("financial advisor before making investment decisions.")
    
    print("\n" + "="*70 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Stock Options Trading Assistant - AI-powered options analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py AAPL           # Analyze Apple stock
  python main.py TSLA -v        # Verbose analysis of Tesla
  python main.py MSFT --verbose # Detailed Microsoft analysis

Note: Make sure to set up your API keys in .env file before running.
        """
    )
    
    parser.add_argument(
        'ticker',
        type=str,
        help='Stock ticker symbol (e.g., AAPL, TSLA, MSFT)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed analysis with all signals and scores'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Stock Options Trading Assistant v1.0.0'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyze_ticker(args.ticker, args.verbose)


if __name__ == "__main__":
    main()
