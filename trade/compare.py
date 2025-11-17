#!/usr/bin/env python3
"""
Compare multiple tickers to find the best options trading opportunity
"""
import sys
from main import analyze_ticker
from data_collector import DataCollector
from indicators import TechnicalAnalyzer
from sentiment_analyzer import SentimentAnalyzer
from recommendation import RecommendationEngine


def compare_tickers(tickers: list):
    """Compare multiple tickers and rank them"""
    
    print("\n" + "="*70)
    print("  COMPARING MULTIPLE TICKERS")
    print("="*70)
    
    results = []
    
    for ticker in tickers:
        print(f"\n🔍 Analyzing {ticker}...")
        
        try:
            # Collect and analyze
            collector = DataCollector(ticker)
            price_data = collector.get_price_data()
            fundamentals = collector.get_fundamentals()
            news = collector.get_news()
            options_data = collector.get_options_data()
            earnings_date = collector.get_earnings_date()
            
            tech_analyzer = TechnicalAnalyzer(price_data)
            tech_analysis = tech_analyzer.analyze_signals()
            
            sent_analyzer = SentimentAnalyzer(news)
            sent_analysis = sent_analyzer.get_sentiment_signal()
            
            engine = RecommendationEngine(ticker)
            engine.add_technical_analysis(tech_analysis)
            engine.add_sentiment_analysis(sent_analysis)
            engine.add_fundamentals(fundamentals)
            engine.add_options_data(options_data)
            engine.add_earnings_date(earnings_date)
            
            rec = engine.generate_recommendation()
            
            results.append({
                'ticker': ticker,
                'recommendation': rec['recommendation'],
                'confidence': rec['confidence'],
                'confidence_label': rec['confidence_label'],
                'price': rec['current_price'],
                'final_score': rec['component_scores']['final'],
                'iv': rec['implied_volatility'],
                'prob_profit': rec['strike_prices'].get('greeks', [{}])[0].get('prob_profit', 0) if rec['strike_prices'].get('greeks') else 0,
            })
            
            print(f"   ✓ {rec['recommendation']} - {rec['confidence_label']} confidence")
            
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
            continue
    
    if not results:
        print("\n❌ No successful analyses")
        return
    
    # Sort by confidence * abs(final_score)
    results.sort(key=lambda x: x['confidence'] * abs(x['final_score']), reverse=True)
    
    # Display comparison table
    print("\n" + "="*70)
    print("  RANKING (Best to Worst)")
    print("="*70)
    print(f"\n{'Rank':<6}{'Ticker':<8}{'Action':<8}{'Confidence':<12}{'Score':<8}{'IV':<8}{'Prob%':<8}")
    print("-" * 70)
    
    for i, result in enumerate(results, 1):
        prob_str = f"{result['prob_profit']:.0%}" if result['prob_profit'] > 0 else "N/A"
        iv_str = f"{result['iv']:.0f}%" if result['iv'] > 0 else "N/A"
        
        print(f"{i:<6}{result['ticker']:<8}{result['recommendation']:<8}"
              f"{result['confidence_label']:<12}{result['final_score']:+.2f}   "
              f"{iv_str:<8}{prob_str:<8}")
    
    # Show recommendation
    best = results[0]
    print("\n" + "="*70)
    print(f"🏆 BEST OPPORTUNITY: {best['ticker']}")
    print("="*70)
    print(f"Action: {best['recommendation']}")
    print(f"Current Price: ${best['price']:.2f}")
    print(f"Confidence: {best['confidence_label']} ({best['confidence']:.0%})")
    print(f"Combined Score: {best['final_score']:+.2f}")
    if best['iv'] > 0:
        print(f"Implied Volatility: {best['iv']:.1f}%")
    if best['prob_profit'] > 0:
        print(f"Est. Probability of Profit: {best['prob_profit']:.1%}")
    
    print("\n💡 Run detailed analysis:")
    print(f"   python main.py {best['ticker']} --verbose")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare.py TICKER1 TICKER2 TICKER3 ...")
        print("Example: python compare.py AAPL TSLA NVDA MSFT")
        sys.exit(1)
    
    tickers = [t.upper() for t in sys.argv[1:]]
    compare_tickers(tickers)
