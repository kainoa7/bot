#!/usr/bin/env python3
"""
Stock Options Trading Assistant - Web Interface
"""
from flask import Flask, render_template, request, jsonify
from datetime import datetime
from data_collector import DataCollector
from indicators import TechnicalAnalyzer
from sentiment_analyzer import SentimentAnalyzer
from recommendation import RecommendationEngine
from position_analyzer import analyze_existing_position

app = Flask(__name__)


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze a ticker and return recommendation"""
    try:
        ticker = request.json.get('ticker', '').strip().upper()
        
        if not ticker:
            return jsonify({'error': 'Please enter a ticker symbol'}), 400
        
        # Collect data
        collector = DataCollector(ticker)
        
        try:
            price_data = collector.get_price_data()
        except Exception as e:
            return jsonify({'error': f'Could not fetch data for {ticker}. Please check the ticker symbol.'}), 400
        
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
        
        # Format response
        response = {
            'ticker': recommendation['ticker'],
            'current_price': recommendation['current_price'],
            'recommendation': recommendation['recommendation'],
            'direction_label': recommendation['direction_label'],
            'confidence': recommendation['confidence'],
            'confidence_label': recommendation['confidence_label'],
            'strike_prices': {
                'recommended': recommendation['strike_prices']['recommended'],
                'strikes': [
                    {
                        'price': strike,
                        'label': label
                    }
                    for strike, label in zip(
                        recommendation['strike_prices']['strikes'],
                        recommendation['strike_prices'].get('labels', [])
                    )
                ]
            },
            'expiration': {
                'recommended': recommendation['expiration']['recommended'],
                'min_date': recommendation['expiration']['min_date'].strftime('%Y-%m-%d'),
                'max_date': recommendation['expiration']['max_date'].strftime('%Y-%m-%d'),
                'reason': recommendation['expiration']['reason']
            },
            'implied_volatility': recommendation['implied_volatility'],
            'risk_factors': recommendation['risk_factors'],
            'component_scores': recommendation['component_scores'],
            'signals': recommendation['signals'],
            'earnings_date': earnings_date.strftime('%Y-%m-%d') if earnings_date else None,
            'news_count': len(news),
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500


@app.route('/analyze-position', methods=['POST'])
def analyze_position():
    """Analyze an existing options position"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['ticker', 'type', 'strike', 'premium', 'contracts', 
                          'entry_date', 'expiration_date']
        
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Analyze position
        result = analyze_existing_position(
            ticker=data['ticker'].strip().upper(),
            position_type=data['type'].lower(),
            strike_price=float(data['strike']),
            premium_paid=float(data['premium']),
            contracts=int(data['contracts']),
            entry_date=data['entry_date'],
            expiration_date=data['expiration_date']
        )
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("  Stock Options Trading Assistant - Web Interface")
    print("="*70)
    print("\n🚀 Starting web server...")
    print("📱 Open your browser to: http://localhost:5001")
    print("🛑 Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
