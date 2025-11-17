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
from database import db, Position, Account
import os

app = Flask(__name__)

# Database configuration
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "portfolio.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Create tables
with app.app_context():
    db.create_all()


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


# Dashboard API endpoints
@app.route('/api/positions', methods=['GET'])
def get_positions():
    """Get all positions"""
    positions = Position.query.order_by(Position.created_at.desc()).all()
    return jsonify([p.to_dict() for p in positions])


@app.route('/api/positions', methods=['POST'])
def create_position():
    """Create a new position"""
    try:
        data = request.json
        
        position = Position(
            ticker=data.get('ticker', '').strip().upper(),
            asset_type=data.get('asset_type', 'option'),
            position_type=data.get('position_type'),
            strike_price=data.get('strike_price'),
            premium_paid=data.get('premium_paid'),
            contracts=data.get('contracts', 1),
            shares=data.get('shares'),
            entry_price=data.get('entry_price'),
            entry_date=datetime.strptime(data['entry_date'], '%Y-%m-%d').date(),
            expiration_date=datetime.strptime(data['expiration_date'], '%Y-%m-%d').date() if data.get('expiration_date') else None,
            notes=data.get('notes')
        )
        
        db.session.add(position)
        db.session.commit()
        
        return jsonify(position.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@app.route('/api/positions/<int:position_id>', methods=['PUT'])
def update_position(position_id):
    """Update a position"""
    try:
        position = Position.query.get_or_404(position_id)
        data = request.json
        
        if 'ticker' in data:
            position.ticker = data['ticker'].strip().upper()
        if 'asset_type' in data:
            position.asset_type = data['asset_type']
        if 'position_type' in data:
            position.position_type = data['position_type']
        if 'strike_price' in data:
            position.strike_price = data['strike_price']
        if 'premium_paid' in data:
            position.premium_paid = data['premium_paid']
        if 'contracts' in data:
            position.contracts = data['contracts']
        if 'shares' in data:
            position.shares = data['shares']
        if 'entry_price' in data:
            position.entry_price = data['entry_price']
        if 'entry_date' in data:
            position.entry_date = datetime.strptime(data['entry_date'], '%Y-%m-%d').date()
        if 'expiration_date' in data:
            position.expiration_date = datetime.strptime(data['expiration_date'], '%Y-%m-%d').date() if data['expiration_date'] else None
        if 'notes' in data:
            position.notes = data['notes']
        
        position.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify(position.to_dict())
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@app.route('/api/positions/<int:position_id>', methods=['GET'])
def get_position(position_id):
    """Get a single position"""
    position = Position.query.get_or_404(position_id)
    return jsonify(position.to_dict())


@app.route('/api/positions/<int:position_id>', methods=['DELETE'])
def delete_position(position_id):
    """Delete a position"""
    try:
        position = Position.query.get_or_404(position_id)
        db.session.delete(position)
        db.session.commit()
        return jsonify({'message': 'Position deleted'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@app.route('/api/accounts', methods=['GET'])
def get_accounts():
    """Get all accounts"""
    accounts = Account.query.order_by(Account.created_at.desc()).all()
    return jsonify([a.to_dict() for a in accounts])


@app.route('/api/accounts', methods=['POST'])
def create_account():
    """Create a new account"""
    try:
        data = request.json
        
        account = Account(
            name=data.get('name'),
            account_type=data.get('account_type', 'checking'),
            institution=data.get('institution'),
            balance=float(data.get('balance', 0)),
            currency=data.get('currency', 'USD'),
            notes=data.get('notes')
        )
        
        db.session.add(account)
        db.session.commit()
        
        return jsonify(account.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@app.route('/api/accounts/<int:account_id>', methods=['PUT'])
def update_account(account_id):
    """Update an account"""
    try:
        account = Account.query.get_or_404(account_id)
        data = request.json
        
        if 'name' in data:
            account.name = data['name']
        if 'account_type' in data:
            account.account_type = data['account_type']
        if 'institution' in data:
            account.institution = data['institution']
        if 'balance' in data:
            account.balance = float(data['balance'])
        if 'currency' in data:
            account.currency = data['currency']
        if 'notes' in data:
            account.notes = data['notes']
        
        account.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify(account.to_dict())
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@app.route('/api/accounts/<int:account_id>', methods=['DELETE'])
def delete_account(account_id):
    """Delete an account"""
    try:
        account = Account.query.get_or_404(account_id)
        db.session.delete(account)
        db.session.commit()
        return jsonify({'message': 'Account deleted'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@app.route('/api/dashboard', methods=['GET'])
def get_dashboard():
    """Get dashboard summary"""
    positions = Position.query.all()
    accounts = Account.query.all()
    
    # Calculate totals
    total_accounts_balance = sum(a.balance for a in accounts)
    
    # Group positions by type
    options_positions = [p for p in positions if p.asset_type == 'option']
    stock_positions = [p for p in positions if p.asset_type == 'stock']
    crypto_positions = [p for p in positions if p.asset_type == 'crypto']
    
    return jsonify({
        'total_positions': len(positions),
        'options_count': len(options_positions),
        'stocks_count': len(stock_positions),
        'crypto_count': len(crypto_positions),
        'total_accounts': len(accounts),
        'total_accounts_balance': total_accounts_balance,
        'positions': [p.to_dict() for p in positions],
        'accounts': [a.to_dict() for a in accounts],
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("  Stock Options Trading Assistant - Web Interface")
    print("="*70)
    print("\n🚀 Starting web server...")
    print("📱 Open your browser to: http://localhost:5001")
    print("🛑 Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
