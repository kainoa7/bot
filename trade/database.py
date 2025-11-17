"""
Database models for personal financial dashboard
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Position(db.Model):
    """Options/Stock/Crypto positions"""
    __tablename__ = 'positions'
    
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), nullable=False)
    asset_type = db.Column(db.String(20), nullable=False)  # 'option', 'stock', 'crypto'
    position_type = db.Column(db.String(10))  # 'call', 'put', 'long', 'short' (for options)
    strike_price = db.Column(db.Float)  # For options
    premium_paid = db.Column(db.Float)  # For options
    contracts = db.Column(db.Integer, default=1)
    shares = db.Column(db.Float)  # For stocks/crypto
    entry_price = db.Column(db.Float)
    entry_date = db.Column(db.Date, nullable=False)
    expiration_date = db.Column(db.Date)  # For options
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'ticker': self.ticker,
            'asset_type': self.asset_type,
            'position_type': self.position_type,
            'strike_price': self.strike_price,
            'premium_paid': self.premium_paid,
            'contracts': self.contracts,
            'shares': self.shares,
            'entry_price': self.entry_price,
            'entry_date': self.entry_date.strftime('%Y-%m-%d') if self.entry_date else None,
            'expiration_date': self.expiration_date.strftime('%Y-%m-%d') if self.expiration_date else None,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class Account(db.Model):
    """Bank accounts, savings, checking accounts"""
    __tablename__ = 'accounts'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)  # "Chase Checking", "Savings Account"
    account_type = db.Column(db.String(50), nullable=False)  # 'checking', 'savings', 'investment', 'crypto_wallet'
    institution = db.Column(db.String(100))  # "Chase", "Bank of America", "Coinbase"
    balance = db.Column(db.Float, default=0.0)
    currency = db.Column(db.String(10), default='USD')
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'account_type': self.account_type,
            'institution': self.institution,
            'balance': self.balance,
            'currency': self.currency,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

