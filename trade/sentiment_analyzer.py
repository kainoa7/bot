"""
Sentiment analysis for news articles
"""
from textblob import TextBlob
from typing import List, Dict
from datetime import datetime, timedelta
import config


class SentimentAnalyzer:
    """Analyzes sentiment from news articles"""
    
    def __init__(self, news_articles: List[Dict]):
        self.articles = news_articles
        
    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment polarity score (-1 to 1)
        """
        if not text:
            return 0.0
        
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def analyze_article(self, article: Dict) -> Dict:
        """
        Analyze sentiment of a single article
        
        Args:
            article: Article dictionary with title and description
            
        Returns:
            Dictionary with sentiment scores
        """
        title = article.get('title', '')
        description = article.get('description', '')
        
        # Combine title and description, giving more weight to title
        combined_text = f"{title} {title} {description}"
        
        sentiment = self.analyze_text(combined_text)
        
        return {
            'title': title,
            'sentiment': sentiment,
            'label': self._get_sentiment_label(sentiment),
        }
    
    def analyze_all(self) -> Dict:
        """
        Analyze sentiment of all news articles
        
        Returns:
            Dictionary with aggregated sentiment analysis
        """
        if not self.articles:
            return {
                'available': False,
                'sentiment_score': 0,
                'sentiment_label': 'NEUTRAL',
                'articles_analyzed': 0,
            }
        
        sentiments = []
        article_sentiments = []
        
        for article in self.articles:
            result = self.analyze_article(article)
            sentiments.append(result['sentiment'])
            article_sentiments.append(result)
        
        # Calculate aggregate metrics
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        positive_count = sum(1 for s in sentiments if s > config.SENTIMENT_THRESHOLD_POSITIVE)
        negative_count = sum(1 for s in sentiments if s < config.SENTIMENT_THRESHOLD_NEGATIVE)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        # Calculate weighted sentiment (recent articles have more weight)
        weighted_sentiment = 0
        total_weight = 0
        for i, sentiment in enumerate(sentiments):
            weight = len(sentiments) - i  # More recent = higher weight
            weighted_sentiment += sentiment * weight
            total_weight += weight
        
        weighted_avg = weighted_sentiment / total_weight if total_weight > 0 else 0
        
        return {
            'available': True,
            'sentiment_score': weighted_avg,
            'sentiment_label': self._get_sentiment_label(weighted_avg),
            'avg_sentiment': avg_sentiment,
            'articles_analyzed': len(sentiments),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'article_details': article_sentiments[:5],  # Top 5 for display
        }
    
    def _get_sentiment_label(self, score: float) -> str:
        """
        Convert sentiment score to label
        
        Args:
            score: Sentiment score (-1 to 1)
            
        Returns:
            Label string
        """
        if score > config.SENTIMENT_THRESHOLD_POSITIVE:
            if score > 0.3:
                return 'VERY POSITIVE'
            return 'POSITIVE'
        elif score < config.SENTIMENT_THRESHOLD_NEGATIVE:
            if score < -0.3:
                return 'VERY NEGATIVE'
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'
    
    def get_sentiment_signal(self) -> Dict:
        """
        Get trading signal based on sentiment
        
        Returns:
            Dictionary with sentiment signal and score
        """
        analysis = self.analyze_all()
        
        if not analysis['available']:
            return {
                'signal': 'NEUTRAL',
                'score': 0,
                'confidence': 0,
                'reason': 'No news data available',
                'overall_sentiment': 0,
                'news_count': 0,
                'headlines': [],
            }
        
        score = analysis['sentiment_score']
        
        # Convert sentiment to trading signal
        if score > 0.2:
            signal = 'BULLISH'
            confidence = min(abs(score), 1.0)
            reason = f"Strong positive sentiment ({analysis['positive_count']} positive articles)"
        elif score < -0.2:
            signal = 'BEARISH'
            confidence = min(abs(score), 1.0)
            reason = f"Strong negative sentiment ({analysis['negative_count']} negative articles)"
        else:
            signal = 'NEUTRAL'
            confidence = 0.3
            reason = "Mixed or neutral sentiment"
        
        # Get headlines with sentiment
        headlines = []
        for article in analysis.get('article_details', [])[:7]:
            headlines.append({
                'title': article['title'],
                'sentiment': article['sentiment'],
                'label': article['label']
            })
        
        return {
            'signal': signal,
            'score': score,  # -1 to 1
            'confidence': confidence,
            'reason': reason,
            'articles_count': analysis['articles_analyzed'],
            'overall_sentiment': score,
            'news_count': analysis['articles_analyzed'],
            'headlines': headlines,
        }
    
    def get_news_momentum(self) -> Dict:
        """
        Calculate news momentum trend over time
        
        Returns:
            Dictionary with momentum analysis
        """
        if not self.articles:
            return {
                'trend': 'NEUTRAL',
                'momentum_score': 0,
                'recent_sentiment': 0,
                'older_sentiment': 0,
            }
        
        # Split articles into recent (last 24h) and older
        now = datetime.now()
        recent_articles = []
        older_articles = []
        
        for article in self.articles:
            pub_time = article.get('published_at', '')
            if not pub_time:
                older_articles.append(article)
                continue
            
            try:
                # Try parsing ISO format
                if 'T' in pub_time:
                    pub_dt = datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
                else:
                    pub_dt = datetime.fromtimestamp(int(pub_time))
                
                # Check if within last 24 hours
                if (now - pub_dt.replace(tzinfo=None)).total_seconds() < 86400:
                    recent_articles.append(article)
                else:
                    older_articles.append(article)
            except:
                older_articles.append(article)
        
        # Calculate sentiment for each group
        recent_sentiments = [self.analyze_article(a)['sentiment'] for a in recent_articles]
        older_sentiments = [self.analyze_article(a)['sentiment'] for a in older_articles]
        
        recent_avg = sum(recent_sentiments) / len(recent_sentiments) if recent_sentiments else 0
        older_avg = sum(older_sentiments) / len(older_sentiments) if older_sentiments else 0
        
        momentum_score = recent_avg - older_avg
        
        if momentum_score > 0.1:
            trend = 'IMPROVING'
        elif momentum_score < -0.1:
            trend = 'DETERIORATING'
        else:
            trend = 'STABLE'
        
        return {
            'trend': trend,
            'momentum_score': momentum_score,
            'recent_sentiment': recent_avg,
            'older_sentiment': older_avg,
            'recent_count': len(recent_articles),
            'older_count': len(older_articles),
        }
    
    def get_breaking_news(self, hours: int = 24) -> List[Dict]:
        """
        Get breaking news from the last N hours
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of breaking news articles with sentiment
        """
        if not self.articles:
            return []
        
        now = datetime.now()
        breaking = []
        
        for article in self.articles:
            pub_time = article.get('published_at', '')
            if not pub_time:
                continue
            
            try:
                if 'T' in pub_time:
                    pub_dt = datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
                else:
                    pub_dt = datetime.fromtimestamp(int(pub_time))
                
                hours_ago = (now - pub_dt.replace(tzinfo=None)).total_seconds() / 3600
                
                if hours_ago <= hours:
                    analyzed = self.analyze_article(article)
                    breaking.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'sentiment': analyzed['sentiment'],
                        'label': analyzed['label'],
                        'hours_ago': round(hours_ago, 1),
                    })
            except:
                continue
        
        # Sort by most recent
        breaking.sort(key=lambda x: x.get('hours_ago', 999))
        return breaking[:5]  # Top 5 breaking news
    
    def get_critical_news(self) -> List[Dict]:
        """
        Identify critical news (earnings, FDA, lawsuits, etc.)
        
        Returns:
            List of critical news articles
        """
        if not self.articles:
            return []
        
        critical_keywords = [
            'earnings', 'fda', 'approval', 'lawsuit', 'settlement', 'merger', 'acquisition',
            'bankruptcy', 'delisting', 'recall', 'investigation', 'subpoena', 'ceo', 'cfo',
            'resignation', 'guidance', 'forecast', 'downgrade', 'upgrade', 'rating'
        ]
        
        critical = []
        for article in self.articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            text = f"{title} {description}"
            
            # Check for critical keywords
            for keyword in critical_keywords:
                if keyword in text:
                    analyzed = self.analyze_article(article)
                    critical.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'sentiment': analyzed['sentiment'],
                        'label': analyzed['label'],
                        'keyword': keyword,
                    })
                    break
        
        return critical[:7]  # Top 7 critical news


if __name__ == "__main__":
    # Test the sentiment analyzer
    from data_collector import DataCollector
    
    collector = DataCollector("AAPL")
    news = collector.get_news()
    
    analyzer = SentimentAnalyzer(news)
    analysis = analyzer.analyze_all()
    signal = analyzer.get_sentiment_signal()
    
    print("Sentiment Analysis Results:")
    print(f"Overall Sentiment: {analysis['sentiment_label']}")
    print(f"Sentiment Score: {analysis['sentiment_score']:.3f}")
    print(f"Articles Analyzed: {analysis['articles_analyzed']}")
    print(f"Positive: {analysis['positive_count']}, Negative: {analysis['negative_count']}, Neutral: {analysis['neutral_count']}")
    
    print("\nTrading Signal:")
    print(f"Signal: {signal['signal']}")
    print(f"Confidence: {signal['confidence']:.2f}")
    print(f"Reason: {signal['reason']}")
    
    print("\nRecent Article Sentiments:")
    for article in analysis['article_details']:
        print(f"  [{article['label']}] {article['title'][:60]}...")
