import urllib.request
import xml.etree.ElementTree as ET
import random
import re
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class SentimentSnapshot:
    timestamp: pd.Timestamp
    sentiment_score: float      # -1 to +1
    confidence: float           # 0 to 1
    positive_count: int
    negative_count: int
    neutral_count: int
    bullish_ratio: float
    bearish_ratio: float
    probability_up: float
    probability_down: float

class SentimentEngine:
    """
    NLP Sentiment Engine.
    Fetches live RSS feeds (e.g., CoinTelegraph) for live evaluation, or uses mock generator for historical backtesting.
    """
    def __init__(self, model_name: str = "keyword"):
        self.model_name = model_name
        
        # Keyword-based heuristic scoring
        self.bullish_keywords = {'bull', 'bullish', 'surge', 'jump', 'moon', 'adoption', 'etf', 'upgrade', 'breakout', 'ath', 'buy'}
        self.bearish_keywords = {'bear', 'bearish', 'drop', 'crash', 'hack', 'ban', 'sec', 'sell', 'scam', 'plunge', 'lawsuit'}

    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process a single piece of text and return sentiment based on keyword heuristics.
        """
        words = set(re.findall(r'\b\w+\b', text.lower()))
        
        bull_matches = len(words.intersection(self.bullish_keywords))
        bear_matches = len(words.intersection(self.bearish_keywords))
        
        if bull_matches > bear_matches:
            score = min(1.0, (bull_matches - bear_matches) * 0.25 + 0.1)
            label = "POSITIVE"
        elif bear_matches > bull_matches:
            score = max(-1.0, (bull_matches - bear_matches) * 0.25 - 0.1)
            label = "NEGATIVE"
        else:
            score = 0.0
            label = "NEUTRAL"
            
        return {
            "sentiment_score": score,
            "confidence": 0.8 if label != "NEUTRAL" else 0.5,
            "label": label
        }

    def fetch_live_headlines(self) -> List[str]:
        """Fetches live RSS headlines."""
        headlines = []
        try:
            url = "https://cointelegraph.com/rss"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                xml_data = response.read()
                
            root = ET.fromstring(xml_data)
            for item in root.findall('./channel/item'):
                title = item.find('title')
                if title is not None and title.text:
                    headlines.append(title.text)
        except Exception as e:
            print(f"Failed to fetch live RSS: {e}")
            
        return headlines

    def generate_snapshot(self, current_time: pd.Timestamp = None, live: bool = False) -> SentimentSnapshot:
        """
        Generates a consolidated sentiment snapshot.
        If live=True, fetches live RSS feeds. Otherwise (or on failure), falls back to historical mock.
        """
        if live:
            headlines = self.fetch_live_headlines()
            if headlines:
                pos = 0
                neg = 0
                neu = 0
                score_sum = 0
                
                for h in headlines:
                    res = self.process_text(h)
                    if res['label'] == 'POSITIVE': pos += 1
                    elif res['label'] == 'NEGATIVE': neg += 1
                    else: neu += 1
                    score_sum += res['sentiment_score']
                    
                total = pos + neg + neu
                bullish_ratio = pos / total if total > 0 else 0
                bearish_ratio = neg / total if total > 0 else 0
                
                # prob_up and prob_down
                prob_up = min(1.0, bullish_ratio + random.uniform(0, 0.1))
                prob_down = min(1.0, bearish_ratio + random.uniform(0, 0.1))
                total_prob = prob_up + prob_down
                if total_prob > 0:
                    prob_up /= total_prob
                    prob_down /= total_prob
                    
                return SentimentSnapshot(
                    timestamp=pd.Timestamp.now() if current_time is None else current_time,
                    sentiment_score=score_sum / total if total > 0 else 0,
                    confidence=0.85, # Live heuristic
                    positive_count=pos,
                    negative_count=neg,
                    neutral_count=neu,
                    bullish_ratio=bullish_ratio,
                    bearish_ratio=bearish_ratio,
                    probability_up=max(0.0, prob_up),
                    probability_down=max(0.0, prob_down)
                )

        # Historical Mock generator
        pos = random.randint(10, 100)
        neg = random.randint(10, 100)
        neu = random.randint(5, 50)
        total = pos + neg + neu
        
        bullish_ratio = pos / total if total > 0 else 0
        bearish_ratio = neg / total if total > 0 else 0
        
        prob_up = min(1.0, bullish_ratio + random.uniform(-0.1, 0.1))
        prob_down = min(1.0, bearish_ratio + random.uniform(-0.1, 0.1))
        
        total_prob = prob_up + prob_down
        if total_prob > 0:
            prob_up /= total_prob
            prob_down /= total_prob
            
        score = (pos - neg) / total if total > 0 else 0
            
        return SentimentSnapshot(
            timestamp=pd.Timestamp.now() if current_time is None else current_time,
            sentiment_score=score,
            confidence=random.uniform(0.6, 0.95),
            positive_count=pos,
            negative_count=neg,
            neutral_count=neu,
            bullish_ratio=bullish_ratio,
            bearish_ratio=bearish_ratio,
            probability_up=max(0.0, prob_up),
            probability_down=max(0.0, prob_down)
        )
