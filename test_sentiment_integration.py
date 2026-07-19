import unittest
import pandas as pd
import numpy as np

from strategy.renko_theses import RenkoATRConsecutiveThesis
from sentiment.validator import SentimentValidationEngine, ValidationDecision
from sentiment.engine import SentimentSnapshot

class TestSentimentIntegration(unittest.TestCase):
    
    def test_sentiment_validator(self):
        validator = SentimentValidationEngine(prob_threshold=0.7)
        
        # Bullish snapshot
        snapshot_bullish = SentimentSnapshot(
            timestamp=pd.Timestamp.now(),
            sentiment_score=0.8,
            confidence=0.9,
            positive_count=100,
            negative_count=10,
            neutral_count=20,
            bullish_ratio=0.8,
            bearish_ratio=0.1,
            probability_up=0.75,
            probability_down=0.25
        )
        
        # Validate BUY with Bullish snapshot -> APPROVE
        decision = validator.validate_signal('BUY', snapshot_bullish)
        self.assertEqual(decision, ValidationDecision.APPROVE)
        
        # Validate SELL with Bullish snapshot -> REJECT
        decision = validator.validate_signal('SELL', snapshot_bullish)
        self.assertEqual(decision, ValidationDecision.REJECT)
        
    def test_renko_atr_consecutive(self):
        thesis = RenkoATRConsecutiveThesis(fast_period=2, slow_period=5)
        
        # Mock bricks
        # Directions: 1, 1, -1, -1, -1
        # Prices: 10, 11, 10, 9, 8
        
        df = pd.DataFrame({
            'brick_close_time': pd.date_range('2024-01-01', periods=5, freq='h'),
            'direction': [1, 1, -1, -1, -1],
            'brick_close': [10, 11, 10, 9, 8]
        })
        
        signals = thesis.generate_signals(df)
        
        self.assertIn('raw_signal', signals.columns)
        self.assertEqual(len(signals), 5)
        
if __name__ == '__main__':
    unittest.main()
