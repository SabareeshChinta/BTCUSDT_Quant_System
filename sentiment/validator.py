from enum import Enum
from .engine import SentimentSnapshot

class ValidationDecision(Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    NEUTRAL = "NEUTRAL"

class SentimentValidationEngine:
    """
    Validates externally generated trade signals against sentiment data.
    """
    def __init__(self, prob_threshold: float = 0.6):
        self.prob_threshold = prob_threshold

    def validate_signal(self, raw_signal: str, snapshot: SentimentSnapshot) -> ValidationDecision:
        """
        Validate a raw signal (BUY/SELL) against the sentiment snapshot.
        raw_signal can be 'BUY', 'SELL', or 'NEUTRAL' (or 1, -1, 0)
        """
        signal_str = str(raw_signal).upper()
        
        # Convert numeric to string for uniform processing
        if signal_str == '1' or signal_str == '1.0':
            signal_str = 'BUY'
        elif signal_str == '-1' or signal_str == '-1.0':
            signal_str = 'SELL'
            
        if signal_str not in ['BUY', 'SELL']:
            return ValidationDecision.NEUTRAL
            
        if signal_str == 'BUY':
            # Bullish Confirmation: If sentiment probability for UP is high enough
            if snapshot.probability_up >= self.prob_threshold or snapshot.positive_count > snapshot.negative_count:
                return ValidationDecision.APPROVE
            # Bearish Rejection: If sentiment is heavily bearish
            elif snapshot.probability_down >= self.prob_threshold or snapshot.negative_count > snapshot.positive_count * 1.5:
                return ValidationDecision.REJECT
            else:
                return ValidationDecision.NEUTRAL
                
        if signal_str == 'SELL':
            # Bearish Confirmation
            if snapshot.probability_down >= self.prob_threshold or snapshot.negative_count > snapshot.positive_count:
                return ValidationDecision.APPROVE
            # Bullish Rejection
            elif snapshot.probability_up >= self.prob_threshold or snapshot.positive_count > snapshot.negative_count * 1.5:
                return ValidationDecision.REJECT
            else:
                return ValidationDecision.NEUTRAL
                
        return ValidationDecision.NEUTRAL
