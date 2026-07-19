import math

class WhaleSentimentEngine:
    """
    Whale Sentiment Engine.
    Assigns higher importance to influential sources.
    """
    
    @staticmethod
    def calculate_weight(followers: int, default_weight: float = 1.0) -> float:
        """
        Calculate weight based on follower count.
        Using logarithmic scaling: weight = log(followers + 1)
        """
        if followers < 0:
            return default_weight
        # Log base 10 of followers + 1, plus base weight so small accounts aren't 0
        return math.log10(followers + 1) + default_weight

    @staticmethod
    def apply_whale_weight(sentiment_score: float, followers: int) -> float:
        """
        Apply whale weight to a raw sentiment score.
        """
        weight = WhaleSentimentEngine.calculate_weight(followers)
        return sentiment_score * weight
