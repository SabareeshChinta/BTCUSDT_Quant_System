from sentiment.engine import SentimentSnapshot

class SentimentExitGuardian:
    """
    Monitors active positions and exits trades when sentiment reverses significantly.
    """
    def __init__(self, prob_exit_threshold: float = 0.80):
        self.prob_exit_threshold = prob_exit_threshold

    def check_exit(self, direction: int, snapshot: SentimentSnapshot) -> bool:
        """
        Returns True if the position should be exited immediately based on sentiment.
        direction: 1 for LONG, -1 for SHORT
        """
        if direction == 1:
            # We are LONG, but sentiment strongly points DOWN
            if snapshot.probability_down >= self.prob_exit_threshold:
                return True
            if snapshot.negative_count > snapshot.positive_count * 2: # Heavy bearishness
                return True
                
        elif direction == -1:
            # We are SHORT, but sentiment strongly points UP
            if snapshot.probability_up >= self.prob_exit_threshold:
                return True
            if snapshot.positive_count > snapshot.negative_count * 2: # Heavy bullishness
                return True
                
        return False
