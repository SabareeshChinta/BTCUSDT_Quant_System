import pandas as pd
from typing import List, Dict, Optional

class EventDetectionEngine:
    """
    Event Detection Engine.
    Detects macro events like FOMC, CPI, etc., to reduce risk or trigger alerts.
    """
    def __init__(self):
        # In a real system, this would load an economic calendar from an API.
        self.known_events: List[Dict] = [
            {"event_type": "FOMC", "date": "2024-01-31", "impact": "HIGH"},
            {"event_type": "CPI", "date": "2024-02-13", "impact": "HIGH"},
            # Mock data
        ]
        
    def check_for_events(self, current_time: pd.Timestamp, window_hours: int = 24) -> Optional[Dict]:
        """
        Check if there is a major event within the specified window.
        """
        # Simplistic mock matching for demonstration
        for event in self.known_events:
            event_date = pd.to_datetime(event["date"]).tz_localize(current_time.tz)
            time_diff = abs((event_date - current_time).total_seconds()) / 3600
            
            if time_diff <= window_hours:
                return event
                
        return None
        
    def get_risk_multiplier(self, current_time: pd.Timestamp) -> float:
        """
        Returns a risk multiplier (e.g., 0.5 to halve position size) if near a major event.
        Returns 1.0 if no event.
        """
        event = self.check_for_events(current_time, window_hours=12) # Halve risk 12 hours before/after
        if event:
            if event["impact"] == "HIGH":
                return 0.5
            elif event["impact"] == "MEDIUM":
                return 0.75
        return 1.0
